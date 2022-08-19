from copy import deepcopy
from ..util.process_orchestrator import Init, WorkerSpace
from torch.multiprocessing import Event
import time

import numpy as np
import torch


def make_sac_init_defs(in_size: int, out_size: int, batch_size: int):
    MINI_BATCH_PREFIX = "mb:"
    default_type = np.float32
    defs = []

    defs.append((MINI_BATCH_PREFIX + "states", default_type, (batch_size, in_size)))
    defs.append(
        (MINI_BATCH_PREFIX + "next_states", default_type, (batch_size, in_size))
    )
    defs.append(
        (
            MINI_BATCH_PREFIX + "state_actions",
            default_type,
            (batch_size, (in_size + out_size)),
        )
    )
    defs.append((MINI_BATCH_PREFIX + "actions", default_type, (batch_size, out_size)))
    defs.append((MINI_BATCH_PREFIX + "rewards", default_type, (batch_size, 1)))
    defs.append((MINI_BATCH_PREFIX + "dones", default_type, (batch_size, 1)))
    defs.append(("actor_forward", default_type, (batch_size, out_size)))
    defs.append(("q_1_forward", default_type, (batch_size, 1)))
    defs.append(("q_2_forward", default_type, (batch_size, 1)))
    defs.append(("value_forward", default_type, (batch_size, 1)))
    defs.append(("target_value_forward", default_type, (batch_size, 1)))
    defs.append(("log_probs", default_type, (batch_size,)))
    defs.append(("new_qs", default_type, (batch_size, 1)))
    defs.append(("target_qs", default_type, (batch_size, 1)))

    def _mult_shape(*shape):
        out = 1
        for dim in shape:
            out *= dim
        return out

    size_needed = 0
    for elem in defs:
        size_needed += _mult_shape(*elem[2]) * np.dtype(elem[1]).itemsize

    return defs, size_needed


class SACWorker:
    """
    Defines the worker methods and attributes for the soft-actor-critic algorithm.
    """

    def __init__(self, in_size: int, out_size: int, batch_size: int):
        self._in = in_size
        self._out = out_size

        self._init_defs, self._shmem_size = make_sac_init_defs(
            in_size, out_size, batch_size
        )

    def init_shared_worker_space(
        self,
        num_workers: int,
        actor: dict,
        q_1: dict,
        q_2: dict,
        value: dict,
        target: dict,
    ):
        """
        model arguments must be picklable as they will be sent to new processes
 
        all model params are expected to be dicts with model and optimizer under the "model" and "optim" keys
        if applicable.
        """
        MODEL_PREFIX = "model:"
        OPTIMIZER_PREFIX = "optim:"
        args = [
            (MODEL_PREFIX + "actor", actor["model"]),
            (MODEL_PREFIX + "q_1", q_1["model"]),
            (MODEL_PREFIX + "q_2", q_2["model"]),
            (MODEL_PREFIX + "value", value["model"]),
            (MODEL_PREFIX + "target", target["model"]),
            (OPTIMIZER_PREFIX + "actor", actor["optim"]),
            (OPTIMIZER_PREFIX + "q_1", q_1["optim"]),
            (OPTIMIZER_PREFIX + "q_2", q_2["optim"]),
            (OPTIMIZER_PREFIX + "value", value["optim"]),
            ("new_qs_done", Event()),
            ("q_1_done", Event()),
            ("q_2_done", Event()),
            ("value_and_target_done", Event()),
        ]
        return Init(num_workers, self._shmem_size, deepcopy(self._init_defs), args)

    @property
    def shared_memory_size(self):
        return self._shmem_size

    @staticmethod
    def actor(device, provides):
        # print(f"{provides} starting at {time.time()}")
        try:
            data = WorkerSpace.data
            actor = data["model:actor"]
            actor.zero_grad()

            states = torch.from_numpy(data["mb:states"]).to(device)
            new_actions, log_probs = actor.forward(states)

            data["actor_forward"][:] = new_actions.detach()[:]
            data["log_probs"][:] = log_probs.detach()[:]

            new_state_actions = torch.cat((states, new_actions), 1)

            q_1 = data["model:q_1"]
            q_2 = data["model:q_2"]

            new_qs = torch.minimum(
                q_1.forward(new_state_actions), q_2.forward(new_state_actions)
            )
            data["new_qs"][:] = new_qs.detach()[:]

            data["new_qs_done"].set()
            # print(f"new_qs_done set at {time.time()}")

            loss = (log_probs - new_qs).mean()
            loss.backward()

            data["optim:actor"].step()

        except Exception as e:
            raise Exception(f"exception in {provides}: [{e}]")
            raise e

        return provides

    @staticmethod
    def value(TAU, device, provides):
        # print(f"{provides} starting at {time.time()}")
        try:
            data = WorkerSpace.data
            value = data["model:value"]
            value.zero_grad()

            states = torch.from_numpy(data["mb:states"]).to(device)
            forward = value.forward(states)

            data["new_qs_done"].wait(timeout=5.0)
            # print(f"{provides} woke up at {time.time()}")
            if not data["new_qs_done"].is_set():
                raise Exception(f"timeout on new qs [{provides}]")

            log_probs = torch.from_numpy(data["log_probs"]).to(device)
            new_qs = torch.from_numpy(data["new_qs"]).to(device)

            targets = new_qs - log_probs.view(-1, 1)
            assert (
                forward.shape == targets.shape
            ), f"forward={forward.shape} and targets={targets.shape} don't match"
            loss = torch.nn.MSELoss()(forward, targets)
            loss.backward()

            data["optim:value"].step()

            with torch.no_grad():
                for target, update in zip(
                    data["model:target"].parameters(), value.parameters()
                ):
                    target.copy_(TAU * update + (1.0 - TAU) * target)

        except Exception as e:
            raise Exception(f"exception in {provides}: [{e}]")

        return provides

    @staticmethod
    def q(one_or_two, GAMMA, device, provides):
        # print(f"{provides} starting at {time.time()}")
        try:
            data = WorkerSpace.data
            q_model = data["model:q_1" if one_or_two == "1" else "model:q_2"]
            target = data["model:target"]

            target.requires_grad = False
            q_model.requires_grad = True
            q_model.zero_grad()
            q_forward = q_model.forward(
                torch.from_numpy(data["mb:state_actions"]).to(device)
            )

            rewards = torch.from_numpy(data["mb:rewards"]).to(device)
            dones = torch.from_numpy(data["mb:dones"]).to(device)
            target_next_state_values = target.forward(
                torch.from_numpy(data["mb:next_states"]).to(device)
            )

            target_qs = rewards + dones * GAMMA * target_next_state_values

            start = time.time()
            data["new_qs_done"].wait(timeout=5.0)
            # print(f"itme waiting in {provides}: {time.time() - start}")
            # print(f"{provides} woke up at {time.time()}")
            if not data["new_qs_done"].is_set():
                raise Exception(f"timeout on new qs [{provides}]")

            assert (
                target_qs.shape == q_forward.shape
            ), f"bad shapes target={target_qs.shape}, forwad={q_forward.shape}"
            loss = torch.nn.MSELoss()(q_forward, target_qs)
            loss.backward()

            data["optim:q_1" if one_or_two == "1" else "optim:q_2"].step()

        except Exception as e:
            raise Exception(f"exception in {provides}: [{e}]")

        return provides
