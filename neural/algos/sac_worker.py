from copy import deepcopy
from neural.util.process_orchestrator import Init

import numpy as np
import torch


def make_sac_init_defs(in_size: int, out_size: int, batch_size: int):
    MINI_BATCH_PREFIX = "mb:"
    default_type = np.float32
    defs = []

    defs.append((MINI_BATCH_PREFIX + "states", default_type, in_size * batch_size))
    defs.append((MINI_BATCH_PREFIX + "next_states", default_type, in_size * batch_size))
    defs.append(
        (
            MINI_BATCH_PREFIX + "state_actions",
            default_type,
            (in_size + out_size) * batch_size,
        )
    )
    defs.append((MINI_BATCH_PREFIX + "actions", default_type, out_size * batch_size))
    defs.append((MINI_BATCH_PREFIX + "rewards", default_type, batch_size))
    defs.append((MINI_BATCH_PREFIX + "dones", default_type, batch_size))
    defs.append(("actor_forward", default_type, out_size * batch_size))
    defs.append(("q_1_forward", default_type, batch_size))
    defs.append(("q_2_forward", default_type, batch_size))
    defs.append(("value_forward", default_type, batch_size))
    defs.append(("target_value_forward", default_type, batch_size))
    defs.append(("log_probs", default_type, batch_size))
    defs.append(("new_qs", default_type, batch_size))
    defs.append(("target_qs", default_type, batch_size))

    size_needed = 0
    for elem in defs:
        size_needed += elem[2] * np.dtype(elem[1]).itemsize

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
        self, num_workers: int, actor, q_1, q_2, value, target
    ):
        """
        model arguments must be picklable as they will be sent to new processes
        """
        MODEL_PREFIX = "model:"
        args = [
            (MODEL_PREFIX + "actor", actor),
            (MODEL_PREFIX + "q_1", q_1),
            (MODEL_PREFIX + "q_2", q_2),
            (MODEL_PREFIX + "value", value),
            (MODEL_PREFIX + "target", target),
        ]
        return Init(num_workers, self._shmem_size, deepcopy(self._init_defs), args)

    @property
    def shared_memory_size(self):
        return self._shmem_size

    @staticmethod
    def actor_forward(device, provides):
        data = WorkerSpace.data
        actor = data["model:actor"]
        states = torch.from_numpy(data["states"]).to(device)
        new_actions, log_probs = actor.forward(states)
        data["actor_forward"][:] = new_actions[:]
        data["log_probs"][:] = log_probs[:]

        new_state_actions = torch.cat((states, new_actions), 1)

        q_1 == data["model:q_1"]
        q_2 == data["model:q_2"]

        new_qs = torch.minimum(
            q_1.forward(new_state_actions), q_2.forward(new_state_actions)
        )
        data["new_qs"][:] = new_qs[:]

        return provides

    @staticmethod
    def model_forward(model_name, mb_input_name, output_key, device, provides):
        data = WorkerSpace.data
        model = data[model_name]

        X = torch.from_numpy(data[mb_input_name]).to(device)
        out = model.forward(X)
        data[output_key][:] = out[:]

        return provides

    @staticmethod
    def backwards_and_update_value(TAU, provides):
        data = WorkerSpace.data
        value = data["model:value"]
        target = data["model:target"]

        value_optimizer = data["optim:value"]
        loss_fn = data["loss:value"]

        new_qs = torch.from_numpy(data["new_qs"]).to(device)
        log_probs = torch.from_numpy(data["log_probs"]).to(device)
        predicted_values = torch.from_numpy(data["value_forward"]).to(device)

        loss = loss_fn(predicted_values, new_qs - log_probs)
        loss.backward()

        value_optimizer.step()

        with torch.no_grad():
            for target, update in zip(target.parameters(), value.parameters()):
                target.copy_(TAU * update + (1.0 - TAU) * target)

        return provides
