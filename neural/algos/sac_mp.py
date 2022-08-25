from ..algos.algo import Algo
from ..model.model import Model, NormalModel
from ..model.layer import Layer
from ..model.layer_factory import to_layers, env_to_in_out_sizes
from ..tools.timer import timer
from ..algos.sac_context import SACContext
from ..util.exp_replay import SharedBuffers

from ..algos.sac_train import sac_train
from ..algos.sac_replay import sac_replay_sample, sac_replay_store

from copy import deepcopy
from torch.multiprocessing import Queue, Process, Event
from queue import Empty
import torch
import numpy as np
import time


def actor_loss(x, y):
    return (x - y).mean()


class SoftActorCritic(Algo):
    """
    SoftActorCritic is an off-policy actor-critic algorithm that incorporates entropy
    maximization into its objective function leading to improved exploration characteristics.
    """

    defined_hypers = {
        "future_reward_discount": float,
        "q_lr": float,
        "v_lr": float,
        "actor_lr": float,
        "target_update_step": float,
        "experience_replay_size": int,
        "minibatch_size": int,
        "max_action": float,
        "target_entropy_weight": float,
    }

    # The algorithm has a model not listed here for 'target_value' but this by
    # definition must be the same configuration as 'value'.
    required_model_defs = ["actor", "q_1", "q_2", "value"]

    def __init__(self, hypers: dict, layers: list, training_params: dict, env):
        """
        layers is the definition for each model.

        layers = {
            'actor' : [ <list of layer defintions> ],
            ...
        }
        """
        super().__init__(hypers, training_params, env)
        self._device = training_params.get("device", "cpu")

        for model in SoftActorCritic.required_model_defs:
            assert model in layers, f"definition for {model} not provided in {layers}"

        assert len(SoftActorCritic.required_model_defs) == len(
            layers
        ), f"Unexpected model defined, expected={required_model_defs}"

        in_size, out_size = env_to_in_out_sizes(env)
        self._in_size = in_size
        self._out_size = out_size

        self.context = SACContext()

        self.context.actor = NormalModel(
            "actor", to_layers(in_size, out_size, layers["actor"]), None  # graph_sink
        )

        self.context.q_1 = Model(
            "q_1", to_layers(in_size, out_size, layers["q_1"]), None
        )

        self.context.q_2 = Model(
            "q_2",
            to_layers(in_size, out_size, layers["q_2"]),
            None,
        )

        self.context.q_1_target = deepcopy(self.context.q_1)
        self.context.q_2_target = deepcopy(self.context.q_2)
        self.context.q_1_target.requires_grad_(False)
        self.context.q_2_target.requires_grad_(False)

        self.context.entropy_weight = torch.tensor(
            [0.0], dtype=torch.float32, requires_grad=True
        )

        hypers["target_entropy_weight"] = hypers.get(
            "target_entropy_weight", -self._in_size
        )
        self._gamma = hypers.get("future_reward_discount", 0.99)
        self._q_lr = hypers.get("q_lr", 0.001)
        self._v_lr = hypers.get("v_lr", 0.001)
        self._actor_lr = hypers.get("actor_lr", 0.001)
        self._alpha = hypers.get("target_update_step", 0.001)
        self._replay_size = hypers.get("experience_replay_size", 1000 * 1000)
        self._mini_batch_size = hypers.get("minibatch_size", 64)

        self.context.actor_optim = torch.optim.Adam(
            self.context.actor.parameters(), self._actor_lr
        )
        self.context.q_1_optim = torch.optim.Adam(
            self.context.q_1.parameters(), self._q_lr
        )
        self.context.q_2_optim = torch.optim.Adam(
            self.context.q_2.parameters(), self._q_lr
        )
        self.context.entropy_weight_optim = torch.optim.Adam(
            [self.context.entropy_weight], self._actor_lr
        )

        self.context.q_1_loss_fn = torch.nn.MSELoss()
        self.context.q_2_loss_fn = torch.nn.MSELoss()
        self.context.actor_loss_fn = actor_loss

        self.context.update_shared()
        self.set_device_and_shmem()

    def set_device_and_shmem(self):
        self.context.shared.actor.to("cpu").share_memory()
        self.context.shared.q_1.to("cpu").share_memory()
        self.context.shared.q_2.to("cpu").share_memory()
        self.context.shared.q_1_target.to("cpu").share_memory()
        self.context.shared.q_2_target.to("cpu").share_memory()
        self.context.shared.entropy_weight.to("cpu").share_memory_()

    def load(self, settings):
        self.context.actor.load_state_dict(settings["actor"])
        self.context.q_1.load_state_dict(settings["q_1"])
        self.context.q_2.load_state_dict(settings["q_2"])
        self.context.q_1_target.load_state_dict(settings["q_1_target"])
        self.context.q_2_target.load_state_dict(settings["q_2_target"])

        self.context.actor_optim.load_state_dict(settings["actor_optim"])
        self.context.q_1_optim.load_state_dict(settings["q_1_optim"])
        self.context.q_2_optim.load_state_dict(settings["q_2_optim"])

        self.set_device_and_shmem()

        hypers = settings["hyperparameters"]
        hypers["max_action"] = hypers.get("max_action", 1.0)
        hypers["target_entropy_weight"] = hypers.get(
            "target_entropy_weight", -self._in_size
        )
        self._hypers = hypers
        self._gamma = hypers["future_reward_discount"]
        self._q_lr = hypers["q_lr"]
        self._v_lr = hypers["v_lr"]
        self._actor_lr = hypers["actor_lr"]
        self._alpha = hypers["target_update_step"]
        self._replay_size = hypers["experience_replay_size"]
        self._mini_batch_size = hypers["minibatch_size"]

    def save(self):
        return {
            "actor": self.context.shared.actor.state_dict(),
            "q_1": self.context.shared.q_1.state_dict(),
            "q_2": self.context.shared.q_2.state_dict(),
            "q_1_target": self.context.shared.q_1_target.state_dict(),
            "q_2_target": self.context.shared.q_2_target.state_dict(),
            "actor_optim": self.context.shared.actor_optim.state_dict(),
            "q_1_optim": self.context.shared.q_1_optim.state_dict(),
            "q_2_optim": self.context.shared.q_2_optim.state_dict(),
            "hyperparameters": {
                "future_reward_discount": self._gamma,
                "q_lr": self._q_lr,
                "actor_lr": self._actor_lr,
                "target_update_step": self._alpha,
                "experience_replay_size": self._replay_size,
                "minibatch_size": self._mini_batch_size,
                "max_action": self._hypers["max_action"],
                "target_entropy_weight": self._hypers["target_entropy_weight"],
            },
        }

    @classmethod
    def defined_hyperparams(cls) -> dict:
        return SoftActorCritic.defined_hypers

    def train(self):
        pass

    def start(self, render=False, save_hook=None, result_hook=None):
        replay_buf_q = Queue(maxsize=10000)
        next_batch_q = Queue(maxsize=200)
        dones_q = Queue()
        report_queue = Queue()
        start_sampling = Event()
        stop_sampling = Event()

        buffers = SharedBuffers(
            self._hypers["experience_replay_size"],
            20,
            "f",
            [self._in_size, self._out_size, 1, self._in_size, 1],
        )

        replay_store_process = Process(
            name="replay_store", target=sac_replay_store, args=(replay_buf_q, buffers)
        )

        replay_sample_process = Process(
            name="replay_sample",
            target=sac_replay_sample,
            args=(
                next_batch_q,
                dones_q,
                start_sampling,
                stop_sampling,
                self._hypers,
                buffers,
                self._device,
            ),
        )

        train_process = Process(
            name="trainer",
            target=sac_train,
            args=(
                self.context,
                next_batch_q,
                self._hypers,
                200,
                report_queue,
                dones_q,
            ),
        )

        replay_store_process.start()
        replay_sample_process.start()
        train_process.start()

        EPISODES = self._training_params.get("episodes_per_training", 10)
        STEPS = self._training_params.get("max_steps", 200)
        ALL_UPDATE = self._training_params.get("steps_between_updates", 1)

        SAVE_TIME = self._training_params.get("save_on_iteration", 100)

        iteration = 0
        time_limit_truncation = 0
        last_report = {"completed": 0}
        total_buf_writes = 0
        while True:
            with timer("explore-iteration"):
                for episode in range(EPISODES):
                    with timer("explore-episode"):
                        observation = self._env.reset()
                        for step in range(STEPS):
                            action = None
                            with torch.no_grad():
                                actions, log_probs = self.context.shared.actor.forward(
                                    torch.tensor(
                                        np.array([observation]), device="cpu"
                                    ).float()
                                )
                            rng = self._hypers["max_action"]

                            next_observation, reward, done, info = self._env.step(
                                actions[0].numpy() * rng
                            )

                            # Done value can be set if time limit is reached on an environment causing
                            # the model to think this is a valid termination case even though the timing
                            # is arbitrary and not a part of the MDP.
                            done_value = 0.0 if done else 1.0
                            if (
                                done
                                and info is not None
                                and "TimeLimit.truncated" in info
                                and info["TimeLimit.truncated"]
                            ):
                                time_limit_truncation += 1
                                done_value = 1.0

                            replay_buf_q.put(
                                (
                                    "EXP",
                                    [
                                        observation,
                                        actions[0].numpy(),
                                        reward,
                                        next_observation,
                                        0.0 if done else 1.0,
                                    ],
                                )
                            )
                            total_buf_writes += 1

                            observation = next_observation

                            if done or step == STEPS - 1:
                                break
                    if not start_sampling.is_set() and total_buf_writes > 2000:
                        print(f"start sampling!")
                        start_sampling.set()

            done = False
            try:
                while True:
                    last_report = report_queue.get_nowait()
                    print(f"training reported: {last_report}")
                done = (
                    last_report["completed"]
                    >= self._training_params["training_iterations"]
                )
            except Empty as e:
                pass

            with timer("test"):
                test_result = self.test(render)
                print(
                    f"training-iterations={last_report['completed']} results={test_result}, time_limit_reached={time_limit_truncation}"
                )
                print(
                    f"entropy_weight={self.context.shared.entropy_weight}, target={self._hypers['target_entropy_weight']}"
                )
                if result_hook is not None:
                    result_hook(test_result)
            iteration += 1
            if iteration % SAVE_TIME == 0 and save_hook is not None:
                save_hook(self.save())

        replay_buf_q.put("STOP")
        next_batch_q.put("STOP")
        stop_sampling.set()

        replay_sample_process.join()
        replay_store_process.join()
        train_process.join()

    def test(self, render=False) -> dict:
        result = {"average": 0, "max": None, "min": None}

        STEPS = self._training_params.get("max_steps", 200)
        EPISODES = self._training_params.get("episodes_per_test", 10)

        total_reward = 0
        total_steps = 0
        for i in range(EPISODES):
            observation = self._env.reset()
            current_total_reward = 0
            for step in range(STEPS):
                action = None
                with torch.no_grad():
                    actions, log_probs = self.context.shared.actor.forward(
                        torch.tensor(np.array([observation]), device="cpu").float(),
                        deterministic=False,
                    )

                    next_observation, reward, done, info = self._env.step(
                        actions[0].numpy()
                    )
                    current_total_reward += reward

                    if render:
                        self._env.render()

                    # print(f"reward={reward}, info={info}")

                    if done or step == STEPS - 1:
                        total_steps += step
                        total_reward += current_total_reward
                        if (
                            result["max"] is None
                            or current_total_reward > result["max"]
                        ):
                            result["max"] = current_total_reward
                        if (
                            result["min"] is None
                            or current_total_reward < result["min"]
                        ):
                            result["min"] = current_total_reward
                        break
                    observation = next_observation
        result["average"] = round(total_reward / EPISODES, 2)
        result["average_per_step"] = round(total_reward / total_steps, 2)
        result["average_steps_per_episode"] = round(total_steps / EPISODES, 2)

        return result
