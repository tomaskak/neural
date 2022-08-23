from ..algos.algo import Algo
from ..model.model import Model, NormalModel
from ..model.layer import Layer
from ..model.layer_factory import to_layers, env_to_in_out_sizes
from ..tools.timer import timer
from ..algos.sac_context import SACContext

from ..algos.sac_train import sac_train
from ..algos.sac_replay import sac_replay

from copy import deepcopy
from torch.multiprocessing import Queue, Process
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

        self.context.value = Model(
            "value",
            to_layers(in_size, out_size, layers["value"]),
            None,
        )

        self.context.target_value = deepcopy(self.context.value)
        self.context.target_value.requires_grad_(False)

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
        self.context.value_optim = torch.optim.Adam(
            self.context.value.parameters(), self._v_lr
        )

        self.context.q_1_loss_fn = torch.nn.MSELoss()
        self.context.q_2_loss_fn = torch.nn.MSELoss()
        self.context.value_loss_fn = torch.nn.MSELoss()
        self.context.actor_loss_fn = actor_loss

        self.context.update_shared()
        self.set_device_and_shmem()

    def set_device_and_shmem(self):
        self.context.shared.actor.to("cpu").share_memory()
        self.context.shared.q_1.to("cpu").share_memory()
        self.context.shared.q_2.to("cpu").share_memory()
        self.context.shared.value.to("cpu").share_memory()
        self.context.shared.target_value.to("cpu").share_memory()

        self.context.actor.to(self._device)
        self.context.q_1.to(self._device)
        self.context.q_2.to(self._device)
        self.context.value.to(self._device)
        self.context.target_value.to(self._device)

    def load(self, settings):
        self.context.actor.load_state_dict(settings["actor"])
        self.context.q_1.load_state_dict(settings["q_1"])
        self.context.q_2.load_state_dict(settings["q_2"])
        self.context.value.load_state_dict(settings["value"])
        self.context.target_value.load_state_dict(settings["target_value"])

        self.context.actor_optim.load_state_dict(settings["actor_optim"])
        self.context.q_1_optim.load_state_dict(settings["q_1_optim"])
        self.context.q_2_optim.load_state_dict(settings["q_2_optim"])
        self.context.value_optim.load_state_dict(settings["value_optim"])

        self.set_device_and_shmem()

        hypers = settings["hyperparameters"]
        hypers["max_action"] = hypers.get("max_action", 1.0)
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
            "value": self.context.shared.value.state_dict(),
            "actor_optim": self.context.shared.actor_optim.state_dict(),
            "q_1_optim": self.context.shared.q_1_optim.state_dict(),
            "q_2_optim": self.context.shared.q_2_optim.state_dict(),
            "value_optim": self.context.shared.value_optim.state_dict(),
            "target_value": self.context.shared.target_value.state_dict(),
            "hyperparameters": {
                "future_reward_discount": self._gamma,
                "q_lr": self._q_lr,
                "v_lr": self._v_lr,
                "actor_lr": self._actor_lr,
                "target_update_step": self._alpha,
                "experience_replay_size": self._replay_size,
                "minibatch_size": self._mini_batch_size,
                "max_action": self._hypers["max_action"],
            },
        }

    @classmethod
    def defined_hyperparams(cls) -> dict:
        return SoftActorCritic.defined_hypers

    def train(self):
        pass

    def start(self, render=False, save_hook=None, result_hook=None):
        replay_buf_q = Queue(maxsize=10000)
        next_batch_q = Queue()
        report_queue = Queue()

        replay_process = Process(
            name="replay",
            target=sac_replay,
            args=(
                replay_buf_q,
                next_batch_q,
                self._hypers,
                self._in_size,
                self._out_size,
                10,
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
                replay_buf_q,
            ),
        )

        replay_process.start()
        train_process.start()

        EPISODES = self._training_params.get("episodes_per_training", 10)
        STEPS = self._training_params.get("max_steps", 200)
        ALL_UPDATE = self._training_params.get("steps_between_updates", 1)

        SAVE_TIME = self._training_params.get("save_on_iteration", 100)

        iteration = 0
        time_limit_truncation = 0
        last_report = {"completed": 0}
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
                            actions = actions * rng
                            # actions = torch.clamp(actions * rng, min=-rng, max=rng)

                            next_observation, reward, done, info = self._env.step(
                                actions[0].numpy()
                            )
                            # print(f"info={info}")
                            done_value = 0.0 if done else 1.0
                            if done and info["TimeLimit.truncated"]:
                                time_limit_truncation +=1
                                done_value = 1.0

                            replay_buf_q.put(
                                (
                                    "EXP",
                                    [
                                        observation,
                                        actions,
                                        reward,
                                        next_observation,
                                        0.0 if done else 1.0,
                                    ],
                                )
                            )

                            observation = next_observation

                            if done or step == STEPS - 1:
                                break

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
                replay_buf_q.put(("TEST", {}))
                test_result = self.test(render)
                print(
                    f"training-iterations={last_report['completed']} results={test_result}, time_limit_reached={time_limit_truncation}"
                )
                if result_hook is not None:
                    result_hook(test_result)
            iteration += 1
            if iteration % SAVE_TIME == 0 and save_hook is not None:
                save_hook(self.save())

        replay_buf_q.put("STOP")
        next_batch_q.put("STOP")

        replay_process.join()
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
                        torch.tensor(np.array([observation]), device="cpu").float()
                    )

                    next_observation, reward, done, info = self._env.step(
                        actions[0].numpy()
                    )
                    current_total_reward += reward

                    if render and i == EPISODES - 1:
                        self._env.render()

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
        result["average"] = round(total_reward / EPISODES,2)
        result["average_per_step"] = round(total_reward/total_steps,2)
        result["average_steps_per_episode"] = round(total_steps / EPISODES,2)

        return result
