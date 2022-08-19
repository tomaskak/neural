from ..algos.algo import Algo
from ..model.model import Model, NormalModel
from ..model.layer import Layer
from ..model.layer_factory import to_layers, env_to_in_out_sizes
from ..util.exp_replay import ExpReplay
from ..tools.timer import timer
from ..algos.sac_worker import SACWorker
from ..util.process_orchestrator import ProcessOrchestrator, Work, WorkerSpace

from copy import deepcopy

import torch
import numpy as np


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
        self._multi_process = training_params.get("multiprocess", False)

        for model in SoftActorCritic.required_model_defs:
            assert model in layers, f"definition for {model} not provided in {layers}"
        assert len(SoftActorCritic.required_model_defs) == len(
            layers
        ), f"Unexpected model defined, expected={required_model_defs}"

        in_size, out_size = env_to_in_out_sizes(env)

        self._actor = NormalModel(
            "actor", to_layers(in_size, out_size, layers["actor"]), None  # graph_sink
        )

        self._q_1 = Model("q_1", to_layers(in_size, out_size, layers["q_1"]), None)

        self._q_2 = Model(
            "q_2",
            to_layers(in_size, out_size, layers["q_2"]),
            None,
        )

        self._value = Model(
            "value",
            to_layers(in_size, out_size, layers["value"]),
            None,
        )

        self._target_value = Model("target_value", [], None)
        self._target_value._layers = deepcopy(self._value.layers())

        self._gamma = hypers.get("future_reward_discount", 0.99)
        self._q_lr = hypers.get("q_lr", 0.001)
        self._v_lr = hypers.get("v_lr", 0.001)
        self._actor_lr = hypers.get("actor_lr", 0.001)
        self._alpha = hypers.get("target_update_step", 0.001)
        self._replay_size = hypers.get("experience_replay_size", 1000 * 1000)
        self._mini_batch_size = hypers.get("minibatch_size", 64)

        self._actor_optim = torch.optim.Adam(self._actor.parameters(), self._actor_lr)
        self._q_1_optim = torch.optim.Adam(self._q_1.parameters(), self._q_lr)
        self._q_2_optim = torch.optim.Adam(self._q_2.parameters(), self._q_lr)
        self._value_optim = torch.optim.Adam(self._value.parameters(), self._v_lr)

        self._init_replay_buffer()

        self._q_1_loss_fn = torch.nn.MSELoss()
        self._q_2_loss_fn = torch.nn.MSELoss()
        self._value_loss_fn = torch.nn.MSELoss()
        self._actor_loss_fn = lambda x, y: (x - y).mean()

        self._actor.share_memory()
        self._q_1.share_memory()
        self._q_2.share_memory()
        self._value.share_memory()
        self._target_value.share_memory()

        self._actor.to(self._device)
        self._q_1.to(self._device)
        self._q_2.to(self._device)
        self._value.to(self._device)
        self._target_value.to(self._device)

        if self._multi_process:
            worker = SACWorker(in_size, out_size, self._mini_batch_size)
            init = worker.init_shared_worker_space(
                num_workers=8,
                actor={"model": self._actor, "optim": self._actor_optim},
                q_1={"model": self._q_1, "optim": self._q_1_optim},
                q_2={"model": self._q_2, "optim": self._q_2_optim},
                value={"model": self._value, "optim": self._value_optim},
                target={"model": self._target_value},
            )
            work_defs = [
                Work(provides="actor", fn=SACWorker.actor, args=(self._device,)),
                Work(
                    provides="q_1",
                    fn=SACWorker.q,
                    args=("1", self._gamma, self._device),
                ),
                Work(
                    provides="q_2",
                    fn=SACWorker.q,
                    args=("2", self._gamma, self._device),
                ),
                Work(
                    provides="value",
                    fn=SACWorker.value,
                    args=(self._alpha, self._device),
                ),
            ]
            self._process_orchestrator = ProcessOrchestrator(init, work_defs)

    def _init_replay_buffer(self):
        in_size, out_size = env_to_in_out_sizes(self._env)
        state_spec = ("f8", (in_size,))
        action_spec = ("f8", (out_size,))
        self._replay_buffer = ExpReplay(
            self._replay_size, [state_spec, action_spec, float, state_spec, float]
        )

    def load(self, settings):
        self._actor.load_state_dict(settings["actor"])
        self._q_1.load_state_dict(settings["q_1"])
        self._q_2.load_state_dict(settings["q_2"])
        self._value.load_state_dict(settings["value"])
        self._target_value.load_state_dict(settings["target_value"])

        self._actor_optim.load_state_dict(settings["actor_optim"])
        self._q_1_optim.load_state_dict(settings["q_1_optim"])
        self._q_2_optim.load_state_dict(settings["q_2_optim"])
        self._value_optim.load_state_dict(settings["value_optim"])

        hypers = settings["hyperparameters"]
        self._gamma = hypers["future_reward_discount"]
        self._q_lr = hypers["q_lr"]
        self._v_lr = hypers["v_lr"]
        self._actor_lr = hypers["actor_lr"]
        self._alpha = hypers["target_update_step"]
        self._replay_size = hypers["experience_replay_size"]
        self._mini_batch_size = hypers["minibatch_size"]

        self._init_replay_buffer()

    def save(self):
        return {
            "actor": self._actor.state_dict(),
            "q_1": self._q_1.state_dict(),
            "q_2": self._q_2.state_dict(),
            "value": self._value.state_dict(),
            "actor_optim": self._actor_optim.state_dict(),
            "q_1_optim": self._q_1_optim.state_dict(),
            "q_2_optim": self._q_2_optim.state_dict(),
            "value_optim": self._value_optim.state_dict(),
            "target_value": self._target_value.state_dict(),
            "hyperparameters": {
                "future_reward_discount": self._gamma,
                "q_lr": self._q_lr,
                "v_lr": self._v_lr,
                "actor_lr": self._actor_lr,
                "target_update_step": self._alpha,
                "experience_replay_size": self._replay_size,
                "minibatch_size": self._mini_batch_size,
            },
        }

    @classmethod
    def defined_hyperparams(cls) -> dict:
        return SoftActorCritic.defined_hypers

    def train(self):
        EPISODES = self._training_params.get("episodes_per_training", 10)
        STEPS = self._training_params.get("max_steps", 200)
        ALL_UPDATE = self._training_params.get("steps_between_updates", 1)

        total_steps = 0
        for episode in range(EPISODES):
            observation = self._env.reset()
            for step in range(STEPS):
                self._actor.to("cpu")
                total_steps += 1
                action = None
                with torch.no_grad():
                    actions, log_probs = self._actor.forward(
                        torch.tensor(np.array([observation]), device="cpu").float()
                    )

                next_observation, reward, done, info = self._env.step(
                    actions[0].numpy()
                )

                self._replay_buffer.push(
                    [
                        observation,
                        actions,
                        reward,
                        next_observation,
                        0.0 if done else 1.0,
                    ]
                )

                observation = next_observation

                if done or step == STEPS - 1:
                    break

                if (
                    len(self._replay_buffer) > self._mini_batch_size * 2
                    and total_steps % ALL_UPDATE == 0
                ):
                    batch = self._replay_buffer.sample(self._mini_batch_size)
                    self._actor.to(self._device)
                    self.update_all(batch)

    def update_all(self, batch):

        with timer("minibatch-setup"):
            states, actions, rewards, next_states, dones = batch

            device = self._device
            if self._multi_process:
                device = "cpu"

            # Move batch items to device, cast to float32
            states = torch.tensor(states.astype(np.float32), device=device)
            actions = torch.tensor(actions.astype(np.float32), device=device)
            rewards = torch.tensor(
                rewards.astype(np.float32), device=device
            ).reshape(-1, 1)
            next_states = torch.tensor(
                next_states.astype(np.float32), device=device
            )
            dones = torch.tensor(dones.astype(np.float32), device=device).reshape(
                -1, 1
            )

            if self._multi_process:
                data = WorkerSpace.data
                data["mb:states"][:] = states[:]
                data["mb:actions"][:] = actions[:]
                data["mb:rewards"][:] == rewards[:]
                data["mb:next_states"][:] = next_states[:]
                data["mb:dones"][:] = dones[:]
                data["mb:state_actions"][:] = torch.cat((states, actions), 1)[:]

        if self._multi_process:
            with timer("process_orchestrator"):
                self._process_orchestrator.execute()
                return

        with timer("minibatch-forward"):
            self._q_1.zero_grad()
            self._q_2.zero_grad()
            self._actor.zero_grad()
            self._value.zero_grad()
            self._target_value.zero_grad()

            state_actions = torch.cat((states, actions), dim=1)

            # Everything before this line should be done at replay buffer loading time or minibatch creation.

            predicted_values = self._value.forward(states)
            predicted_q_1s = self._q_1.forward(state_actions)
            predicted_q_2s = self._q_2.forward(state_actions)

            new_actions, log_probs = self._actor.forward(states)

            new_state_actions = torch.cat((states, new_actions), 1)

            predicted_new_qs = torch.minimum(
                self._q_1.forward(new_state_actions),
                self._q_2.forward(new_state_actions),
            )

            # Q updates
            target_next_state_values = self._target_value.forward(next_states)
            target_qs = rewards + dones * self._gamma * target_next_state_values
            target_qs = target_qs.detach().reshape(-1, 1)

        with timer("minibatch-backward"):
            q_1_loss = self._q_1_loss_fn(predicted_q_1s, target_qs.float())
            q_2_loss = self._q_2_loss_fn(predicted_q_2s, target_qs.float())

            q_1_loss.backward()
            q_2_loss.backward()

            # Value update
            target_vs = predicted_new_qs.detach() - log_probs.detach().reshape(-1, 1)
            v_loss = self._value_loss_fn(predicted_values, target_vs.float())

            v_loss.backward()

            # Policy update
            loss = self._actor_loss_fn(log_probs.reshape(-1, 1), predicted_new_qs)
            loss.backward()

        with timer("minibatch-update-grads"):
            self._actor_optim.step()
            self._q_1_optim.step()
            self._q_2_optim.step()
            self._value_optim.step()
            with torch.no_grad():
                for target, update in zip(
                    self._target_value.parameters(), self._value.parameters()
                ):
                    target.copy_(self._alpha * update + (1.0 - self._alpha) * target)

    def test(self, render=False) -> dict:
        result = {"average": 0, "max": None, "min": None}

        STEPS = self._training_params.get("max_steps", 200)
        EPISODES = self._training_params.get("episodes_per_test", 10)

        total_reward = 0
        self._actor.to("cpu")
        for i in range(EPISODES):
            observation = self._env.reset()
            current_total_reward = 0
            for step in range(STEPS):
                action = None
                with torch.no_grad():
                    actions, log_probs = self._actor.forward(
                        torch.tensor(np.array([observation]), device="cpu").float()
                    )

                    next_observation, reward, done, info = self._env.step(
                        actions[0].numpy()
                    )
                    current_total_reward += reward

                    if render and i == EPISODES - 1:
                        self._env.render()

                    if done or step == STEPS - 1:
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
        result["average"] = total_reward / EPISODES

        return result
