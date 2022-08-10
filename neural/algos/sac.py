from ..algos.algo import Algo
from ..model.model import Model
from ..model.layer import Layer
from ..model.layer_factory import to_layers, env_to_in_out_sizes
from ..util.exp_replay import ExpReplay
from ..tools.timer import timer

from copy import deepcopy

import torch
import numpy as np


def get_action(norm_params, LOW, HIGH, device):
    """
    Get the action values from the norm_params passed in.

    norm_params can represent multiple sets of actions, but each set must
    be even as each pair of elements in the set are assumed to be a mu and
    sigma in a normal distribution.

    Each mu and sigma will be used to sample a normal distribution to get an
    action value. Additionally the log probability of the actions selected will
    be returned.

    Returns: [ [action0, action1, ..., actionN], ... ] ,
             [ log(P(action0)) + log(P(action1)) ... + log(P(actionN)), ... ]
    """

    actions = []
    log_prob = None
    # Iterate over each pair of mu and sigma.
    for i in range(norm_params.shape[1] // 2):
        first = i * 2
        mu = norm_params[:, first]
        log_sigma = norm_params[:, first + 1]
        std_normal = torch.distributions.normal.Normal(
            torch.tensor(0).float().to(device), torch.tensor(1).float().to(device)
        )
        z = std_normal.sample(sample_shape=mu.shape)
        sigma = torch.exp(log_sigma)

        action = torch.clamp((mu + sigma), min=LOW, max=HIGH)

        if log_prob is None:
            log_prob = torch.distributions.normal.Normal(mu, sigma).log_prob(action)
        else:
            log_prob += torch.distributions.normal.Normal(mu, sigma).log_prob(action)

        actions.append(action.reshape(-1, 1))

    return torch.cat(actions, dim=-1), log_prob


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

        for model in SoftActorCritic.required_model_defs:
            assert model in layers, f"definition for {model} not provided in {layers}"
        assert len(SoftActorCritic.required_model_defs) == len(
            layers
        ), f"Unexpected model defined, expected={required_model_defs}"

        in_size, out_size = env_to_in_out_sizes(env)

        self._actor = Model(
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

        state_spec = ("f8", (in_size,))
        action_spec = ("f8", (out_size,))
        self._replay_buffer = ExpReplay(
            self._replay_size, [state_spec, action_spec, float, state_spec, float]
        )

        self._q_1_loss_fn = torch.nn.MSELoss()
        self._q_2_loss_fn = torch.nn.MSELoss()
        self._value_loss_fn = torch.nn.MSELoss()
        self._actor_loss_fn = lambda x, y: (x - y).mean()

        self._actor.to(self._device)
        self._q_1.to(self._device)
        self._q_2.to(self._device)
        self._value.to(self._device)
        self._target_value.to(self._device)

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
                    norm_params = self._actor.forward(
                        torch.tensor(np.array([observation]), device="cpu").float()
                    )
                    # TODO: Use env's definition of max action values
                    action, _ = get_action(
                        norm_params, LOW=-1.0, HIGH=1.0, device="cpu"
                    )

                next_observation, reward, done, info = self._env.step(action[0].numpy())

                self._replay_buffer.push(
                    [
                        observation,
                        action,
                        reward,
                        next_observation,
                        0.0 if done else 1.0,
                    ]
                )

                observation = next_observation

                if done or step == STEPS - 1:
                    break

                if (
                    len(self._replay_buffer) > self._mini_batch_size * 5
                    and total_steps % ALL_UPDATE == 0
                ):
                    batch = self._replay_buffer.sample(self._mini_batch_size)
                    self._actor.to(self._device)
                    self.update_all(batch)

    def update_all(self, batch):

        with timer("minibatch-forward"):
            states, actions, rewards, next_states, dones = batch

            states = torch.tensor(states.astype(np.float32), device=self._device)
            actions = torch.tensor(actions.astype(np.float32), device=self._device)
            rewards = torch.tensor(rewards.astype(np.float32), device=self._device)
            next_states = torch.tensor(
                next_states.astype(np.float32), device=self._device
            )
            dones = torch.tensor(dones.astype(np.float32), device=self._device)

            self._q_1.zero_grad()
            self._q_2.zero_grad()
            self._actor.zero_grad()
            self._value.zero_grad()
            self._target_value.zero_grad()

            state_actions = torch.cat((states, actions), dim=1)

            predicted_values = self._value.forward(states)
            predicted_q_1s = self._q_1.forward(state_actions)
            predicted_q_2s = self._q_2.forward(state_actions)

            norm_params = self._actor.forward(states)
            new_actions, log_probs = get_action(
                norm_params, LOW=-1.0, HIGH=1.0, device=self._device
            )

            new_state_actions = torch.cat((states, new_actions), 1)

            predicted_new_qs = torch.minimum(
                self._q_1.forward(new_state_actions),
                self._q_2.forward(new_state_actions),
            )

            # Q updates
            target_next_state_values = self._target_value.forward(next_states).reshape(
                1, -1
            )
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
            with torch.no_grad():
                for params in self._q_1.parameters():
                    params -= self._q_lr * params.grad

                for params in self._q_2.parameters():
                    params -= self._q_lr * params.grad

                for params in self._value.parameters():
                    params -= self._v_lr * params.grad

                for params in self._actor.parameters():
                    # Subtract because we wish to minimize divergence.
                    params -= self._actor_lr * params.grad

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
                    norm_params = self._actor.forward(
                        torch.tensor(np.array([observation]), device="cpu").float()
                    )
                    action, _ = get_action(
                        norm_params, LOW=-1.0, HIGH=1.0, device="cpu"
                    )

                    next_observation, reward, done, info = self._env.step(
                        action[0].numpy()
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
