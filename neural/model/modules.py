import torch

from copy import deepcopy

# TODO: Move shared code to base class


class Activation(torch.nn.Module):
    def __init__(self, name, activation_fn, sink_client=None):
        super().__init__()
        self._name = name
        self._fn = activation_fn
        self._sink = sink_client

    def hook_data_sink(self, sink_client):
        self._sink = sink_client
        self._sink.init_namespace(self._name, "activation")

    def forward(self, x):
        out = self._fn(x)
        if self._sink is not None:
            self._sink.push(self._name, "activation", out.detach())
        return out

    def __deepcopy__(self, memo):
        return Activation(
            deepcopy(self._name, memo), deepcopy(self._fn, memo), self._sink
        )


class Loss:
    def __init__(self, name, loss_fn, graph_sink=None):
        self._name = name
        self._loss_fn = loss_fn
        self._sink = graph_sink

        if self._sink is not None:
            self._sink.init_namespace(self._name, "loss")

    def __call__(self, actual, target):
        loss = self._loss_fn(actual, target)
        if self._sink is not None:
            self._sink.push(self._name, "loss", loss.detach())
        return loss


class Env:
    def __init__(self, name, env, graph_sink=None):
        self._name = name
        self._env = env
        self._sink = graph_sink

        self._total_reward = 0.0

        if self._sink is not None:
            self._sink.init_namespace(self._name, "reward")

    def reset(self):
        self._sink.push(self._name, "reward", self._total_reward)
        self._total_reward = 0.0
        return self._env.reset()

    def step(self, *args, **kwargs):
        obs, reward, done, info = self._env.step(*args, **kwargs)
        self._total_reward += reward
        return obs, reward, done, info

    def render(self, *args, **kwargs):
        return self._env.render(*args, **kwargs)
