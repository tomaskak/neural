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


class Layer(torch.nn.Module):
    def __init__(self, name, mod, sink_client=None):
        super().__init__()
        self._name = name
        self._mod = mod
        self._sink = sink_client

    def hook_data_sink(self, sink_client):
        self._sink = sink_client
        # self._sink.init_namespace(self._name, 'layer_output')

    def forward(self, x):
        out = self._mod.forward(x)
        # self._sink.push(self._name, "layer_output", out.detach())
        return out

    def __deepcopy__(self, memo):
        return Layer(deepcopy(self._name, memo), deepcopy(self._mod, memo), self._sink)
