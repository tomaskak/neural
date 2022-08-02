import torch

from copy import deepcopy
from neural.tools.sink import SinkClient


class Model(torch.nn.Module):
    def __init__(self, name, layers, sink=None):
        super().__init__()
        self._metrics_on = False
        self._name = name
        self._layers = torch.nn.ModuleList(layers)
        self._sink = sink

        if self._sink is not None:
            for l in self._layers:
                l.hook_data_sink(SinkClient(self._name, self._sink))
        else:
            print(f"Model: sink is None, not enabling metrics")

    def forward(self, x):
        next_input = x
        for l in self._layers:
            next_input = l.forward(next_input)
        return next_input

    def layers(self):
        return self._layers

    # TODO: Remove global setting for each model
    def metrics_on(self):
        self._sink.enable()

    def metrics_off(self):
        self._sink.disable()

    def set_sink_hooks(self):
        if self._sink is not None:
            for l in self._layers:
                l.hook_data_sink(SinkClient(self._name, self._sink))
        else:
            print(f"Model: sink is None, not enabling metrics")
