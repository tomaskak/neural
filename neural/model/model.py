import torch

from copy import deepcopy
from ..tools.sink import SinkClient


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

    def device(self):
        return next(self.parameters()).device.type

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


class NormalModel(Model):
    def __init__(self, name, layers, sink=None):
        super().__init__(name, layers, sink)

    def forward(self, X):
        output = super().forward(X)
        return self._get_actions(output)

    def _get_actions(self, norm_params):
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
        for i in range(len(norm_params[0]) // 2):
            first = i * 2
            mu = norm_params[:, first]
            log_sigma = norm_params[:, first + 1]
            std_normal = torch.distributions.normal.Normal(
                torch.tensor(0).float().to(self.device()),
                torch.tensor(1).float().to(self.device()),
            )
            z = std_normal.sample(sample_shape=mu.shape)
            sigma = torch.exp(log_sigma)

            action = mu + sigma * z

            if log_prob is None:
                log_prob = torch.distributions.normal.Normal(mu, sigma).log_prob(action)
            else:
                log_prob += torch.distributions.normal.Normal(mu, sigma).log_prob(
                    action
                )

            actions.append(action.reshape(-1, 1))

        return torch.cat(actions, dim=-1), log_prob
