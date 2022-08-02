from plotly.graph_objects import Figure, Scatter, Histogram
import torch

_COLORS = [
    "0,0,255",
    "0,255,0",
    "255,0,0",
    "255,255,0",
    "255,0,255",
    "0,255,255",
    "200,50,50",
    "50,200,50",
    "50,50,200",
]


def make_graph(type_name):
    if type_name == "activation":
        return ActivationGraph()
    elif type_name == "loss":
        return LossGraph()
    elif type_name == "reward":
        return RewardGraph()
    elif type_name == "gradient":
        return GradientGraph()
    else:
        raise ValueError(f"'{type_name}' not supported in make_graph fn")


class ActivationGraph:
    """
    Manages a set of layers for a graph of activation values.

    Each layer should have the same number of values otherwise a
    default will be inserted for missing data points.
    """

    def __init__(self):
        self._data = {}
        self._agg = {}
        self._colors = {}

        self._FLUSH_LENGTH = 200

        self._next_color_idx = 0

    def _gen_next_rgb(self):
        global _COLORS
        color = _COLORS[self._next_color_idx]
        self._next_color_idx += 1
        if self._next_color_idx >= len(_COLORS):
            self._next_color_idx = 0
        return color

    def push(self, layer_name, data):
        """
        data = {
            'layer_name': {
                'mean': array,
                'std_dev': array,
            },
            ...
        }
        each layer's mean and std_dev should be equal length as the x-axis
        will be implied.
        """

        if layer_name not in self._data:
            self._data[layer_name] = []
            self._agg[layer_name] = {
                "98": [],
                "02": [],
                "plus_std_dev": [],
                "sub_std_dev": [],
            }
            self._colors[layer_name] = self._gen_next_rgb()

        self._data[layer_name].append(data)

        if len(self._data[layer_name]) == self._FLUSH_LENGTH:
            outputs = torch.stack(self._data[layer_name])
            q = torch.tensor([0.02, 0.98])
            quantiles = torch.quantile(outputs, q)

            std_dev = torch.std(outputs).item()

            agg = self._agg[layer_name]
            agg["02"] += [quantiles[0]]
            agg["98"] += [quantiles[1]]
            agg["plus_std_dev"] += [std_dev / 1.0]
            agg["sub_std_dev"] += [std_dev / -1.0]

            self._data[layer_name].clear()

    def render(self):
        fig = Figure()

        # Get the shortest set of data so the x-axis can be defined to this
        # length.
        # TODO: Accept variable lengths with default values to fill in values.

        x_len = None
        for layer, values in self._agg.items():
            if x_len is None or x_len > len(values["98"]):
                x_len = len(values["98"])
        if x_len is None:
            x_len = 0

        x_axis = list(range(x_len))
        for layer, values in self._agg.items():
            color = self._colors[layer]

            fig.add_trace(
                Scatter(
                    x=x_axis,
                    y=values["98"][:x_len],
                    mode="lines+markers",
                    line_color="rgba(" + color + ",1.0)",
                    name=layer + "-.98",
                )
            )
            fig.add_trace(
                Scatter(
                    x=x_axis,
                    y=values["02"][:x_len],
                    mode="lines+markers",
                    line_color="rgba( " + color + ",1.0)",
                    name=layer + "-.02",
                )
            )
            fig.add_trace(
                Scatter(
                    x=x_axis + x_axis[::-1],
                    y=values["plus_std_dev"][:x_len]
                    + values["sub_std_dev"][:x_len][::-1],
                    fill="toself",
                    fillcolor="rgba(" + color + ",0.15)",
                    line_color="rgba(0,0,0,0.0)",
                    name=layer,
                )
            )
        fig.update_layout(
            title="Quantile-Activations",
            xaxis_title="update",
            yaxis_title="98/02 +- std_dev",
        )
        return fig


class LossGraph:
    """
    Manages a single loss value.
    """

    def __init__(self):
        self._data = []
        self._agg = {"mean": [], "plus_std_dev": [], "sub_std_dev": []}

        self._FLUSH_LENGTH = 100

    def push(self, name, data):
        """
        data should be a list of numbers.

        agg = {
            'mean': array,
            'std_dev': array,
        }
        """

        self._data.append(data)

        if len(self._data) == self._FLUSH_LENGTH:

            losses = torch.stack(self._data)

            std_dev = torch.std(losses).item()
            mean = losses.mean().item()

            agg = self._agg
            agg["mean"] += [mean]
            agg["plus_std_dev"] += [mean + std_dev]
            agg["sub_std_dev"] += [mean - std_dev]

            self._data.clear()

    def render(self):
        fig = Figure()

        # Get the shortest set of data so the x-axis can be defined to this
        # length.
        # TODO: Accept variable lengths with default values to fill in values.

        x_len = len(self._agg["mean"])
        x_axis = list(range(x_len))

        values = self._agg
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values["mean"],
                mode="lines+markers",
                line_color="rgba(255,0,0,1.0)",
                name="avg",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis + x_axis[::-1],
                y=values["plus_std_dev"] + values["sub_std_dev"][::-1],
                fill="toself",
                fillcolor="rgba(255,0,0,0.15)",
                line_color="rgba(0,0,0,0.0)",
                name="std_dev",
            )
        )
        fig.update_layout(
            title="Loss",
            xaxis_title="updates",
            yaxis_title="average +- std_dev",
        )
        return fig


class RewardGraph:
    """
    Graphs the average reward
    """

    def __init__(self):
        global _COLORS
        self._data = []
        self._agg = {
            "mean": [],
            ".90": [],
            ".10": [],
            "min": [],
            "max": [],
            "plus_std_dev": [],
            "sub_std_dev": [],
        }
        self._colors = {
            "mean": _COLORS[0],
            ".90": _COLORS[1],
            ".10": _COLORS[1],
            "min": _COLORS[2],
            "max": _COLORS[2],
        }

        # TODO: Group rewards, they should be measured per test run so the pusher needs to indicate when to flush.
        self._FLUSH_LENGTH = 5

        self._next_color_idx = 0

    def _gen_next_rgb(self):
        global _COLORS
        color = _COLORS[self._next_color_idx]
        self._next_color_idx += 1
        if self._next_color_idx >= len(_COLORS):
            self._next_color_idx = 0
        return color

    def push(self, name, data):
        """
        data should be a list of numbers each indicating the cumulative reward.
        """

        self._data.append(data)

        if len(self._data) == self._FLUSH_LENGTH:
            # print(f"aggregating reward data {self._data} into {self._agg}")

            rewards = torch.tensor(self._data, dtype=float)

            std_dev = torch.std(rewards).item()
            mean = rewards.mean().item()

            q = torch.tensor([0.1, 0.9], dtype=float)
            quantiles = torch.quantile(rewards, q)

            min = torch.min(rewards)
            max = torch.max(rewards)

            agg = self._agg
            agg["mean"] += [mean]
            agg["plus_std_dev"] += [mean + std_dev]
            agg["sub_std_dev"] += [mean - std_dev]
            agg[".90"] += [quantiles[1]]
            agg[".10"] += [quantiles[0]]
            agg["min"] += [min]
            agg["max"] += [max]

            self._data.clear()

    def render(self):
        fig = Figure()

        # Get the shortest set of data so the x-axis can be defined to this
        # length.
        # TODO: Accept variable lengths with default values to fill in values.

        x_len = len(self._agg["mean"])
        x_axis = list(range(x_len))

        values = self._agg
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values["mean"],
                mode="lines+markers",
                line_color="rgba(" + self._colors["mean"] + ",1.0)",
                name="avg",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values[".90"],
                mode="lines+markers",
                line_color="rgba(" + self._colors[".90"] + ",1.0)",
                name=".90",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values[".10"],
                mode="lines+markers",
                line_color="rgba(" + self._colors[".10"] + ",1.0)",
                name=".10",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values["max"],
                mode="lines+markers",
                line_color="rgba(" + self._colors["max"] + ",1.0)",
                name="max",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis,
                y=values["min"],
                mode="lines+markers",
                line_color="rgba(" + self._colors["min"] + ",1.0)",
                name="min",
            )
        )
        fig.add_trace(
            Scatter(
                x=x_axis + x_axis[::-1],
                y=values["plus_std_dev"] + values["sub_std_dev"][::-1],
                fill="toself",
                fillcolor="rgba(" + self._colors["mean"] + ",0.15)",
                line_color="rgba(0,0,0,0.0)",
                name="std_dev",
            )
        )
        fig.update_layout(
            title="Reward",
            xaxis_title="updates",
            yaxis_title="[average+-std_dev/min/max/.90/.10]",
        )
        return fig


class GradientGraph:
    """
    Manages a set of layers for a histogram of gradient values.

    Each layer should have the same number of values.
    """

    def __init__(self):
        self._agg = {}
        self._colors = {}

        self._NUM_GRADS = 50

        self._next_color_idx = 0

    def _gen_next_rgb(self):
        global _COLORS
        color = _COLORS[self._next_color_idx]
        self._next_color_idx += 1
        if self._next_color_idx >= len(_COLORS):
            self._next_color_idx = 0
        return color

    def push(self, layer_name, data):
        """
        data = {
            'layer_name': {
                'values': [[] * NUM_GRADS]
            },
            ...
        }
        each layers will hold NUM_GRADS number of sets of gradients that will correspond to the last NUM_GRADS pushed and use all of these values to plot the histogram.
        """

        if layer_name not in self._agg:
            self._agg[layer_name] = [[] * self._NUM_GRADS]
            self._colors[layer_name] = self._gen_next_rgb()

        self._agg[layer_name].append(data)
        self._agg[layer_name].pop(0)

    def render(self):
        fig = Figure()

        # Get the shortest set of data so the x-axis can be defined to this
        # length.
        # TODO: Accept variable lengths with default values to fill in values.

        for layer, values in self._agg.items():
            all_values = []
            for list_of_values in values:
                for val in list_of_values:
                    all_values.append(val)

            color = self._colors[layer]

            fig.add_trace(
                Histogram(
                    x=all_values,
                    opacity=0.75,
                    name=layer,
                    histnorm="percent",
                    xbins=dict(start=-0.25, end=0.25, size=0.0001),
                )
            )
        fig.update_layout(title="Gradients", barmode="overlay")
        return fig
