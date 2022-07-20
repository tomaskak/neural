from plotly.graph_objects import Figure, Scatter
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
            agg["plus_std_dev"] += [std_dev / 2.0]
            agg["sub_std_dev"] += [std_dev / -2.0]

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
