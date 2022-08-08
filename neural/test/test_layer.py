from neural.model.layer import Layer, ActivatedLayer

import pytest

import torch
from numpy import sqrt


def xavier_limit(in_size: int, out_size: int):
    return sqrt(6) / sqrt(in_size + out_size)


def make_one(cls, shape, *args, **kwargs):
    l_in, l_out = shape
    layer = cls("l1", l_in, l_out, *args, **kwargs)
    return layer, l_in, l_out, dict(layer.named_parameters())


class TestLayer:
    def test_init_with_bias(self):
        layer, l_in, l_out, params = make_one(Layer, (3, 5), bias=True)

        assert params["weights"].shape == (l_out, l_in)
        assert torch.all(
            torch.le(torch.abs(params["weights"]), xavier_limit(l_in, l_out))
        )

        assert params["bias"].shape == (l_out,)
        assert torch.all(torch.eq(params["bias"], torch.zeros((l_in, l_out))))

    def test_init_no_bias(self):
        layer, l_in, l_out, params = make_one(Layer, (3, 5), bias=False)

        assert "bias" not in params

    def test_forward(self):
        layer, l_in, l_out, params = make_one(Layer, (2, 2), bias=True)

        x_0, x_1 = (1.0, 2.0)
        out = layer.forward(torch.tensor([[x_0, x_1]]))

        weights = params["weights"]
        bias = params["bias"]  # bias in this case is zero so has no effect
        assert (
            out[0][0].item()
            == (weights[0][0] * x_0 + weights[0][1] * x_1 + bias[0]).item()
        )
        assert (
            out[0][1].item()
            == (weights[1][0] * x_0 + weights[1][1] * x_1 + bias[1]).item()
        )

    def test_gradient(self):
        """
        The gradient of a layer should result in convergence when trained on the same input
        and target repeatedly.
        """
        layer, l_in, l_out, params = make_one(Layer, (2, 1), bias=True)

        lr = 0.01
        target = torch.tensor([[0.555]])
        obs = torch.tensor([[2.0, 4.0]], requires_grad=False)

        for _ in range(50):
            layer.zero_grad()
            out = layer.forward(obs)
            loss = (out - target.detach()) ** 2
            loss.backward()

            with torch.no_grad():
                for param in layer.parameters():
                    param -= lr * param.grad

        assert torch.allclose(target, layer.forward(obs))


class TestActivatedLayer:
    def test_activation_fn(self):
        layer, l_in, l_out, params = make_one(
            ActivatedLayer, (2, 2), activation_fn=lambda X: X * 2, bias=False
        )

        x_0, x_1 = (1.0, 2.0)
        out = layer.forward(torch.tensor([[x_0, x_1]]))

        weights = params["weights"]
        assert (
            out[0][0].item() == (weights[0][0] * x_0 + weights[0][1] * x_1).item() * 2
        )
        assert (
            out[0][1].item() == (weights[1][0] * x_0 + weights[1][1] * x_1).item() * 2
        )

    def test_multiple_inputs(self):
        layer, l_in, l_out, params = make_one(
            ActivatedLayer, (2, 2), activation_fn=lambda X: X * 2, bias=False
        )

        x_0, x_1 = (1.0, 2.0)
        x_2, x_3 = (3.0, 4.0)
        out = layer.forward(torch.tensor([[x_0, x_1], [x_2, x_3]]))

        weights = params["weights"]
        assert (
            out[0][0].item() == (weights[0][0] * x_0 + weights[0][1] * x_1).item() * 2
        )
        assert (
            out[0][1].item() == (weights[1][0] * x_0 + weights[1][1] * x_1).item() * 2
        )
        assert (
            out[1][0].item() == (weights[0][0] * x_2 + weights[0][1] * x_3).item() * 2
        )
        assert (
            out[1][1].item() == (weights[1][0] * x_2 + weights[1][1] * x_3).item() * 2
        )
