from neural.model.layer_factory import (
    equation_str_to_int,
    env_to_in_out_sizes,
    to_layers,
)
from neural.model.layer import Layer, ActivatedLayer
from gym import make

import pytest

IN_SIZE = 10
OUT_SIZE = 20


class TestEquationStrParsing:
    def call_fn(self, string):
        return equation_str_to_int(IN_SIZE, OUT_SIZE, string)

    def test_replaces_input(self):
        assert IN_SIZE == self.call_fn("input")

    def test_replaces_output(self):
        assert OUT_SIZE == self.call_fn("output")

    def test_addition(self):
        assert 8 == self.call_fn("5+3")

    def test_strip_spaces(self):
        assert 8 == self.call_fn("5 + 3")

    def test_order_of_operations(self):
        assert 60 == self.call_fn("5*10 + 20/2")

    def test_decimal_division_results_round_down(self):
        assert 18 == self.call_fn("20/3*3")

    def test_floats_cause_exception(self):
        with pytest.raises(ValueError):
            self.call_fn("5.0-2.0")

    def test_operator_first_exception(self):
        with pytest.raises(Exception):
            self.call_fn("+20-3")

    def test_operator_last_exception(self):
        with pytest.raises(Exception):
            self.call_fn("20-3+")

    def test_two_operators_exception(self):
        with pytest.raises(Exception):
            self.call_fn("20--3")

    def test_unknown_keyword_exception(self):
        with pytest.raises(Exception):
            self.call_fn("20-3 + input * foo")


class TestEnvToInOutSizes:
    def test_discrete_action_space(self):
        # Discrete action space(3), Box observation space(6,).
        env = make("Acrobot-v1")
        assert 6, 3 == env_to_in_out_sizes(env)

    def test_box_action_space(self):
        # Box action space(1,), Box observation space(2,).
        env = make("MountainCarContinuous-v0")
        assert 2, 1 == env_to_in_out_sizes(env)


class TestToLayers:
    @pytest.mark.parametrize(
        "layer_defs",
        [
            (123, "tanh", "input", "output"),
            ("l1", 22, "input", "output"),
            ("l1", "nonexistant", "input", "output"),
            ("l1", "tanh", "+3*input", "output"),
            ("l1", "tanh", "input", "*2+output"),
            ("l1", "tanh", 2.1, "output"),
            ("l1", "tanh", "input", 3.3),
            ("l1", "tanh", "input", "output", "extra"),
            ("l1", "tanh", "input"),
        ],
    )
    def test_any_invalid_types_are_asserted(self, layer_defs):
        with pytest.raises(Exception):
            _ = to_layers(100, 200, layer_defs)

    def test_make_one_of_each(self):
        in_size = 100
        out_size = 200
        layer_defs = [
            ("l1", "linear", "input+4", 300),
            ("l2", "tanh", 300, 200),
            ("l3", "sigmoid", 200, 100),
            ("l4", "ReLU", 100, "output*2"),
        ]

        layers = to_layers(in_size, out_size, layer_defs)

        for layer, l_type in zip(layers, [Layer] + [ActivatedLayer] * 4):
            assert isinstance(layer, l_type)

        for layer, shape in zip(
            layers, [(104, 300), (300, 200), (200, 100), (100, 400)]
        ):
            assert layer.shape == shape
