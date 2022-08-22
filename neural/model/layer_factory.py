from .layer import Layer, ActivatedLayer

from gym.spaces import Box, Discrete
from re import split
from torch import nn

# The format of layer_defs are a list of layers ordered as they should be
# ordered in the network. Each element will be a tuple of size 4 that has
# elements (name, type, in_size, out_size)
#
# name - Names the layer, can be any arbitrary name but should be unique to
#        all other layers in the same model.
#
# type - Key to type of the layer. Defined in this module, the type key will
#        be used to determine what to construct for that layer.
#
# in_size - Determines the number of nodes input into the layer, must be an int
#           unless valid equation string is used (see below).
#
# out_size - Determines the number of nodes output out of the layer, must be an int
#           unless valid equation string is used (see below).
#
# equation strings -- Strings that can be parsed into a valid equation that supports
#                     addition, subtraction, multiplication, and division. Also the
#                     keywords 'input' and 'output' will be replaced with the models
#                     input size and final output size respectively. Output size is the
#                     number of actions and does not account for continous actions where
#                     one may have two values for a distribution defining one action.
#                     Note: Division will round down to nearest int.
#
#                     ex. "input*2+1", input_size=3, will use size 7.
#
# layer_defs = [(<name>, <type>, <in_size>, <out_size>), ...]
#


def to_op(token: str):
    if token == "+":
        return lambda x, y: x + y
    elif token == "-":
        return lambda x, y: x - y
    elif token == "*":
        return lambda x, y: x * y
    elif token == "/":
        return lambda x, y: x // y
    else:
        raise ValueError(f"Bad operator {token}")


def compute_stack(stack: list) -> int:
    assert stack, f"Bad equation order {stack}"

    lhs = None
    op = None
    for elem in stack:
        if lhs is None:
            lhs = int(elem)
        elif op is None:
            op = to_op(elem)
        else:
            rhs = int(elem)
            lhs = op(lhs, rhs)
            op = None
    stack.clear()
    return lhs


def equation_str_to_int(in_size: int, out_size: int, equation: str) -> int:
    tokens = split(r"(\+|-|\*|/|\.)", equation)

    first_term = None
    op = None

    term_stack = []

    for t in tokens:
        t = t.strip()

        if t == "input":
            t = in_size
        elif t == "output":
            t = out_size
        elif t == ".":
            raise ValueError(f"No floating point numbers allowed {equation}")
        elif not t.isnumeric() and t not in ["+", "-", "*", "/"]:
            raise ValueError(f'Bad keyword "{t}" found in {equation}')

        # When addition or subtraction is found the term preceding it is finished and can be evaluated.
        if t == "+" or t == "-":
            if first_term is None:
                first_term = compute_stack(term_stack)
                op = to_op(t)
            else:
                # In this case the first term and operator are combined with the second and evaluation
                # may proceed treating this result as the new first term.
                first_term = to_op(t)(first_term, compute_stack(term_stack))
                op = None
        else:
            term_stack.append(t)

    if first_term is None:
        # No + or - term separators were encountered so op will be None.
        return compute_stack(term_stack)
    else:
        # op is not None here so the second term in term_stack must be non-empty.
        return op(first_term, compute_stack(term_stack))


def make_layer(name: str, type_key: str, in_size: int, out_size: int):
    """
    type_key options:
        * linear - no activation
        * tanh
        * sigmoid
        * ReLU
    """
    if type_key == "linear":
        return Layer(name, in_size, out_size, bias=True)
    elif type_key == "linear:no_bias":
        return Layer(name, in_size, out_size, bias=False)
    else:
        activation = None
        if type_key == "tanh":
            activation = nn.Tanh()
        elif type_key == "sigmoid":
            activation = nn.Sigmoid()
        elif type_key == "ReLU":
            activation = nn.ReLU()
        else:
            raise ValueError(f"Unexpected layer type {type_key}")
        return ActivatedLayer(name, in_size, out_size, activation, bias=True)


def to_layers(in_size: int, out_size: int, layer_defs: list) -> list:
    """
    Exceptions raised for:
        * Layer_def elements must be tuples with types (str, str, int|str, int|str).
        * If elements 2,3 of the tuple are str they must be valid equations as defined in this module.
        * Names in layer_defs must be unique.
    """
    layers = []

    visited_names = set()

    prev_out_size = in_size
    for name, type_key, layer_in_size, layer_out_size in layer_defs:
        assert isinstance(name, str)
        assert isinstance(type_key, str)

        if isinstance(layer_in_size, str):
            layer_in_size = equation_str_to_int(in_size, out_size, layer_in_size)
        if isinstance(layer_out_size, str):
            layer_out_size = equation_str_to_int(in_size, out_size, layer_out_size)

        assert isinstance(layer_in_size, int)
        assert isinstance(layer_out_size, int)

        assert name not in visited_names
        visited_names.add(name)

        layers.append(make_layer(name, type_key, layer_in_size, layer_out_size))

    return layers


def env_to_in_out_sizes(env) -> (int, int):
    def flatten_dims(shape: tuple) -> int:
        out = 1
        for dim in list(shape):
            out *= dim
        return out

    # Input size will be each dimension multiplied as it would be flattened.
    input_size = flatten_dims(env.observation_space.shape)

    if isinstance(env.action_space, Discrete):
        return input_size, env.action_space.n
    elif isinstance(env.action_space, Box):
        return input_size, flatten_dims(env.action_space.shape)
    else:
        return input_size, flatten_dims(env.action_space.shape)
