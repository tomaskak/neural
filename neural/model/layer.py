from torch import nn, Tensor, zeros
from numpy import sqrt


def xavier_init(in_size: int, out_size: int) -> Tensor:
    out = zeros((out_size, in_size))
    nn.init.xavier_uniform_(out)
    return out


class Layer(nn.Module):
    """
    Defines a basic layer of weights and optional bias that implements the
    forward functionality of 'Ax + b' .

    Initialization of weights uses 'Xavier initialization' which samples weights
    from a uniform distribution according to:
        U[-sqrt(6)/sqrt(in_size + out_size), sqrt(6)/sqrt(in_size + out_size)]
    """

    def __init__(
        self, name: str, in_size: int, out_size: int, bias: bool, norm: bool = False
    ):
        super().__init__()
        self._name = name
        self.weights = nn.Parameter(data=xavier_init(in_size, out_size))
        self.bias = nn.Parameter(zeros((out_size,))) if bias else None
        self._shape = (in_size, out_size)
        self.norm = norm

    def forward(self, X: Tensor) -> Tensor:
        """
        Assumes each input in this case is a row, and produces output tensor
        where each row is the output for the corresponding input.

        In the traditional form each input in X is a column, and each output
        in y is a column, also b is usually a column.
            A @ x.T + b = y

        For simplicity each row will be an output.
            (A @ x.T + b).T = y.T
            x @ A.T + b.T = y.T
        """
        if self.norm:
            if self.bias is not None:
                return (
                    nn.functional.layer_norm(X.T, (X.shape[0],)).T @ self.weights.T
                    + self.bias
                )
            else:
                return nn.functional.layer_norm(X.T, (X.shape[0],)).T @ self.weights.T
        else:
            if self.bias is not None:
                return X @ self.weights.T + self.bias
            else:
                return X @ self.weights.T

    @property
    def shape(self):
        return self._shape

    @property
    def name(self):
        return self._name


class ActivatedLayer(Layer):
    """
    Defines a layer on top of a linear weighting layer that applies a given
    activation method on the output of the linear weighting layer.
    """

    def __init__(
        self, name: str, in_size: int, out_size: int, activation_fn, bias: bool
    ):
        super().__init__(name, in_size, out_size, bias)
        self.activation = activation_fn

    def forward(self, X: Tensor) -> Tensor:
        out = super().forward(X)
        return self.activation(out)
