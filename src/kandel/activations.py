from jax.nn import celu, elu, leaky_relu, relu, selu, sigmoid, silu, tanh
from jax.typing import ArrayLike


def linear(x: ArrayLike) -> ArrayLike:
    r"""Elementwise linear: :math:`x`."""
    return x


activations = {
    "tanh": tanh,
    "sigmoid": sigmoid,
    "relu": relu,
    "elu": elu,
    "linear": linear,
    "leaky_relu": leaky_relu,
    "celu": celu,
    "silu": silu,
    "selu": selu,
}
