from jax.nn import tanh
from jax.numpy import abs, clip
from jax.typing import ArrayLike


def identity(x: ArrayLike) -> ArrayLike:
    return x


def abs_scale(x: ArrayLike) -> ArrayLike:
    return x / abs(x).max()


def clipper(epsilon: float = 1.0):
    def _clipper(x: ArrayLike) -> ArrayLike:
        return clip(x, min=-epsilon, max=epsilon)

    return _clipper


regularizers = {
    "identity": identity,
    "abs-scale": abs_scale,
    "tanh": tanh,
    "clip_1": clipper(1.0),
    "clip_2": clipper(2.0),
    "clip_5": clipper(5.0),
    "clip_10": clipper(10.0),
    "clip_100": clipper(100.0),
    "clip_1000": clipper(1000.0),
}
