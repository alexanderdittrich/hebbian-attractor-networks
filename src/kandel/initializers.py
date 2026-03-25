# Copyright 2019 The JAX Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Common neural network layer initializers, consistent with definitions
used in Keras and Sonnet.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
from flax.linen import ones_init, zeros_init
from jax import random
from jax._src import core, dtypes
from jax._src.util import set_module
from jax.nn.initializers import (
    Initializer,
    glorot_normal,
    glorot_uniform,
    he_normal,
    he_uniform,
    kaiming_normal,
    kaiming_uniform,
    lecun_normal,
    lecun_uniform,
    normal,
    uniform,
    xavier_normal,
    xavier_uniform,
)
from jax.typing import ArrayLike

export = set_module("jax.nn.initializers")

KeyArray = ArrayLike
DTypeLikeInexact = Any
RealNumeric = Any  # Scalar jnp array or float


@export
def range_uniform(
    minval: RealNumeric = -1.0,
    maxval: RealNumeric = 1.0,
    dtype: DTypeLikeInexact = jnp.float_,
) -> Initializer:
    """Builds an initializer that returns real uniformly-distributed random arrays.

    Args:
      scale: optional; the upper bound of the random distribution.
      dtype: optional; the initializer's default dtype.

    Returns:
      An initializer that returns arrays whose values are uniformly distributed in
      the range ``[0, scale)``.

    >>> import jax, jax.numpy as jnp
    >>> initializer = range_uniform(-10.0, 10.0)
    >>> initializer(jax.random.key(42), (2, 3), jnp.float32)  # doctest: +SKIP
    Array([[ 4.5963764  7.383876   7.446003 ]
           [-5.8362865 -6.2675166  1.0045123]], dtype=float32)
    """

    def init(
        key: KeyArray, shape: core.Shape, dtype: DTypeLikeInexact = dtype
    ) -> ArrayLike:
        dtype = dtypes.canonicalize_dtype(dtype)
        return random.uniform(key, shape, dtype, minval, maxval)

    return init


initializers = {
    "uniform": uniform(),
    "uniform_0.0_1.0": range_uniform(0.0, 1.0),
    "uniform_-0.001_0.001": range_uniform(-0.001, 0.001),
    "uniform_-0.01_0.01": range_uniform(-0.01, 0.01),
    "uniform_-0.1_0.1": range_uniform(-0.1, 0.1),
    "uniform_-0.2_0.2": range_uniform(-0.2, 0.2),
    "uniform_-0.5_0.5": range_uniform(-0.5, 0.5),
    "uniform_-1.0_1.0": range_uniform(-1.0, 1.0),
    "uniform_-2.0_2.0": range_uniform(-2.0, 2.0),
    "uniform_-5.0_5.0": range_uniform(-5.0, 5.0),
    "normal_0.1": normal(stddev=0.1),
    "normal_0.2": normal(stddev=0.2),
    "normal_0.5": normal(stddev=0.5),
    "normal_1.0": normal(stddev=1.0),
    "normal_2.0": normal(stddev=2.0),
    "lecun_normal": lecun_normal(),
    "lecun_uniform": lecun_uniform(),
    "he_normal": he_normal(),
    "he_uniform": he_uniform(),
    "glorot_normal": glorot_normal(),
    "glorot_uniform": glorot_uniform(),
    "zeros": zeros_init(),
    "ones": ones_init(),
    "kaiming_normal": kaiming_normal(),
    "kaiming_uniform": kaiming_uniform(),
    "xavier_normal": xavier_normal(),
    "xavier_uniform": xavier_uniform(),
}
