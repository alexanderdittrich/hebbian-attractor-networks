from typing import Tuple

from flax import struct
from jax import numpy
from jax.numpy import clip, expand_dims, hstack, repeat, sqrt, zeros
from jax.random import normal, uniform
from jax.typing import ArrayLike

from kandel.strategy.strategy import Strategy


@struct.dataclass
class EvoState:
    mean: ArrayLike
    sigma: ArrayLike
    weights: ArrayLike
    gen_counter: int


class AdaptiveES(Strategy):
    """Evolution Strategy with adaptive mutation rate.
    Defaults to maximizing fitness score.
    """

    name = "adaptive-es"

    def __init__(
        self,
        population_size: int,
        num_dims: int,
        elite_ratio: float = 0.2,
        sigma_init: float = 0.05,
        c_m: float = 1.0,
        c_sigma: float = 0.1,
        clip_min: float = -1e8,
        clip_max: float = 1e8,
    ):
        self.population_size = population_size
        self.num_dims = num_dims
        self.elite_ratio = elite_ratio
        self.sigma_init = sigma_init
        self.c_m = c_m
        self.c_sigma = c_sigma
        self.clip_min = clip_min
        self.clip_max = clip_max

    def initialize(self, rng: ArrayLike, minval=-1.0, maxval=1.0) -> EvoState:
        elite_popsize = max(1, int(self.population_size * self.elite_ratio))
        weights = zeros(self.population_size)
        weights = weights.at[:elite_popsize].set(1 / elite_popsize)

        state = EvoState(
            mean=uniform(rng, (self.num_dims,), minval=minval, maxval=maxval),
            sigma=repeat(self.sigma_init, self.num_dims),
            weights=weights,
            gen_counter=0,
        )

        return state

    def set_mean(self, state: EvoState, mean: ArrayLike) -> EvoState:
        assert state.mean.shape == mean.shape, (
            f"Error: Mean has different shape! "
            f"Given shape: {mean.shape}, expected shape: {state.mean.shape}"
        )
        return state.replace(mean=mean)

    def ask(self, rng: ArrayLike, state: EvoState) -> ArrayLike:
        noise = normal(rng, (self.population_size, self.num_dims))
        population = state.mean + state.sigma * noise  # ~ N(m, σ^2 I)
        population_clipped = clip(population, self.clip_min, self.clip_max)
        return population_clipped, state

    def tell(
        self, population: ArrayLike, fitness: ArrayLike, state: EvoState
    ) -> EvoState:
        # minimizing instead of maximizing
        fitness = -1 * fitness

        # Sort new results, extract elite, store best performer
        concat_p_f = hstack([expand_dims(fitness, 1), population])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]

        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        mean, y_k = update_mean(
            sorted_solutions, state.mean, self.c_m, state.weights
        )
        sigma = update_sigma(y_k, state.sigma, self.c_sigma, state.weights)
        return state.replace(
            mean=mean, sigma=sigma, gen_counter=state.gen_counter + 1
        )


def update_mean(
    sorted_solutions: ArrayLike,
    mean: ArrayLike,
    c_m: float,
    weights: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike]:
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = x_k - mean
    y_w = numpy.sum(y_k.T * weights, axis=1)
    mean_new = mean + c_m * y_w
    return mean_new, y_k


def update_sigma(
    y_k: ArrayLike, sigma: ArrayLike, c_sigma: float, weights: ArrayLike
) -> ArrayLike:
    """Update stepsize sigma."""
    sigma_est = sqrt(numpy.sum((y_k.T**2 * weights), axis=1))
    sigma_new = (1 - c_sigma) * sigma + c_sigma * sigma_est
    return sigma_new
