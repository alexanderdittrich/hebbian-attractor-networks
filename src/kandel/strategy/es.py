from typing import Tuple, Union

import optax
from flax import struct
from jax.numpy import argsort, average, clip, maximum, take
from jax.random import normal, uniform
from jax.typing import ArrayLike

from kandel.strategy.strategy import Strategy


@struct.dataclass
class EvoState:
    mean: ArrayLike
    sigma: float
    gen_counter: int


schedules = {
    "constant": optax.constant_schedule,
    "exponential_decay": optax.exponential_decay,
}


class ESComma(Strategy):
    """Evolution Strategy with `comma` variant.
    Defaults to maximizing fitness score.
    """

    name = "es"

    def __init__(
        self,
        population_size: int,
        num_dims: int,
        elite_ratio: float = 0.5,
        sigma_schedule: str = "exponential_decay",
        sigma_cfg: dict = {
            "initial_value": 0.05,
            "transition_steps": 1000,
            "decay_rate": 0.2,
        },
        clip_min: float = -1e8,
        clip_max: float = 1e8,
    ):
        self.population_size = population_size
        self.num_dims = num_dims
        self.elite_ratio = elite_ratio
        self.sigma_schedule = schedules[sigma_schedule](**sigma_cfg)
        self.clip_min = clip_min
        self.clip_max = clip_max

        self.elite_popsize = max(
            1, int(self.population_size * self.elite_ratio)
        )

    def initialize(self, rng: ArrayLike, minval=-1.0, maxval=1.0) -> EvoState:
        state = EvoState(
            mean=uniform(rng, (self.num_dims,), minval=minval, maxval=maxval),
            sigma=self.sigma_schedule(0),
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
        population = state.mean + state.sigma * noise
        population_clipped = clip(population, self.clip_min, self.clip_max)
        return population_clipped, state

    def tell(
        self, population: ArrayLike, fitness: ArrayLike, state: EvoState
    ) -> EvoState:
        # since last elite popsize is having a zero weight, we increment by 1
        sorted_indices = argsort(fitness)[::-1]
        sorted_indices = sorted_indices[: self.elite_popsize]

        elite_x = take(population, indices=sorted_indices, axis=0)
        elite_scores = take(fitness, indices=sorted_indices, axis=0)

        # minmax-normalize scores to get non-negative scores
        min_elite_score = elite_scores.min()
        normed_scores = (elite_scores - min_elite_score) / (
            elite_scores.max() - min_elite_score + 1e-10
        )

        # renormalize to get a summed weight of 1
        normed_scores = normed_scores / normed_scores.sum()

        # sum weights of weighted elite solutions
        mean = average(elite_x, axis=0, weights=normed_scores)

        # update sigma
        sigma = self.sigma_schedule(state.gen_counter)
        return state.replace(
            mean=mean, sigma=sigma, gen_counter=state.gen_counter + 1
        )
