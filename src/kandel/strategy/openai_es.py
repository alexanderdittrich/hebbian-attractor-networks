from typing import Callable, Tuple

import optax
from flax import struct
from jax.nn import standardize
from jax.numpy import (
    allclose,
    clip,
    concatenate,
    dot,
    isnan,
    nanmax,
    nanmin,
    ones_like,
    where,
    zeros,
)
from jax.random import normal, uniform
from jax.scipy.stats import rankdata
from jax.typing import ArrayLike

from kandel.strategy.strategy import Strategy


@struct.dataclass
class EvoState:
    mean: ArrayLike
    sigma: float
    gen_counter: int
    opt_state: optax.OptState


optimizers = {
    "sgd": optax.sgd,
    "adam": optax.adam,
    "adamw": optax.adamw,
}

schedules = {
    "constant": optax.constant_schedule,
    "exponential_decay": optax.exponential_decay,
}


def identity(fitness: ArrayLike) -> ArrayLike:
    return fitness


def center_ranks(fitness: ArrayLike) -> ArrayLike:
    """Center the fitness values around 0 between -0.5 to 0.5."""
    ranks = rankdata(fitness, axis=-1) - 1.0
    return ranks / (fitness.shape[-1] - 1) - 0.5


def standard_normalize(fitness: ArrayLike) -> ArrayLike:
    """Return standardized fitness."""
    return standardize(fitness, axis=-1, epsilon=1e-8, where=~isnan(fitness))


def normalize(
    fitness: ArrayLike,
    axis: int = -1,
    minval: float = -1.0,
    maxval: float = 1.0,
) -> ArrayLike:
    """Normalize fitness."""
    a_min = nanmin(fitness, axis=axis, keepdims=True)
    a_max = nanmax(fitness, axis=axis, keepdims=True)

    return where(
        allclose(a_max, a_min),
        ones_like(fitness) * (minval + maxval) / 2,
        minval + (maxval - minval) * (fitness - a_min) / (a_max - a_min),
    )


fitness_shaper = {
    "identity": identity,
    "center": center_ranks,
    "standard": standard_normalize,
    "normalize": normalize,
}


class OpenAI_ES(Strategy):
    """Single-file implementation of OpenAI-ES.
    Reference: https://arxiv.org/abs/1703.03864
    The implementation is based on the great work of Robert T. Lange.
    Reference: https://github.com/RobertTLange/evosax
    """

    def __init__(
        self,
        population_size: int,
        num_dims: int,
        optimizer: str = "adam",
        optimizer_cfg: dict = {
            "b1": 0.9,
            "b2": 0.999,
            "eps": 1e-8,
        },
        learning_rate_schedule: str = "exponential_decay",
        learning_rate_cfg: dict = {
            "initial_value": 0.01,
            "transition_steps": 1000,
            "decay_rate": 0.1,
        },
        sigma_schedule: str = "exponential_decay",
        sigma_cfg: dict = {
            "initial_value": 0.05,
            "transition_steps": 1000,
            "decay_rate": 0.2,
        },
        fitness_shaping: str = "center",
        clip_min: float = -1e8,
        clip_max: float = 1e8,
        antithetic: bool = True,
    ):
        if antithetic and population_size % 2 != 0:
            raise ValueError(
                "population_size must be even for antithetic sampling"
            )
        self.population_size = population_size
        self.antithetic = antithetic
        self.num_dims = num_dims

        self.sigma_schedule = schedules[sigma_schedule](**sigma_cfg)
        lrate_schedule = schedules[learning_rate_schedule](**learning_rate_cfg)
        self.optimizer = optimizers[optimizer](
            learning_rate=lrate_schedule, **optimizer_cfg
        )

        self.fitness_shaper = fitness_shaper[fitness_shaping]
        self.clip_min = clip_min
        self.clip_max = clip_max

    def initialize(self, rng: ArrayLike, minval=-1.0, maxval=1.0) -> EvoState:
        state = EvoState(
            mean=uniform(rng, (self.num_dims,), minval=minval, maxval=maxval),
            sigma=self.sigma_schedule(0),
            opt_state=self.optimizer.init(zeros(self.num_dims)),
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
        if self.antithetic:
            noise = normal(rng, (self.population_size // 2, self.num_dims))
            noise = concatenate([noise, -noise], axis=0)
        else:
            noise = normal(rng, (self.population_size, self.num_dims))
        population = state.mean + state.sigma * noise

        population_clipped = clip(population, self.clip_min, self.clip_max)
        return population_clipped, state

    def tell(
        self,
        population: ArrayLike,
        fitness: ArrayLike,
        state: EvoState,
    ) -> EvoState:
        fitness = -fitness
        fitness = self.fitness_shaper(fitness)

        noise = (population - state.mean) / state.sigma
        theta_grad = dot(fitness, noise) / (self.population_size * state.sigma)

        # Update mean
        updates, opt_state = self.optimizer.update(theta_grad, state.opt_state)
        mean = optax.apply_updates(state.mean, updates)

        sigma = self.sigma_schedule(state.gen_counter + 1)

        return state.replace(
            mean=mean,
            sigma=sigma,
            opt_state=opt_state,
            gen_counter=state.gen_counter + 1,
        )
