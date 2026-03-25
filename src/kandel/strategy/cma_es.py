from typing import Optional, Tuple

from flax import struct
from jax.numpy import (
    abs,
    array,
    clip,
    diag,
    exp,
    expand_dims,
    eye,
    hstack,
    log,
    maximum,
    minimum,
    outer,
    sqrt,
    sum,
    swapaxes,
    where,
    zeros,
)
from jax.numpy.linalg import norm
from jax.random import normal, uniform
from jax.typing import ArrayLike

from kandel.strategy.strategy import Strategy


@struct.dataclass
class EvoState:
    p_sigma: ArrayLike
    p_c: ArrayLike
    C: ArrayLike
    D: Optional[ArrayLike]
    B: Optional[ArrayLike]
    mean: ArrayLike
    sigma: float
    weights: ArrayLike
    weights_truncated: ArrayLike
    mu_eff: float
    c_1: float
    c_mu: float
    c_sigma: float
    d_sigma: float
    c_c: float
    chi_n: float
    c_m: float
    gen_counter: int


@struct.dataclass
class EvoConfig:
    population_size: int
    num_dims: int
    elite_ratio: float
    sigma_init: float
    c_m: float
    clip_min: float
    clip_max: float


def get_cma_elite_weights(
    popsize: int,
    elite_popsize: int,
    num_dims: int,
    max_dims_sq: int,
) -> Tuple[ArrayLike, ArrayLike, float, float, float]:
    """Utility helper to create truncated elite weights for mean
    update and full weights for covariance update."""
    weights_prime = array(
        [log((popsize + 1) / 2) - log(i + 1) for i in range(popsize)]
    )
    mu_eff = (sum(weights_prime[:elite_popsize]) ** 2) / sum(
        weights_prime[:elite_popsize] ** 2
    )
    mu_eff_minus = (sum(weights_prime[elite_popsize:]) ** 2) / sum(
        weights_prime[elite_popsize:] ** 2
    )

    # lrates for rank-one and rank-μ C updates
    alpha_cov = 2
    c_1 = alpha_cov / ((max_dims_sq + 1.3) ** 2 + mu_eff)
    c_mu = minimum(
        1 - c_1 - 1e-8,
        alpha_cov
        * (mu_eff - 2 + 1 / mu_eff)
        / ((max_dims_sq + 2) ** 2 + alpha_cov * mu_eff / 2),
    )
    min_alpha = minimum(1 + c_1 / c_mu, 1 + (2 * mu_eff_minus) / (mu_eff + 2))
    min_alpha = minimum(min_alpha, (1 - c_1 - c_mu) / (num_dims * c_mu))
    positive_sum = sum(weights_prime * (weights_prime > 0))
    negative_sum = sum(abs(weights_prime * (weights_prime < 0)))
    weights = where(
        weights_prime >= 0,
        1 / positive_sum * weights_prime,
        min_alpha / negative_sum * weights_prime,
    )
    weights_truncated = weights.at[elite_popsize:].set(0)
    return weights, weights_truncated, mu_eff, c_1, c_mu


def full_eigen_decomp(
    C: ArrayLike, B: ArrayLike, D: ArrayLike, gen_counter: int
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Perform eigendecomposition of covariance matrix."""
    if B is not None and D is not None:
        return C, B, D
    C = C + 1e-10 * (gen_counter == 0)
    C = (C + C.T) / 2  # Make sure matrix is symmetric
    D2, B = jnp.linalg.eigh(C)
    D = jnp.sqrt(jnp.where(D2 < 0, 1e-20, D2))
    C = jnp.dot(jnp.dot(B, jnp.diag(D**2)), B.T)
    return C, B, D


class CMA_ES(Strategy):
    def __init__(
        self,
        population_size: int,
        num_dims: int,
        elite_ratio: float = 0.5,
        sigma_init: float = 1.0,
        c_m: float = 1.0,
        clip_min: float = -1e8,
        clip_max: float = 1e8,
    ):
        """Single file implementation of Covariance Matrix Adaptation -
        Evolution Strategy (CMA-ES) [Hansen, 2016]
        Reference: https://arxiv.org/abs/1604.00772
        Inspired by: https://github.com/CyberAgentAILab/cmaes

        The implementation is based on the great work of Robert T. Lange.
        Reference: https://github.com/RobertTLange/evosax
        """
        self.population_size = population_size
        self.num_dims = num_dims
        assert 0 <= elite_ratio <= 1
        self.elite_ratio = elite_ratio
        self.elite_popsize = max(1, int(self.popsize * self.elite_ratio))

        # Set core kwargs es_params
        self.sigma_init = sigma_init
        self.c_m = c_m
        self.clip_min = clip_min
        self.clip_max = clip_max

        # Robustness for int32 - squaring in hyperparameter calculations
        self.max_dims_sq = minimum(self.num_dims, 40000)

    def initialize(
        self, rng: ArrayLike, minval: float = -1.0, maxval: float = 1.0
    ) -> Tuple[EvoState, EvoConfig]:
        """`initialize` the evolution strategy."""
        weights, weights_truncated, mu_eff, c_1, c_mu = get_cma_elite_weights(
            self.popsize, self.elite_popsize, self.num_dims, self.max_dims_sq
        )

        # lrate for cumulation of step-size control and rank-one update
        c_sigma = (mu_eff + 2) / (self.num_dims + mu_eff + 5)
        d_sigma = (
            1
            + 2 * maximum(0, sqrt((mu_eff - 1) / (self.num_dims + 1)) - 1)
            + c_sigma
        )
        c_c = (4 + mu_eff / self.num_dims) / (
            self.num_dims + 4 + 2 * mu_eff / self.num_dims
        )
        chi_n = sqrt(self.num_dims) * (
            1.0
            - (1.0 / (4.0 * self.num_dims))
            + 1.0 / (21.0 * (self.max_dims_sq**2))
        )

        state = EvoState(
            p_sigma=zeros(self.num_dims),
            p_c=zeros(self.num_dims),
            sigma=self.sigma_init,
            mean=uniform(rng, (self.num_dims,), minval=minval, maxval=maxval),
            C=eye(self.num_dims),
            D=None,
            B=None,
            weights=weights,
            weights_truncated=weights_truncated,
            mu_eff=mu_eff,
            c_1=c_1,
            c_mu=c_mu,
            c_sigma=c_sigma,
            d_sigma=d_sigma,
            c_c=c_c,
            chi_n=chi_n,
            gen_counter=0,
        )

        config = EvoConfig(
            sigma_init=self.sigma_init,
            c_m=self.c_m,
            clip_min=self.clip_min,
            clip_max=self.clip_max,
        )
        return state, config

    def set_mean(self, state: EvoState, mean: ArrayLike) -> EvoState:
        assert state.mean.shape == mean.shape, (
            f"Error: Mean has different shape! "
            f"Given shape: {mean.shape}, expected shape: {state.mean.shape}"
        )
        return state.replace(mean=mean)

    def ask(
        self, rng: ArrayLike, state: EvoState, config: EvoConfig
    ) -> Tuple[ArrayLike, EvoState]:
        """`ask` for new parameter candidates to evaluate next."""
        C, B, D = full_eigen_decomp(
            state.C, state.B, state.D, state.gen_counter
        )
        x = sample(
            rng,
            state.mean,
            state.sigma,
            B,
            D,
            config.num_dims,
            config.popsize,
        )
        x_clipped = clip(x, config.clip_min, config.clip_max)
        return x_clipped, state.replace(C=C, B=B, D=D)

    def tell_strategy(
        self,
        x: ArrayLike,
        fitness: ArrayLike,
        state: EvoState,
        config: EvoConfig,
    ) -> EvoState:
        """`tell` performance data for strategy state update."""
        # Sort new results, extract elite, store best performer
        concat_p_f = hstack([expand_dims(fitness, 1), x])
        sorted_solutions = concat_p_f[concat_p_f[:, 0].argsort()]
        # Update mean, isotropic/anisotropic paths, covariance, stepsize
        y_k, y_w, mean = update_mean(
            state.mean,
            state.sigma,
            sorted_solutions,
            config.c_m,
            state.weights_truncated,
        )

        p_sigma, C_2, C, B, D = update_p_sigma(
            state.C,
            state.B,
            state.D,
            state.p_sigma,
            y_w,
            state.c_sigma,
            state.mu_eff,
            state.gen_counter,
        )

        p_c, norm_p_sigma, h_sigma = update_p_c(
            mean,
            p_sigma,
            state.p_c,
            state.gen_counter + 1,
            y_w,
            state.c_sigma,
            state.chi_n,
            state.c_c,
            state.mu_eff,
        )

        C = update_covariance(
            mean,
            p_c,
            C,
            y_k,
            h_sigma,
            C_2,
            state.weights,
            state.c_c,
            state.c_1,
            state.c_mu,
        )
        sigma = update_sigma(
            state.sigma,
            norm_p_sigma,
            state.c_sigma,
            state.d_sigma,
            state.chi_n,
        )

        return state.replace(
            mean=mean, p_sigma=p_sigma, C=C, B=B, D=D, p_c=p_c, sigma=sigma
        )


def update_mean(
    mean: ArrayLike,
    sigma: float,
    sorted_solutions: ArrayLike,
    c_m: float,
    weights_truncated: ArrayLike,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike]:
    """Update mean of strategy."""
    x_k = sorted_solutions[:, 1:]  # ~ N(m, σ^2 C)
    y_k = (x_k - mean) / sigma  # ~ N(0, C)
    y_w = sum(y_k.T * weights_truncated, axis=1)
    mean += c_m * sigma * y_w
    return y_k, y_w, mean


def update_p_sigma(
    C: ArrayLike,
    B: ArrayLike,
    D: ArrayLike,
    p_sigma: ArrayLike,
    y_w: ArrayLike,
    c_sigma: float,
    mu_eff: float,
    gen_counter: int,
) -> Tuple[ArrayLike, ArrayLike, ArrayLike, None, None]:
    """Update evolution path for covariance matrix."""
    C, B, D = full_eigen_decomp(C, B, D, gen_counter)
    C_2 = B.dot(diag(1 / D)).dot(B.T)  # C^(-1/2) = B D^(-1) B^T
    p_sigma_new = (1 - c_sigma) * p_sigma + sqrt(
        c_sigma * (2 - c_sigma) * mu_eff
    ) * C_2.dot(y_w)
    _B, _D = None, None
    return p_sigma_new, C_2, C, _B, _D


def update_p_c(
    mean: ArrayLike,
    p_sigma: ArrayLike,
    p_c: ArrayLike,
    gen_counter: int,
    y_w: ArrayLike,
    c_sigma: float,
    chi_n: float,
    c_c: float,
    mu_eff: float,
) -> Tuple[ArrayLike, float, float]:
    """Update evolution path for sigma/stepsize."""
    norm_p_sigma = norm(p_sigma)
    h_sigma_cond_left = norm_p_sigma / sqrt(
        1 - (1 - c_sigma) ** (2 * gen_counter)
    )
    h_sigma_cond_right = (1.4 + 2 / (mean.shape[0] + 1)) * chi_n
    h_sigma = 1.0 * (h_sigma_cond_left < h_sigma_cond_right)
    p_c_new = (1 - c_c) * p_c + h_sigma * sqrt(c_c * (2 - c_c) * mu_eff) * y_w
    return p_c_new, norm_p_sigma, h_sigma


def update_covariance(
    mean: ArrayLike,
    p_c: ArrayLike,
    C: ArrayLike,
    y_k: ArrayLike,
    h_sigma: float,
    C_2: ArrayLike,
    weights: ArrayLike,
    c_c: float,
    c_1: float,
    c_mu: float,
) -> ArrayLike:
    """Update cov. matrix estimator using rank 1 + μ updates."""
    w_io = weights * where(
        weights >= 0,
        1,
        mean.shape[0] / (norm(C_2.dot(y_k.T), axis=0) ** 2 + 1e-20),
    )
    delta_h_sigma = (1 - h_sigma) * c_c * (2 - c_c)
    rank_one = outer(p_c, p_c)
    rank_mu = sum(array([w * outer(y, y) for w, y in zip(w_io, y_k)]), axis=0)
    C = (
        (1 + c_1 * delta_h_sigma - c_1 - c_mu * sum(weights)) * C
        + c_1 * rank_one
        + c_mu * rank_mu
    )
    return C


def update_sigma(
    sigma: float,
    norm_p_sigma: float,
    c_sigma: float,
    d_sigma: float,
    chi_n: float,
) -> float:
    """Update stepsize sigma."""
    sigma_new = sigma * exp((c_sigma / d_sigma) * (norm_p_sigma / chi_n - 1))
    return sigma_new


def sample(
    rng: ArrayLike,
    mean: ArrayLike,
    sigma: float,
    B: ArrayLike,
    D: ArrayLike,
    n_dim: int,
    population_size: int,
) -> ArrayLike:
    """Jittable Gaussian Sample Helper."""
    z = normal(rng, (n_dim, population_size))  # ~ N(0, I)
    y = B.dot(diag(D)).dot(z)  # ~ N(0, C)
    y = swapaxes(y, 1, 0)
    x = mean + sigma * y  # ~ N(m, σ^2 C)
    return x
