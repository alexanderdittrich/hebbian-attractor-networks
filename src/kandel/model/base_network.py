from abc import ABC, abstractmethod
from typing import Any

from jax.typing import ArrayLike

ParameterPytree = Any
StatePytree = Any


class BaseNeuralNetwork(ABC):
    """Abstract base class for all neural networks."""

    @abstractmethod
    def initialize(self, random_key) -> ParameterPytree:
        pass

    @abstractmethod
    def forward(self, params, input: ArrayLike, *args, **kwargs) -> ArrayLike:
        pass

    def update(self, params, *args, **kwargs) -> ParameterPytree:
        return params

    def reset(self, params, random_key, *args, **kwargs) -> ParameterPytree:
        # Reset can be utilized for Hebbian networks to reset kernel params.
        # For reseting fixed weight architectures, instantiate a new network.
        return params


class FixedWeightNeuralNetwork(BaseNeuralNetwork):
    def opt_params(self, params):
        return params["kernel_params"]

    def update_opt_params(self, params, updated_params):
        params["kernel_params"] = updated_params
        return params


class HebbianNeuralNetwork(BaseNeuralNetwork):
    def opt_params(self, params):
        return params["hebbian_params"]

    def update_opt_params(self, params, updated_params):
        params["hebbian_params"] = updated_params
        return params
