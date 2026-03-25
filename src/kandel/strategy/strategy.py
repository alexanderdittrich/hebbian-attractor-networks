from abc import ABC, abstractmethod
from typing import Tuple

from flax import struct
from jax.typing import ArrayLike


@struct.dataclass
class EvoState:
    mean: ArrayLike
    sigma: float


class Strategy(ABC):
    @abstractmethod
    def initialize(self):
        """Generates initial state."""
        pass

    @abstractmethod
    def ask(
        self, rng: ArrayLike, state: EvoState, *args, **kwargs
    ) -> Tuple[ArrayLike, ArrayLike]:
        """Generate candidate solutions."""
        pass

    @abstractmethod
    def tell(self, fitness_scores: ArrayLike, *args, **kwargs) -> EvoState:
        """Update the strategy based on fitness scores."""
        pass
