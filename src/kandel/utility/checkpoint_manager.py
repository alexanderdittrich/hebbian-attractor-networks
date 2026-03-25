import json
from pathlib import Path
from typing import Any, Optional, Sequence

import jax
import numpy as np
from orbax.checkpoint import StandardCheckpointer

EPISODE_NAME = "generation"
ENSEMBLE_NAME = "ensemble"
INDIVIDUAL_NAME = "policy"


def select_population_member(policy_params: Any, population_member: int):
    return jax.tree.map(lambda x: x[population_member], policy_params)


class CheckpointManager:
    def __init__(
        self,
        save_interval_steps: int,
        save_directory: str,
        number_elites: Optional[int] = None,
        save_last_at_close: bool = True,
    ):
        self.save_interval_steps = save_interval_steps
        self.save_directory = Path(save_directory)
        self.number_elites = number_elites
        self.save_last_at_close = save_last_at_close

        self.checkpointer = StandardCheckpointer()
        self.episode_index = 0
        self._last_policy_params = None
        self._last_rewards = None

    def save_policy_ensemble(self):
        save_path = (
            self.save_directory
            / f"{EPISODE_NAME}_{self.episode_index}"
            / ENSEMBLE_NAME
        )
        self.checkpointer.save(save_path, self._last_policy_params)

    def save_top_policies(self):
        elite_indices = np.argsort(self._last_rewards)[-self.number_elites :][
            ::-1
        ]
        for i, elite_idx in enumerate(elite_indices):
            single_policy_param = select_population_member(
                self._last_policy_params, elite_idx
            )
            save_path = (
                self.save_directory
                / f"{EPISODE_NAME}_{self.episode_index}"
                / f"{INDIVIDUAL_NAME}_{i}"
            )
            self.checkpointer.save(save_path, single_policy_param)

    def save(
        self,
        policy_params: Any,
        rewards: Optional[Sequence[float]] = None,
        meta_data: Optional[dict] = None,
    ):
        self._last_policy_params = policy_params
        self._last_rewards = rewards
        self._last_meta_data = meta_data

        if self.episode_index % self.save_interval_steps == 0:
            self.execute_save()

        self.episode_index += 1

    def execute_save(self):
        if (
            isinstance(self.number_elites, int)
            and self.number_elites > 0
            and self._last_rewards is not None
        ):
            self.save_top_policies()
        else:
            self.save_policy_ensemble()

        if self._last_meta_data is not None:
            self.save_checkpoint_meta()

    def close(self):
        self.episode_index -= 1  # Checkpoint iterated before saving.

        if (
            self.episode_index % self.save_interval_steps != 0
            and self.save_last_at_close
        ):
            self.execute_save()

        self.checkpointer.wait_until_finished()

    def save_checkpoint_meta(self):
        save_path = (
            self.save_directory
            / f"{EPISODE_NAME}_{self.episode_index}"
            / "checkpoint_metadata.json"
        )

        with open(save_path, "w") as f:
            json.dump(self._last_meta_data, f)
