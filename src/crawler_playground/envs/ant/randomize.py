# Morphological domain randomization for the ant environment.
#
# Follows the mujoco_playground domain-randomisation convention so this
# function works with BraxDomainRandomizationVmapWrapper (Kandel ES) and
# with brax's built-in PPO randomization calling convention.
#
# Two calling modes
# -----------------
# Kandel / BraxDomainRandomizationVmapWrapper (batch_size explicit):
#
#     rand_fn = partial(domain_randomize,
#                       mj_model=env.mj_model,
#                       variants=["none", "FR"],
#                       batch_size=2048)
#     wrapped = BraxDomainRandomizationVmapWrapper(env, rand_fn)
#
# Brax PPO (batch_size from rng shape):
#
#     rand_fn = partial(domain_randomize,
#                       mj_model=env.mj_model,
#                       variants=["none", "FR"])
#     # brax calls: functools.partial(rand_fn, rng=rng_batch)(mjx_model)

from typing import List, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

# ── Variant metadata ───────────────────────────────────────────────────────────

# Maps variant name → ankle geom name in ant.xml.
ANKLE_GEOM: dict[str, str] = {
    "FL": "left_ankle_geom",
    "FR": "right_ankle_geom",
    "RL": "third_ankle_geom",
    "RR": "fourth_ankle_geom",
}

VALID_VARIANTS = frozenset({"none", "FL", "FR", "RL", "RR"})

# Najarro et al. (2020): ankle endpoint shortened to 25 % of original length.
_DAMAGE_SCALE = 0.25
_GEOM_DENSITY = 5.0  # default density in ant.xml

# Only these five mjx.Model fields differ across morphological variants.
_VARYING = ("geom_pos", "geom_size", "body_mass", "body_ipos", "body_inertia")


# ── Physics helpers ────────────────────────────────────────────────────────────


def _capsule_mass(density: float, radius: float, half_length: float) -> float:
    """Mass of a MuJoCo capsule (cylinder + two hemispheres)."""
    cyl_vol = np.pi * radius**2 * 2 * half_length
    sphere_vol = (4 / 3) * np.pi * radius**3
    return density * (cyl_vol + sphere_vol)


def _capsule_principal_inertia(
    density: float, radius: float, half_length: float
) -> tuple[float, float]:
    """Principal moments of inertia for a capsule (axial, transverse)."""
    r, h = radius, half_length
    m_cyl = density * np.pi * r**2 * 2 * h
    m_sph = density * (4 / 3) * np.pi * r**3
    m_hemi = m_sph / 2

    I_axial = m_cyl * r**2 / 2 + (2 / 5) * m_sph * r**2

    I_cyl = m_cyl * (r**2 / 4 + h**2 / 3)
    I_hemi_flat = (2 / 5) * m_hemi * r**2
    I_hemi_com = I_hemi_flat - m_hemi * (3 * r / 8) ** 2
    d = h + 3 * r / 8
    I_hemi_cap = I_hemi_com + m_hemi * d**2
    I_transverse = I_cyl + 2 * I_hemi_cap

    return I_axial, I_transverse


def make_damaged_mjx_model(
    base_mjx_model: mjx.Model,
    mj_model: mujoco.MjModel,
    ankle_geom_name: str,
    scale: float = _DAMAGE_SCALE,
    density: float = _GEOM_DENSITY,
) -> mjx.Model:
    """Return a new mjx.Model with the named ankle geom endpoint scaled.

    Recomputes mass and inertia analytically so the physics remain consistent.
    """
    geom_id = mj_model.geom(ankle_geom_name).id
    body_id = int(mj_model.geom_bodyid[geom_id])

    r = float(base_mjx_model.geom_size[geom_id, 0])
    old_hl = float(base_mjx_model.geom_size[geom_id, 1])
    new_hl = scale * old_hl

    old_pos = np.array(base_mjx_model.geom_pos[geom_id])
    new_pos = scale * old_pos

    new_mass = _capsule_mass(density, r, new_hl)
    I_axial, I_transverse = _capsule_principal_inertia(density, r, new_hl)
    # Preserve MuJoCo's principal-axis ordering from the base model.
    # For ant ankles the axial moment is smallest and lives at index 2.
    base_inertia = np.array(base_mjx_model.body_inertia[body_id])
    axial_idx = int(np.argmin(base_inertia))
    new_inertia = np.where(np.arange(3) == axial_idx, I_axial, I_transverse)

    return base_mjx_model.replace(
        geom_pos=base_mjx_model.geom_pos.at[geom_id].set(new_pos),
        geom_size=base_mjx_model.geom_size.at[geom_id, 1].set(new_hl),
        body_mass=base_mjx_model.body_mass.at[body_id].set(new_mass),
        body_ipos=base_mjx_model.body_ipos.at[body_id].set(new_pos),
        body_inertia=base_mjx_model.body_inertia.at[body_id].set(new_inertia),
    )


# ── Domain randomization ───────────────────────────────────────────────────────


def domain_randomize(
    model: mjx.Model,
    mj_model: mujoco.MjModel,
    variants: List[str],
    batch_size: Optional[int] = None,
    rng: Optional[jax.Array] = None,
) -> tuple[mjx.Model, mjx.Model]:
    """Assign each environment in the batch to a morphological variant.

    Variants are assigned in round-robin order.  With
    ``batch_size = pop_size × n_variants × repeats`` every policy sees every
    variant exactly ``repeats`` times per generation.

    Args:
        model:      Base ``mjx.Model`` (undamaged ant).
        mj_model:   CPU ``mujoco.MjModel`` used for geom-ID lookups.
        variants:   Ordered list of variant names, e.g. ``["none", "FR"]``.
                    ``"none"`` means undamaged.
        batch_size: Total number of parallel environments.  Either this or
                    ``rng`` must be provided.
        rng:        Array of shape ``(batch_size, ...)``; its leading dimension
                    is used as the batch size.  Ignored otherwise.  Passed by
                    brax's PPO randomization calling convention.

    Returns:
        ``(batched_model, in_axes)`` following the mujoco_playground
        ``BraxDomainRandomizationVmapWrapper`` convention.
    """
    if batch_size is None:
        if rng is None:
            raise ValueError("Either batch_size or rng must be provided.")
        batch_size = rng.shape[0]

    n_variants = len(variants)
    variant_models = [
        model
        if v == "none"
        else make_damaged_mjx_model(model, mj_model, ANKLE_GEOM[v])
        for v in variants
    ]

    def tile_field(field: str) -> jax.Array:
        stacked = jnp.stack(
            [getattr(vm, field) for vm in variant_models]
        )  # [n_variants, *field_shape]
        indices = jnp.arange(batch_size) % n_variants
        return stacked[indices]  # [batch_size, *field_shape]

    batched_model = model.tree_replace({f: tile_field(f) for f in _VARYING})
    in_axes = jax.tree_util.tree_map(lambda x: None, model)
    in_axes = in_axes.tree_replace({f: 0 for f in _VARYING})

    return batched_model, in_axes
