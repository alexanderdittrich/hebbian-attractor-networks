import random
from pathlib import Path
from typing import Any, Callable, Optional, Union

import jax
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from scipy.signal import savgol_filter
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from kandel.sim_api.gymnasium import evaluate
from kandel.utility.checkpoint_loader import load_environment, load_policy


def add_mse_network_plasticity(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    add_labels: bool = False,
    eval_fcn: Callable = evaluate,
    apply_savgol: bool = False,
    savgol_window_length: int = 5,
    savgol_polyorder: int = 3,
    env_kwargs: Optional[dict] = {},
    **ax_kwargs,
):
    """Visualizes neural dynamics in original training environment.
    Only works with single policy export."""

    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, network_hist = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_policy=True,
    )

    print(f"Achieved rewards: {rewards}")

    mse_hist = []  # should be 999
    layers = network_hist[0]["kernel_params"].keys()

    for t in range(len(network_hist) - 1):
        mse = 0.0
        for layer in layers:
            params_t1 = network_hist[t]["kernel_params"][layer]["W"]
            params_t2 = network_hist[t + 1]["kernel_params"][layer]["W"]
            mse += np.linalg.norm(params_t2 - params_t1)
        mse_hist.append(mse)

    # Plotting
    if info["update_interval"] >= info["step_interval"]:
        ts = np.arange(len(mse_hist)) * info["step_interval"]
    else:
        ts = np.arange(len(mse_hist)) * info["update_interval"]

    # filtering
    if apply_savgol:
        mse_hist = savgol_filter(
            x=mse_hist,
            window_length=savgol_window_length,
            polyorder=savgol_polyorder,
        )

    ax.plot(ts, mse_hist, **ax_kwargs)

    if add_labels:
        ax.set_xlabel("Time step (s)")
        ax.set_ylabel("MSE of weight changes (-)")


def add_step_rewards(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    eval_fcn: Callable = evaluate,
    add_labels: bool = False,
    env_kwargs: Optional[dict] = {},
    **ax_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, step_rewards = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_stepwise_rewards=True,
    )

    print(f"Achieved rewards: {rewards}")

    ts = np.arange(len(step_rewards)) * info["step_interval"]
    ax.plot(ts, step_rewards, **ax_kwargs)

    if add_labels:
        ax.set_xlabel("Time step (s)")
        ax.set_ylabel("Step rewards (-)")


def add_random_neural_time_series(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    num_neurons: int = 10,
    seed: int = 0,
    selection_seed=0,
    add_legend: bool = True,
    eval_fcn: Callable = evaluate,
    env_kwargs: Optional[dict] = {},
    **ax_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, network_hist = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_policy=True,
    )

    print(f"Achieved rewards: {rewards}")

    mse_hist = []  # should be 999
    layers = list(network_hist[0]["kernel_params"].keys())
    array_shape = network_hist[0]["kernel_params"][layers[0]]["W"].shape

    random.seed(a=selection_seed)

    indices = []
    for _ in range(num_neurons):
        selected_layer = random.choice(layers)
        array_shape = network_hist[0]["kernel_params"][selected_layer][
            "W"
        ].shape

        # Random indices
        ind_1 = random.randint(0, array_shape[0] - 1)
        ind_2 = random.randint(0, array_shape[1] - 1)

        indices.append((selected_layer, ind_1, ind_2))

    neuron_ts = {}
    for i in indices:
        neuron_signal = []
        for t in range(len(network_hist)):
            neuron_signal.append(
                network_hist[t]["kernel_params"][i[0]]["W"][i[1], i[2]]
            )

        neuron_ts[i] = neuron_signal

    for i, neuron_signal in neuron_ts.items():
        ts = np.arange(len(neuron_signal)) * info["step_interval"]
        ax.plot(
            ts,
            neuron_signal,
            label=f"{i[0].replace('_', ' ').replace('l', 'L')}, ({i[1]}, {i[2]})",
            **ax_kwargs,
        )

    ax.set_xlabel("Time step (s)")
    ax.set_ylabel("Weight strength (-)")
    if add_legend:
        ax.legend()


def add_network_2d_pca(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    cmap=cm.Wistia,
    alpha_scatter: float = 0.8,
    eval_fcn: Callable = evaluate,
    env_kwargs: Optional[dict] = {},
    **ax_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, network_hist = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_policy=True,
    )

    print(f"Achieved rewards: {rewards}")

    pca = PCA(n_components=2)
    scaler = StandardScaler()

    flat_hist = []
    for t in range(episode_length):
        params = network_hist[t]["kernel_params"]["layer_0"]["W"].flatten()
        flat_hist.append(params)

    flat_hist = np.array(flat_hist)
    flat_hist = scaler.fit_transform(flat_hist)
    weights_pca = pca.fit_transform(flat_hist)

    # Define the actual time scale (e.g., 0 to episode_length * step_interval)
    time_steps = np.arange(
        0, episode_length * info["step_interval"], info["step_interval"]
    )

    # Update normalization to reflect the time scale
    norm = mcolors.Normalize(vmin=time_steps[0], vmax=time_steps[-1])

    # Plotting
    ax.plot(
        weights_pca[:, 0],
        weights_pca[:, 1],
        **ax_kwargs,
    )
    for t in range(episode_length):
        ax.scatter(
            weights_pca[t, 0],
            weights_pca[t, 1],
            alpha=alpha_scatter,
            color=cmap(norm(t)),
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    # ax.set_aspect("equal")

    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])

    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_label("Time (s)", loc="top")  # Label for the color bar


def add_network_3d_pca(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    cmap=cm.Wistia,
    alpha_scatter: float = 0.8,
    eval_fcn: Callable = evaluate,
    env_kwargs: Optional[dict] = {},
    **ax_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, network_hist = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_policy=True,
    )

    print(f"Achieved rewards: {rewards}")

    pca = PCA(n_components=3)
    scaler = StandardScaler()

    flat_hist = []
    for t in range(1000):
        params = network_hist[t]["kernel_params"]["layer_0"]["W"].flatten()
        flat_hist.append(params)

    flat_hist = np.array(flat_hist)
    flat_hist = scaler.fit_transform(flat_hist)
    weights_pca = pca.fit_transform(flat_hist)

    norm = mcolors.Normalize(vmin=0, vmax=len(weights_pca) - 1)

    # Plotting
    ax.plot(
        weights_pca[:, 0],
        weights_pca[:, 1],
        weights_pca[:, 2],
        **ax_kwargs,
    )
    for t in range(1000):
        ax.scatter(
            weights_pca[t, 0],
            weights_pca[t, 1],
            weights_pca[t, 2],
            alpha=alpha_scatter,
            color=cmap(norm(t)),
        )
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_zlabel("PC 3")

    ax.set_aspect("equal")


def add_histogram_hebbian_parameters(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    add_legend: bool = True,
    **ax_kwargs,
):
    """Generate histogram of Hebbian parameters over all layers."""
    _, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    # Access the Hebbian parameters from the policy parameters
    hebbian_params = policy_params["hebbian_params"]
    layers = list(hebbian_params.keys())

    # Collect Hebbian parameters across all layers
    param_dict = {}  # Key: parameter name, Value: list of values from all layers

    for layer in layers:
        layer_params = hebbian_params[layer]
        for param_name, param_values in layer_params.items():
            if param_name not in param_dict:
                param_dict[param_name] = []
            param_dict[param_name].append(param_values.flatten())

    # Plot histograms for each Hebbian parameter
    colors = plt.cm.tab10.colors  # Color map for different parameters

    for i, (param_name, values_list) in enumerate(param_dict.items()):
        values = np.concatenate(values_list)
        ax.hist(
            values,
            bins=50,
            alpha=0.4,
            label=param_name,
            color=colors[i % len(colors)],
            **ax_kwargs,
        )

    if add_legend:
        ax.set_xlabel("Parameter Value")
        ax.set_ylabel("Frequency")


def generate_heatmap_hebbian_parameters(
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    num_neurons: int = 10,
    seed: int = 0,
    selection_seed=0,
    **ax_kwargs,
):
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    num_layer = ...
    num_hebbian_parameters = ...

    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    axs.set_xlabel("Time step (s)")
    axs.set_ylabel("Weight strength (-)")

    return fig, axs


def add_self_similarity_heatmap(
    ax: Any,
    checkpoint_directory: Union[str, Path],
    checkpoint: int = 999,
    policy_id: int = 0,
    environment_id: Optional[str] = None,
    episode_length: int = 1000,
    seed: int = 0,
    cmap=cm.summer,
    alpha_scatter: float = 0.8,
    eval_fcn: Callable = evaluate,
    env_kwargs: Optional[dict] = {},
    colorbar_range: tuple = (0.0, 200.0),
    **ax_kwargs,
):
    env, info = load_environment(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        environment_id=environment_id,
        **env_kwargs,
    )
    policy, policy_params = load_policy(
        checkpoint_directory=checkpoint_directory,
        checkpoint=checkpoint,
        policy_id=policy_id,
    )

    forward_fcn = jax.jit(policy.forward)
    update_fcn = jax.jit(policy.update)
    reset_fcn = policy.reset

    rng = jax.random.key(seed)
    rng_reset, rng_eval = jax.random.split(rng)

    policy_params = reset_fcn(policy_params, rng_reset)

    rewards, _, _, network_hist = eval_fcn(
        random_key=rng_eval,
        envs=env,
        forward_fcn=forward_fcn,
        update_fcn=update_fcn,
        policy_params=policy_params,
        episode_steps=episode_length,
        environment_interval=info["step_interval"],
        hebbian_update_interval=info["update_interval"],
        record_policy=True,
    )

    layers = list(network_hist[0]["kernel_params"].keys())
    elements = list(network_hist[0]["kernel_params"][layers[0]].keys())

    distance_matrix = np.zeros((episode_length, episode_length))

    ts = np.arange(len(network_hist)) * info["step_interval"]
    yvals = [ts[-1], ts[0]]
    xvals = [ts[0], ts[-1]]

    for i in range(episode_length):
        for j in range(episode_length):
            distance = 0.0
            for layer in layers:
                for element in elements:
                    weights_i = network_hist[i]["kernel_params"][layer][
                        element
                    ]
                    weights_j = network_hist[j]["kernel_params"][layer][
                        element
                    ]

                    # Euclidian norm
                    distance += np.linalg.norm(weights_i - weights_j)

            distance_matrix[i, j] = distance

    heatmap = ax.imshow(
        distance_matrix,
        cmap=cmap,
        interpolation="nearest",
        vmin=colorbar_range[0],
        vmax=colorbar_range[1],
        extent=[xvals[0], xvals[1], yvals[0], yvals[1]],
    )
    cbar = plt.colorbar(heatmap, ax=ax)
    cbar.set_label("$\\ell_{2}-distance$", loc="top")
