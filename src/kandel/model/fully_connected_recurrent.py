from jax.numpy import dot, float32
from jax.random import split

from kandel.activations import activations
from kandel.initializers import initializers
from kandel.model.activation_buffer import activation_buffers
from kandel.model.base_network import (
    FixedWeightNeuralNetwork,
    HebbianNeuralNetwork,
)
from kandel.model.learning_rule import learning_rules

DTYPE = float32


class FullyConnectedRecurrentedNeuralNetwork(FixedWeightNeuralNetwork):
    """Fully connected neural network with recurrent connections."""

    name = "fcrnn"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_neurons: int,
        weights_init: str = "uniform_-0.1_0.1",
        state_init: str = "zeros",
        hidden_activation: str = "tanh",
        use_bias: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neurons = hidden_neurons
        self.weights_init = initializers[weights_init]
        self.state_init = initializers[state_init]
        self.hidden_activation = activations[hidden_activation]
        self.use_bias = use_bias

    def initialize(self, random_key):
        rng_w, rng_state = split(random_key)
        hidden_dims = self.input_dim + self.hidden_neurons + self.output_dim

        kernel_params = {}
        kernel_params["layer_0"] = self._init_layer(
            random_key=rng_w, input_dim=hidden_dims, output_dim=hidden_dims
        )

        state = {}
        state["layer_0"] = self._initialize_states(
            rng_state, state_dim=hidden_dims
        )

        return {"kernel_params": kernel_params, "state": state}

    def _init_layer(self, random_key, input_dim, output_dim):
        rng_w, rng_b = split(random_key, 2)

        W = self.weights_init(rng_w, (input_dim, output_dim), DTYPE)

        if self.use_bias:
            b = self.weights_init(rng_b, (output_dim,), DTYPE)
            return {"W": W, "b": b}

        return {"W": W}

    def _initialize_states(self, random_key, state_dim):
        return {"h_t": self.state_init(random_key, (state_dim,), DTYPE)}

    def forward(self, params, input):
        x = input
        h_prev = params["state"]["layer_0"]["h_t"]
        kernel_params = params["kernel_params"]["layer_0"]

        h_i = h_prev.at[: x.size].add(x)
        h_t = dot(h_i, kernel_params["W"])

        if self.use_bias:
            h_t += kernel_params["b"]
        h_t = self.hidden_activation(h_t)
        y_t = h_t[-self.output_dim :]

        params["state"]["layer_0"]["h_t"] = h_t

        return params, y_t

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        params["state"] = _params["state"]
        return params


class HebbianFullyConnectedRecurrentedNeuralNetwork(HebbianNeuralNetwork):
    """Fully connected neural network with recurrent connections."""

    name = "fcrnn-hebbian"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_neurons: int,
        weights_init: str = "uniform_-0.1_0.1",
        state_init: str = "zeros",
        hidden_activation: str = "tanh",
        learning_rule_cls: str = "abcd",
        learning_rule_cfg: dict = {},
        activation_buffer_cls: str = "direct",
        activation_buffer_cfg: dict = {},
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_neurons = hidden_neurons
        self.weights_init = initializers[weights_init]
        self.state_init = initializers[state_init]
        self.hidden_activation = activations[hidden_activation]

        update_rule = learning_rules[learning_rule_cls]
        self.update_rule = update_rule(**learning_rule_cfg)

        activation_buffer = activation_buffers[activation_buffer_cls]
        self.activation_buffer = activation_buffer(**activation_buffer_cfg)

    def initialize(self, random_key):
        rng_w, rng_hebb, rng_state = split(random_key, 3)
        hidden_dims = self.input_dim + self.hidden_neurons + self.output_dim

        # Network parameters
        kernel_params = {}
        kernel_params["layer_0"] = self._init_layer(
            random_key=rng_w, input_dim=hidden_dims, output_dim=hidden_dims
        )

        # Hebbian parameters
        hebbian_params = {}
        hebbian_params["layer_0"] = self.update_rule.initialize(
            random_key=rng_hebb, input_dim=hidden_dims, output_dim=hidden_dims
        )

        # activations
        activations = {}
        activations["layer_0"] = self.activation_buffer.initialize()

        # State initialization
        state = {}
        state["layer_0"] = self._initialize_state(
            random_key=rng_state, state_dim=hidden_dims
        )

        return {
            "kernel_params": kernel_params,
            "hebbian_params": hebbian_params,
            "activations": activations,
            "state": state,
        }

    def _init_layer(self, random_key, input_dim: int, output_dim: int):
        return {
            "W": self.weights_init(random_key, (input_dim, output_dim), DTYPE)
        }

    def _initialize_state(self, random_key, state_dim: int):
        return {"h_t": self.state_init(random_key, (state_dim,), DTYPE)}

    def forward(self, params, input):
        W = params["kernel_params"]["layer_0"]["W"]
        h_prev = params["state"]["layer_0"]["h_t"]
        x = input

        presyn_act = h_prev

        # Compute the forward pass
        h_t = h_prev.at[: x.size].add(x)
        h_t = dot(h_t, W)
        h_t = self.hidden_activation(h_t)

        postsyn_act = h_t

        y_t = h_t[-self.output_dim :]

        # Store activations and states
        params["activations"]["layer_0"] = self.activation_buffer.add(
            params["activations"]["layer_0"],
            {"x": presyn_act, "y": postsyn_act},
        )
        params["state"]["layer_0"]["h_t"] = h_t

        # forward pass computation
        return params, y_t

    def update(self, params, *args, **kwargs):
        kernel_params = params["kernel_params"]
        hebbian_params = params["hebbian_params"]
        activations = params["activations"]

        kernel_params["layer_0"] = self.update_rule.update(
            kernel_params=kernel_params["layer_0"],
            hebbian_params=hebbian_params["layer_0"],
            activations=self.activation_buffer.get(activations["layer_0"]),
        )

        params["kernel_params"] = kernel_params

        return params

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        # Only reset kernel params
        params["kernel_params"] = _params["kernel_params"]
        params["state"] = _params["state"]
        return params
