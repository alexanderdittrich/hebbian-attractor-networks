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


class ElmanRecurrentNeuralNetwork(FixedWeightNeuralNetwork):
    """Simple Recurrent Neural Network."""

    name = "elman-rnn"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        weights_init: str = "uniform_-1.0_1.0",
        state_init: str = "zeros",
        hidden_activation: str = "tanh",
        output_activation: str = "tanh",
        use_bias: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.weights_init = initializers[weights_init]
        self.state_init = initializers[state_init]
        self.hidden_activation = activations[hidden_activation]
        self.output_activation = activations[output_activation]
        self.use_bias = use_bias

    def initialize(self, random_key):
        rngs = split(random_key, 6)

        # Initialize input to hidden weights
        kernel_params = {"layer_xh": {}, "layer_hh": {}, "layer_ho": {}}

        kernel_params["layer_xh"]["W"] = self.weights_init(
            rngs[0], (self.input_dim, self.hidden_dim), DTYPE
        )

        # Initialize hidden to hidden weights
        kernel_params["layer_hh"]["W"] = self.weights_init(
            rngs[1], (self.hidden_dim, self.hidden_dim), DTYPE
        )

        # Initialize hidden to output weights
        kernel_params["layer_ho"]["W"] = self.weights_init(
            rngs[2], (self.hidden_dim, self.output_dim), DTYPE
        )

        if self.use_bias:
            # Initialize biases
            kernel_params["layer_hh"]["b"] = self.weights_init(
                rngs[3], (self.hidden_dim,), DTYPE
            )
            kernel_params["layer_ho"]["b"] = self.weights_init(
                rngs[4], (self.output_dim,), DTYPE
            )
        else:
            kernel_params["layer_hh"]["b"] = None
            kernel_params["layer_ho"]["b"] = None

        state = self._initialize_states(rngs[5], state_dim=self.hidden_dim)
        params = {"kernel_params": kernel_params, "state": state}

        return params

    def _initialize_states(self, random_key, state_dim: int):
        return {"h_t": self.state_init(random_key, (state_dim,), DTYPE)}

    def forward(self, params, input):
        h_t = params["state"]["h_t"]
        kernel_params = params["kernel_params"]

        # Compute hidden state
        h_t = dot(input, kernel_params["layer_xh"]["W"]) + dot(
            h_t, kernel_params["layer_hh"]["W"]
        )
        if kernel_params["layer_hh"]["b"] is not None:
            h_t += kernel_params["layer_hh"]["b"]
        h_t = self.hidden_activation(h_t)

        # Compute output
        y_t = dot(h_t, kernel_params["layer_ho"]["W"])
        if kernel_params["layer_ho"]["b"] is not None:
            y_t += kernel_params["layer_ho"]["b"]
        y_t = self.output_activation(y_t)

        params["state"]["h_t"] = h_t

        return params, y_t

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        params["state"] = _params["state"]
        return params


class HebbianElmanRecurrentNeuralNetwork(HebbianNeuralNetwork):
    """Simple Recurrent Neural Network."""

    name = "elman-rnn-hebbian"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        weights_init: str = "range_uniform_-1.0_1.0",
        state_init: str = "range_uniform_-1.0_1.0",
        hidden_activation: str = "tanh",
        output_activation: str = "tanh",
        learning_rule_cls: str = "abcd",
        learning_rule_cfg: dict = {},
        activation_buffer_cls: str = "direct",
        activation_buffer_cfg: dict = {},
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.weights_init = initializers[weights_init]
        self.state_init = initializers[state_init]
        self.hidden_activation = activations[hidden_activation]
        self.output_activation = activations[output_activation]

        update_rule = learning_rules[learning_rule_cls]
        self.update_rule = update_rule(**learning_rule_cfg)

        activation_buffer = activation_buffers[activation_buffer_cls]
        self.activation_buffer = activation_buffer(**activation_buffer_cfg)

    def initialize(self, random_key):
        rngs = split(random_key, 7)

        # Initialize kernel parameters
        kernel_params = {}
        kernel_params["layer_xh"] = self._init_layer(
            random_key=rngs[0],
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
        )
        kernel_params["layer_hh"] = self._init_layer(
            random_key=rngs[1],
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        kernel_params["layer_ho"] = self._init_layer(
            random_key=rngs[2],
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

        # Initialize Hebbian coefficients
        hebbian_params = {}
        hebbian_params["layer_xh"] = self.update_rule.initialize(
            random_key=rngs[3],
            input_dim=self.input_dim,
            output_dim=self.hidden_dim,
        )
        hebbian_params["layer_hh"] = self.update_rule.initialize(
            random_key=rngs[4],
            input_dim=self.hidden_dim,
            output_dim=self.hidden_dim,
        )
        hebbian_params["layer_ho"] = self.update_rule.initialize(
            random_key=rngs[5],
            input_dim=self.hidden_dim,
            output_dim=self.output_dim,
        )

        activations = {}
        activations["layer_xh"] = self.activation_buffer.initialize()
        activations["layer_hh"] = self.activation_buffer.initialize()
        activations["layer_ho"] = self.activation_buffer.initialize()

        # Initialize hidden state
        state = {}
        state["layer_hh"] = self._initialize_state(
            random_key=rngs[6], state_dim=self.hidden_dim
        )

        return {
            "kernel_params": kernel_params,
            "hebbian_params": hebbian_params,
            "activations": activations,
            "state": state,
        }

    def _initialize_state(self, random_key, state_dim: int):
        return self.state_init(random_key, (state_dim,), DTYPE)

    def _init_layer(self, random_key, input_dim: int, output_dim: int):
        return {
            "W": self.weights_init(random_key, (input_dim, output_dim), DTYPE)
        }

    def forward(self, params, input):
        kernel_params = params["kernel_params"]
        activations = params["activations"]
        h_t = params["state"]["layer_hh"]

        x = input

        # Compute hidden state
        presyn_act_xh = x
        presyn_act_hh = h_t
        h_t = dot(x, kernel_params["layer_xh"]["W"]) + dot(
            h_t, kernel_params["layer_hh"]["W"]
        )
        h_t = self.hidden_activation(h_t)
        postsyn_act_xh = h_t
        postsyn_act_hh = h_t

        # Compute output
        presyn_act_ho = h_t
        y_t = dot(h_t, kernel_params["layer_ho"]["W"])
        y_t = self.output_activation(y_t)
        postsyn_act_ho = y_t

        # Add activations
        activations["layer_xh"] = self.activation_buffer.add(
            activations["layer_xh"], {"x": presyn_act_xh, "y": postsyn_act_xh}
        )
        activations["layer_hh"] = self.activation_buffer.add(
            activations["layer_hh"], {"x": presyn_act_hh, "y": postsyn_act_hh}
        )
        activations["layer_ho"] = self.activation_buffer.add(
            activations["layer_ho"], {"x": presyn_act_ho, "y": postsyn_act_ho}
        )

        params["activations"] = activations
        params["state"]["layer_hh"] = h_t

        return params, y_t

    def update(self, params, *args, **kwargs):
        kernel_params = params["kernel_params"]
        hebbian_params = params["hebbian_params"]
        activations = params["activations"]

        # Input to hidden
        kernel_params["layer_xh"] = self.update_rule.update(
            kernel_params=kernel_params["layer_xh"],
            hebbian_params=hebbian_params["layer_xh"],
            activations=self.activation_buffer.get(activations["layer_xh"]),
        )

        # Hidden to hidden
        kernel_params["layer_hh"] = self.update_rule.update(
            kernel_params=kernel_params["layer_hh"],
            hebbian_params=hebbian_params["layer_hh"],
            activations=self.activation_buffer.get(activations["layer_hh"]),
        )

        # Hidden to output
        kernel_params["layer_ho"] = self.update_rule.update(
            kernel_params=kernel_params["layer_ho"],
            hebbian_params=hebbian_params["layer_ho"],
            activations=self.activation_buffer.get(activations["layer_ho"]),
        )

        params["kernel_params"] = kernel_params

        return params

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        # Only reset kernel params
        params["kernel_params"] = _params["kernel_params"]
        params["state"] = _params["state"]
        return params
