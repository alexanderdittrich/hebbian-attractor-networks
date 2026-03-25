from jax.nn import sigmoid, tanh
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


class GRUNeuralNetwork(FixedWeightNeuralNetwork):
    """GRU-based Recurrent Neural Network."""

    name = "gru-rnn"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        weights_init: str = "uniform_-1.0_1.0",
        state_init: str = "zeros",
        output_activation: str = "tanh",
        use_bias: bool = True,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.weights_init = initializers[weights_init]
        self.state_init = initializers[state_init]
        self.output_activation = activations[output_activation]
        self.use_bias = use_bias

    def initialize(self, random_key):
        rngs = split(random_key, 6)

        # Initialize GRU parameters
        kernel_params = {
            "update_gate": {},
            "reset_gate": {},
            "out_gate": {},
            "layer_ho": {},
        }

        for gate in ["update_gate", "reset_gate", "out_gate"]:
            kernel_params[gate]["W_x"] = self.weights_init(
                rngs[0], (self.input_dim, self.hidden_dim), DTYPE
            )
            kernel_params[gate]["W_h"] = self.weights_init(
                rngs[1], (self.hidden_dim, self.hidden_dim), DTYPE
            )

            if self.use_bias:
                kernel_params[gate]["b"] = self.weights_init(
                    rngs[2], (self.hidden_dim,), DTYPE
                )

        kernel_params["layer_ho"]["W"] = self.weights_init(
            rngs[3], (self.hidden_dim, self.output_dim), DTYPE
        )

        if self.use_bias:
            kernel_params["layer_ho"]["b"] = self.weights_init(
                rngs[4], (self.output_dim,), DTYPE
            )

        state = self._initialize_states(rngs[5], state_dim=self.hidden_dim)
        params = {"kernel_params": kernel_params, "state": state}

        return params

    def _initialize_states(self, random_key, state_dim: int):
        return {"h_t": self.state_init(random_key, (state_dim,), DTYPE)}

    def forward(self, params, input):
        h_t = params["state"]["h_t"]
        kernel_params = params["kernel_params"]

        # Update gate
        z_t = dot(input, kernel_params["update_gate"]["W_x"]) + dot(
            h_t, kernel_params["update_gate"]["W_h"]
        )
        if "b" in kernel_params["update_gate"]:
            z_t += kernel_params["update_gate"]["b"]
        z_t = sigmoid(z_t)

        # Reset gate
        r_t = dot(input, kernel_params["reset_gate"]["W_x"]) + dot(
            h_t, kernel_params["reset_gate"]["W_h"]
        )
        if "b" in kernel_params["reset_gate"]:
            r_t += kernel_params["reset_gate"]["b"]
        r_t = sigmoid(r_t)

        # Candidate new state
        h_hat = dot(input, kernel_params["out_gate"]["W_x"]) + dot(
            r_t * h_t, kernel_params["out_gate"]["W_h"]
        )
        if "b" in kernel_params["out_gate"]:
            h_hat += kernel_params["out_gate"]["b"]
        h_hat = tanh(h_hat)

        # Compute new hidden state
        h_t = (1 - z_t) * h_t + z_t * h_hat

        # Compute output
        y_t = dot(h_t, kernel_params["layer_ho"]["W"])
        if "b" in kernel_params["layer_ho"]:
            y_t += kernel_params["layer_ho"]["b"]
        y_t = self.output_activation(y_t)

        params["state"]["h_t"] = h_t

        return params, y_t

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        params["state"] = _params["state"]
        return params


class HebbianGRUNeuralNetwork(HebbianNeuralNetwork):
    """GRU-based Recurrent Neural Network with Hebbian Learning, without bias."""

    name = "gru-rnn-hebbian"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int,
        weights_init: str = "range_uniform_-1.0_1.0",
        state_init: str = "range_uniform_-1.0_1.0",
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
        self.output_activation = activations[output_activation]

        update_rule = learning_rules[learning_rule_cls]
        self.update_rule = update_rule(**learning_rule_cfg)

        activation_buffer = activation_buffers[activation_buffer_cls]
        self.activation_buffer = activation_buffer(**activation_buffer_cfg)

    def initialize(self, random_key):
        rngs = split(random_key, 15)

        # Initialize kernel parameters with layer-specific naming for weights
        kernel_params = {
            "reset_input": {
                "W": self.weights_init(
                    rngs[0], (self.input_dim, self.hidden_dim), DTYPE
                )
            },
            "reset_hidden": {
                "W": self.weights_init(
                    rngs[1], (self.hidden_dim, self.hidden_dim), DTYPE
                )
            },
            "update_input": {
                "W": self.weights_init(
                    rngs[2], (self.input_dim, self.hidden_dim), DTYPE
                )
            },
            "update_hidden": {
                "W": self.weights_init(
                    rngs[3], (self.hidden_dim, self.hidden_dim), DTYPE
                )
            },
            "new_input": {
                "W": self.weights_init(
                    rngs[4], (self.input_dim, self.hidden_dim), DTYPE
                )
            },
            "new_hidden": {
                "W": self.weights_init(
                    rngs[5], (self.hidden_dim, self.hidden_dim), DTYPE
                )
            },
            "layer_ho": {
                "W": self.weights_init(
                    rngs[6], (self.hidden_dim, self.output_dim), DTYPE
                )
            },
        }

        # Initialize Hebbian parameters
        hebbian_params = {
            "reset_input": self.update_rule.initialize(
                rngs[7], self.input_dim, self.hidden_dim
            ),
            "reset_hidden": self.update_rule.initialize(
                rngs[8], self.hidden_dim, self.hidden_dim
            ),
            "update_input": self.update_rule.initialize(
                rngs[9], self.input_dim, self.hidden_dim
            ),
            "update_hidden": self.update_rule.initialize(
                rngs[10], self.hidden_dim, self.hidden_dim
            ),
            "new_input": self.update_rule.initialize(
                rngs[11], self.input_dim, self.hidden_dim
            ),
            "new_hidden": self.update_rule.initialize(
                rngs[12], self.hidden_dim, self.hidden_dim
            ),
            "layer_ho": self.update_rule.initialize(
                rngs[13], self.hidden_dim, self.output_dim
            ),
        }

        # Initialize activations for Hebbian learning
        activations = {
            "reset_input": self.activation_buffer.initialize(),
            "reset_hidden": self.activation_buffer.initialize(),
            "update_input": self.activation_buffer.initialize(),
            "update_hidden": self.activation_buffer.initialize(),
            "new_input": self.activation_buffer.initialize(),
            "new_hidden": self.activation_buffer.initialize(),
            "layer_ho": self.activation_buffer.initialize(),
        }

        # Initialize hidden state
        state = {
            "layer_hh": self.state_init(rngs[14], (self.hidden_dim,), DTYPE)
        }

        return {
            "kernel_params": kernel_params,
            "hebbian_params": hebbian_params,
            "activations": activations,
            "state": state,
        }

    def forward(self, params, input):
        kernel_params = params["kernel_params"]
        activations = params["activations"]
        h_t = params["state"]["layer_hh"]

        # Reset gate
        r_t = sigmoid(
            dot(input, kernel_params["reset_input"]["W"])
            + dot(h_t, kernel_params["reset_hidden"]["W"])
        )
        activations["reset_input"] = self.activation_buffer.add(
            activations["reset_input"], {"x": input, "y": r_t}
        )
        activations["reset_hidden"] = self.activation_buffer.add(
            activations["reset_hidden"], {"x": h_t, "y": r_t}
        )

        # Update gate
        z_t = sigmoid(
            dot(input, kernel_params["update_input"]["W"])
            + dot(h_t, kernel_params["update_hidden"]["W"])
        )
        activations["update_input"] = self.activation_buffer.add(
            activations["update_input"], {"x": input, "y": z_t}
        )
        activations["update_hidden"] = self.activation_buffer.add(
            activations["update_hidden"], {"x": h_t, "y": z_t}
        )

        # Candidate new state
        h_hat = tanh(
            dot(input, kernel_params["new_input"]["W"])
            + dot(r_t * h_t, kernel_params["new_hidden"]["W"])
        )
        activations["new_input"] = self.activation_buffer.add(
            activations["new_input"], {"x": input, "y": h_hat}
        )
        activations["new_hidden"] = self.activation_buffer.add(
            activations["new_hidden"], {"x": h_t, "y": h_hat}
        )

        # Compute new hidden state
        h_t = (1 - z_t) * h_t + z_t * h_hat
        params["state"]["layer_hh"] = h_t

        # Compute output
        y_t = dot(h_t, kernel_params["layer_ho"]["W"])
        y_t = self.output_activation(y_t)
        activations["layer_ho"] = self.activation_buffer.add(
            activations["layer_ho"], {"x": h_t, "y": y_t}
        )

        params["activations"] = activations
        return params, y_t

    def update(self, params, *args, **kwargs):
        kernel_params = params["kernel_params"]
        hebbian_params = params["hebbian_params"]
        activations = params["activations"]

        # Update weights with Hebbian rules
        for layer in [
            "reset_input",
            "reset_hidden",
            "update_input",
            "update_hidden",
            "new_input",
            "new_hidden",
        ]:
            kernel_params[layer] = self.update_rule.update(
                kernel_params=kernel_params[layer],
                hebbian_params=hebbian_params[layer],
                activations=self.activation_buffer.get(activations[layer]),
            )

        kernel_params["layer_ho"] = self.update_rule.update(
            kernel_params=kernel_params["layer_ho"],
            hebbian_params=hebbian_params["layer_ho"],
            activations=self.activation_buffer.get(activations["layer_ho"]),
        )

        params["kernel_params"] = kernel_params
        return params

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        params["kernel_params"] = _params["kernel_params"]
        params["state"] = _params["state"]
        return params
