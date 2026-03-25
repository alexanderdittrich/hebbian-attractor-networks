from typing import Sequence

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


class FeedforwardNeuralNetwork(FixedWeightNeuralNetwork):
    """Conventional feedforward MLP network."""

    name = "mlp"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        weights_init: str = "uniform_-0.1_0.1",
        bias_init: str = "uniform_-0.1_0.1",
        hidden_activation: str = "tanh",
        output_activation: str = "tanh",
        use_bias: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.weights_init = initializers[weights_init]
        self.bias_init = initializers[bias_init]
        self.hidden_activation = activations[hidden_activation]
        self.output_activation = activations[output_activation]
        self.use_bias = use_bias

    def _init_layer(self, random_key, input_dim: int, output_dim: int):
        rng_w, rng_b = split(random_key, 2)

        # Generate weights
        W = self.weights_init(rng_w, (input_dim, output_dim), DTYPE)

        # Generate bias
        if self.use_bias:
            b = self.bias_init(rng_b, (output_dim,), DTYPE)
            return {"W": W, "b": b}

        return {"W": W}

    def initialize(self, random_key):
        params = {}

        input_dims = [self.input_dim] + self.hidden_dims
        output_dims = self.hidden_dims + [self.output_dim]

        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            random_key, rng_w = split(random_key)
            layer_params = self._init_layer(
                random_key=rng_w, input_dim=in_dim, output_dim=out_dim
            )
            params[f"layer_{i}"] = layer_params

        return {"kernel_params": params}

    def forward(self, params, input):
        kernel_params = params["kernel_params"]
        x = input
        for i in range(len(kernel_params) - 1):
            W = kernel_params[f"layer_{i}"]["W"]
            x = dot(x, W)
            if "b" in kernel_params[f"layer_{i}"]:
                b = kernel_params[f"layer_{i}"]["b"]
                x += b
            x = self.hidden_activation(x)

        # Output layer
        W = kernel_params[f"layer_{len(kernel_params) - 1}"]["W"]
        x = dot(x, W)
        if "b" in kernel_params[f"layer_{len(kernel_params) - 1}"]:
            b = kernel_params[f"layer_{len(kernel_params) - 1}"]["b"]
            x += b
        x = self.output_activation(x)

        return params, x


class HebbianABCDNeuralNetwork(HebbianNeuralNetwork):
    """Conventional feedforward MLP network with Hebbian learning."""

    name = "mlp-hebbian"

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: Sequence[int],
        weights_init: str = "uniform_-0.1_0.1",
        hidden_activation: str = "tanh",
        output_activation: str = "tanh",
        learning_rule_cls: str = "abcd",
        learning_rule_cfg: dict = {},
        activation_buffer_cls: str = "direct",
        activation_buffer_cfg: dict = {},
        use_bias: bool = False,
    ) -> None:
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dims = hidden_dims
        self.weights_init = initializers[weights_init]
        self.hidden_activation = activations[hidden_activation]
        self.output_activation = activations[output_activation]

        update_rule = learning_rules[learning_rule_cls]
        self.update_rule = update_rule(**learning_rule_cfg)

        activation_buffer = activation_buffers[activation_buffer_cls]
        self.activation_buffer = activation_buffer(**activation_buffer_cfg)

        self.use_bias = use_bias
        if use_bias and "bias" not in learning_rule_cls:
            raise ValueError(
                "Learning rule must support bias for use_bias=True"
            )

    def initialize(self, random_key):
        input_dims = [self.input_dim] + self.hidden_dims
        output_dims = self.hidden_dims + [self.output_dim]

        kernel_params = {}
        hebbian_params = {}
        activations = {}
        for i, (in_dim, out_dim) in enumerate(zip(input_dims, output_dims)):
            random_key, rng_kernel, rng_hebbian = split(random_key, 3)

            kernel_params[f"layer_{i}"] = self._init_layer(
                random_key=rng_kernel, input_dim=in_dim, output_dim=out_dim
            )
            hebbian_params[f"layer_{i}"] = self.update_rule.initialize(
                random_key=rng_hebbian,
                input_dim=in_dim,
                output_dim=out_dim,
            )
            activations[f"layer_{i}"] = self.activation_buffer.initialize(
                input_dim=in_dim,
                output_dim=out_dim,
            )

        return {
            "kernel_params": kernel_params,
            "hebbian_params": hebbian_params,
            "activations": activations,
        }

    def _init_layer(self, random_key, input_dim, output_dim):
        rng_w, rng_b = split(random_key, 2)

        W = self.weights_init(rng_w, (input_dim, output_dim), DTYPE)

        if self.use_bias:
            b = self.weights_init(rng_b, (output_dim,), DTYPE)
            return {"W": W, "b": b}

        return {"W": W}

    def forward(self, params, input):
        kernel_params = params["kernel_params"]
        activations = params["activations"]

        x = input
        # Iterate over the layers (hidden layers and output layer)
        for i, layer in enumerate(kernel_params.values()):
            presyn_act = x

            x = dot(x, layer["W"])
            if "b" in layer:
                x += layer["b"]

            if i < len(kernel_params) - 1:  # Hidden layers
                x = self.hidden_activation(x)
            else:  # Output layer
                x = self.output_activation(x)

            postsyn_act = x
            activations[f"layer_{i}"] = self.activation_buffer.add(
                activations[f"layer_{i}"], {"x": presyn_act, "y": postsyn_act}
            )

        # Add activations to params and return the final output
        params["activations"] = activations
        return params, x

    def update(self, params, *args, **kwargs):
        kernel_params = params["kernel_params"]
        hebbian_params = params["hebbian_params"]
        activations = params["activations"]

        for layer in kernel_params.keys():
            kernel_params[layer] = self.update_rule.update(
                kernel_params=kernel_params[layer],
                hebbian_params=hebbian_params[layer],
                activations=self.activation_buffer.get(activations[layer]),
            )

        params["kernel_params"] = kernel_params

        return params

    def reset(self, params, random_key):
        _params = self.initialize(random_key)
        # Only reset kernel params and activations
        params["kernel_params"] = _params["kernel_params"]
        params["activations"] = _params["activations"]
        return params
