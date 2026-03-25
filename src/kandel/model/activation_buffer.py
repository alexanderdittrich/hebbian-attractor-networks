import jax.numpy as jnp


class DirectBuffer:
    name = "direct"

    def __init__(self):
        pass

    def initialize(self, input_dim: int, output_dim: int):
        return {"x": jnp.zeros((input_dim,)), "y": jnp.zeros((output_dim,))}

    def add(self, activation_buffer: dict, activations: dict):
        return {"x": activations["x"], "y": activations["y"]}

    def get(self, activation_buffer: dict):
        # Return the most recent activation stored
        return {"x": activation_buffer["x"], "y": activation_buffer["y"]}


class MovingAverageBuffer:
    name = "moving-average"

    def __init__(self, window_size: int):
        self.window_size = window_size

    def initialize(self, input_dim: int, output_dim: int):
        return {
            "x": jnp.zeros((self.window_size, input_dim)),
            "y": jnp.zeros((self.window_size, output_dim)),
            "count": jnp.array(0, dtype=jnp.uint32),
        }

    def add(self, buffer: dict, activations: dict):
        idx = buffer["count"] % self.window_size
        x = buffer["x"].at[idx].set(activations["x"])
        y = buffer["y"].at[idx].set(activations["y"])

        return {
            "x": x,
            "y": y,
            "count": buffer["count"] + 1,
        }

    def get(self, buffer: dict):
        denom = jnp.minimum(buffer["count"], self.window_size)
        return {
            "x": jnp.sum(buffer["x"], axis=0) / denom,
            "y": jnp.sum(buffer["y"], axis=0) / denom,
        }


class ExponentialMovingAverageBuffer:
    name = "exponential-moving-average"

    def __init__(self, alpha: float):
        self.alpha = alpha

    def initialize(self, input_dim: int, output_dim: int):
        return {"x": jnp.zeros((input_dim,)), "y": jnp.zeros((output_dim,))}

    def add(self, activation_buffer: dict, new_activations: dict):
        # Apply EMA formula for both "x" and "y"
        activation_buffer["x"] = (
            self.alpha * new_activations["x"]
            + (1 - self.alpha) * activation_buffer["x"]
        )
        activation_buffer["y"] = (
            self.alpha * new_activations["y"]
            + (1 - self.alpha) * activation_buffer["y"]
        )

        return activation_buffer

    def get(self, activation_buffer: dict):
        return {"x": activation_buffer["x"], "y": activation_buffer["y"]}


activation_buffers = {
    DirectBuffer.name: DirectBuffer,
    MovingAverageBuffer.name: MovingAverageBuffer,
    ExponentialMovingAverageBuffer.name: ExponentialMovingAverageBuffer,
}
