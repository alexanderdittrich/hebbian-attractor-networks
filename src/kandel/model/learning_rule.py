from jax.numpy import float32, newaxis, outer, square, where, zeros_like
from jax.random import split

from kandel.initializers import initializers
from kandel.regularizers import regularizers

DTYPE = float32


class etaPlainRule:
    name = "eta-plain"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]

    def initialize(self, random_key, input_dim, output_dim):
        rng_theta, rng_eta = split(random_key, 2)

        return {
            "eta": self.init_fcn(rng_eta, (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        eta = hebbian_params["eta"]

        x = activations["x"]
        y = activations["y"]

        W += eta * outer(x, y)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class OjaABCDRule:
    name = "oja-abcd"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 4)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]
        lr = self.learning_rate

        x = activations["x"]
        y = activations["y"]

        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]

        W += lr * (O_A + O_B + O_C + D - square(y) * W)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class ABCDRule:
    name = "abcd"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 4)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]
        lr = self.learning_rate

        x = activations["x"]
        y = activations["y"]

        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]
        W += lr * (O_A + O_B + O_C + D)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class etaABCDRule:
    name = "eta-abcd"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 5)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "eta": self.init_fcn(rngs[4], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]
        eta = hebbian_params["eta"]

        x = activations["x"]
        y = activations["y"]

        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]
        W += eta * (O_A + O_B + O_C + D)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class ABCDRuleTrace:
    name = "abcd-et"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        trace_decay: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate
        self.trace_decay = trace_decay

        assert 0.0 != trace_decay, "Trace decay cannot be zero!"

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 4)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]

        lr = self.learning_rate
        tau = self.trace_decay

        x = activations["x"]
        y = activations["y"]

        if "e" in kernel_params:
            e = kernel_params["e"]
        else:
            e = zeros_like(W)

        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]
        delta_hebb = lr * (O_A + O_B + O_C + D)

        delta_e = delta_hebb - e / tau
        delta_W = lr * delta_e

        W += delta_W
        e += delta_e

        kernel_params["W"] = self.regulate_fcn(W)
        kernel_params["e"] = e

        return kernel_params


class ABCDRuleBias:
    name = "abcd-bias"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 6)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "E": self.init_fcn(rngs[4], (output_dim,), DTYPE),
            "F": self.init_fcn(rngs[5], (output_dim,), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        lr = self.learning_rate

        W = kernel_params["W"]
        b = kernel_params["b"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]

        x = activations["x"]
        y = activations["y"]

        # Weight update
        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]
        W += lr * (O_A + O_B + O_C + D)
        kernel_params["W"] = self.regulate_fcn(W)

        # Bias update
        E = hebbian_params["E"]
        F = hebbian_params["F"]
        O_E = E * y
        b += lr * (O_E + F)
        kernel_params["b"] = self.regulate_fcn(b)

        return kernel_params


class etaABCDRuleBias:
    name = "eta-abcd-bias"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 8)

        return {
            "A": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "B": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "C": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "D": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "E": self.init_fcn(rngs[4], (output_dim,), DTYPE),
            "F": self.init_fcn(rngs[5], (output_dim,), DTYPE),
            "eta_W": self.init_fcn(rngs[6], (input_dim, output_dim), DTYPE),
            "eta_b": self.init_fcn(rngs[7], (output_dim,), DTYPE),
        }

    def update(
        self, kernel_params, hebbian_params, activations, *args, **kwargs
    ):
        W = kernel_params["W"]
        b = kernel_params["b"]

        x = activations["x"]
        y = activations["y"]

        A = hebbian_params["A"]
        B = hebbian_params["B"]
        C = hebbian_params["C"]
        D = hebbian_params["D"]
        E = hebbian_params["E"]
        F = hebbian_params["F"]
        eta_W = hebbian_params["eta_W"]
        eta_b = hebbian_params["eta_b"]

        # Weight update
        O_A = outer(x, y) * A
        O_B = B * y[newaxis, :]
        O_C = C * x[:, newaxis]
        W += eta_W * (O_A + O_B + O_C + D)
        kernel_params["W"] = self.regulate_fcn(W)

        # Bias update
        O_E = E * y
        b += eta_b * (O_E + F)
        kernel_params["b"] = self.regulate_fcn(b)

        return kernel_params


class BCMRule:
    name = "bcm"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate

    def initialize(self, random_key, output_dim, **kwargs):
        return {"theta": self.init_fcn(random_key, (output_dim,), DTYPE)}

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        theta = hebbian_params["theta"]
        lr = self.learning_rate

        x = activations["x"]
        y = activations["y"]

        y_theta = square(y) - theta
        W += lr * outer(x, y_theta)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class etaBCMRule:
    name = "eta-bcm"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]

    def initialize(self, random_key, input_dim, output_dim):
        rng_theta, rng_eta = split(random_key, 2)

        return {
            "theta": self.init_fcn(rng_theta, (output_dim,), DTYPE),
            "eta": self.init_fcn(rng_eta, (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        theta = hebbian_params["theta"]
        eta = hebbian_params["eta"]

        x = activations["x"]
        y = activations["y"]

        y_theta = square(y) - theta
        W += eta * outer(x, y_theta)

        kernel_params["W"] = self.regulate_fcn(W)

        return kernel_params


class SimpleDHLRule:
    name = "simple-dhl"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate
        self.delta_t = 1.0  # could be replaced with an actual value

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 8)

        return {
            "sigma_pp": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "sigma_pn": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "sigma_np": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "sigma_nn": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "eta_sp": self.init_fcn(rngs[4], (input_dim, output_dim), DTYPE),
            "eta_sn": self.init_fcn(rngs[5], (input_dim, output_dim), DTYPE),
            "eta_ps": self.init_fcn(rngs[6], (input_dim, output_dim), DTYPE),
            "eta_ns": self.init_fcn(rngs[7], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        # Hebbian coefficients
        sigma_pp = hebbian_params["sigma_pp"]
        sigma_pn = hebbian_params["sigma_pn"]
        sigma_np = hebbian_params["sigma_np"]
        sigma_nn = hebbian_params["sigma_nn"]
        eta_sp = hebbian_params["eta_sp"]
        eta_sn = hebbian_params["eta_sn"]
        eta_ps = hebbian_params["eta_ps"]
        eta_ns = hebbian_params["eta_ns"]

        lr = self.learning_rate
        delta_t = self.delta_t

        # activations
        x = activations["x"]
        y = activations["y"]

        if "prev_x" not in kernel_params:
            kernel_params["prev_x"] = zeros_like(x)

        if "prev_y" not in kernel_params:
            kernel_params["prev_y"] = zeros_like(y)

        dx_dt = (x - kernel_params["prev_x"]) / delta_t
        dy_dt = (y - kernel_params["prev_y"]) / delta_t

        self.prev_presyn_act = x
        self.prev_postsyn_act = y

        dx_dt_pos = where(dx_dt < 0.0, 0.0, dx_dt)
        dx_dt_neg = where(dx_dt > 0.0, 0.0, dx_dt)

        dy_dt_pos = where(dy_dt < 0.0, 0.0, dy_dt)
        dy_dt_neg = where(dy_dt > 0.0, 0.0, dy_dt)

        delta_1 = outer(dx_dt_pos, dy_dt_pos) * sigma_pp
        delta_2 = outer(dx_dt_pos, dy_dt_neg) * sigma_pn
        delta_3 = outer(dx_dt_neg, dy_dt_pos) * sigma_np
        delta_4 = outer(dx_dt_neg, dy_dt_neg) * sigma_nn
        delta_5 = outer(x, dy_dt_pos) * eta_sp
        delta_6 = outer(x, dy_dt_neg) * eta_sn
        delta_7 = outer(dx_dt_pos, y) * eta_ps
        delta_8 = outer(dx_dt_neg, y) * eta_ns

        W += lr * (
            delta_1
            + delta_2
            + delta_3
            + delta_4
            + delta_5
            + delta_6
            + delta_7
            + delta_8
        )

        kernel_params["W"] = self.regulate_fcn(W)
        kernel_params["prev_x"] = x
        kernel_params["prev_y"] = y

        return kernel_params


class GDHLRule:
    name = "gdhl"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        learning_rate: float,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.learning_rate = learning_rate
        self.delta_t = 1.0  # could be replaced with an actual value

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 8)

        return {
            "sigma_pp": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "sigma_pn": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "sigma_np": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "sigma_nn": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "eta_sp": self.init_fcn(rngs[4], (input_dim, output_dim), DTYPE),
            "eta_sn": self.init_fcn(rngs[5], (input_dim, output_dim), DTYPE),
            "eta_ps": self.init_fcn(rngs[6], (input_dim, output_dim), DTYPE),
            "eta_ns": self.init_fcn(rngs[7], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        # Hebbian coefficients
        sigma_pp = hebbian_params["sigma_pp"]
        sigma_pn = hebbian_params["sigma_pn"]
        sigma_np = hebbian_params["sigma_np"]
        sigma_nn = hebbian_params["sigma_nn"]
        eta_sp = hebbian_params["eta_sp"]
        eta_sn = hebbian_params["eta_sn"]
        eta_ps = hebbian_params["eta_ps"]
        eta_ns = hebbian_params["eta_ns"]

        lr = self.learning_rate
        delta_t = self.delta_t

        # activations
        x = activations["x"]
        y = activations["y"]

        if "prev_x" not in kernel_params:
            kernel_params["prev_x"] = zeros_like(x)

        if "prev_y" not in kernel_params:
            kernel_params["prev_y"] = zeros_like(y)

        dx_dt = (x - kernel_params["prev_x"]) / delta_t
        dy_dt = (y - kernel_params["prev_y"]) / delta_t

        self.prev_presyn_act = x
        self.prev_postsyn_act = y

        dx_dt_pos = where(dx_dt < 0.0, 0.0, dx_dt)
        dx_dt_neg = where(dx_dt > 0.0, 0.0, dx_dt)

        dy_dt_pos = where(dy_dt < 0.0, 0.0, dy_dt)
        dy_dt_neg = where(dy_dt > 0.0, 0.0, dy_dt)

        delta_1 = outer(dx_dt_pos, dy_dt_pos) * sigma_pp
        delta_2 = outer(dx_dt_pos, dy_dt_neg) * sigma_pn
        delta_3 = outer(dx_dt_neg, dy_dt_pos) * sigma_np
        delta_4 = outer(dx_dt_neg, dy_dt_neg) * sigma_nn
        delta_5 = outer(x, dy_dt_pos) * eta_sp
        delta_6 = outer(x, dy_dt_neg) * eta_sn
        delta_7 = outer(dx_dt_pos, y) * eta_ps
        delta_8 = outer(dx_dt_neg, y) * eta_ns

        W += lr * (
            delta_1
            + delta_2
            + delta_3
            + delta_4
            + delta_5
            + delta_6
            + delta_7
            + delta_8
        )

        kernel_params["W"] = self.regulate_fcn(W)
        kernel_params["prev_x"] = x
        kernel_params["prev_y"] = y

        return kernel_params


class etaGDHLRule:
    name = "eta-gdhl"

    def __init__(
        self,
        coeff_init: str,
        regularizer: str,
        *args,
        **kwargs,
    ):
        self.init_fcn = initializers[coeff_init]
        self.regulate_fcn = regularizers[regularizer]
        self.delta_t = 1.0  # could be replaced with an actual value

    def initialize(self, random_key, input_dim, output_dim):
        rngs = split(random_key, 9)

        return {
            "sigma_pp": self.init_fcn(rngs[0], (input_dim, output_dim), DTYPE),
            "sigma_pn": self.init_fcn(rngs[1], (input_dim, output_dim), DTYPE),
            "sigma_np": self.init_fcn(rngs[2], (input_dim, output_dim), DTYPE),
            "sigma_nn": self.init_fcn(rngs[3], (input_dim, output_dim), DTYPE),
            "eta_sp": self.init_fcn(rngs[4], (input_dim, output_dim), DTYPE),
            "eta_sn": self.init_fcn(rngs[5], (input_dim, output_dim), DTYPE),
            "eta_ps": self.init_fcn(rngs[6], (input_dim, output_dim), DTYPE),
            "eta_ns": self.init_fcn(rngs[7], (input_dim, output_dim), DTYPE),
            "eta": self.init_fcn(rngs[8], (input_dim, output_dim), DTYPE),
        }

    def update(
        self,
        kernel_params,
        hebbian_params,
        activations,
        *args,
        **kwargs,
    ):
        W = kernel_params["W"]

        # Hebbian coefficients
        sigma_pp = hebbian_params["sigma_pp"]
        sigma_pn = hebbian_params["sigma_pn"]
        sigma_np = hebbian_params["sigma_np"]
        sigma_nn = hebbian_params["sigma_nn"]
        eta_sp = hebbian_params["eta_sp"]
        eta_sn = hebbian_params["eta_sn"]
        eta_ps = hebbian_params["eta_ps"]
        eta_ns = hebbian_params["eta_ns"]
        eta = hebbian_params["eta"]

        delta_t = self.delta_t

        # activations
        x = activations["x"]
        y = activations["y"]

        if "prev_x" not in kernel_params:
            kernel_params["prev_x"] = zeros_like(x)

        if "prev_y" not in kernel_params:
            kernel_params["prev_y"] = zeros_like(y)

        dx_dt = (x - kernel_params["prev_x"]) / delta_t
        dy_dt = (y - kernel_params["prev_y"]) / delta_t

        self.prev_presyn_act = x
        self.prev_postsyn_act = y

        dx_dt_pos = where(dx_dt < 0.0, 0.0, dx_dt)
        dx_dt_neg = where(dx_dt > 0.0, 0.0, dx_dt)

        dy_dt_pos = where(dy_dt < 0.0, 0.0, dy_dt)
        dy_dt_neg = where(dy_dt > 0.0, 0.0, dy_dt)

        delta_1 = outer(dx_dt_pos, dy_dt_pos) * sigma_pp
        delta_2 = outer(dx_dt_pos, dy_dt_neg) * sigma_pn
        delta_3 = outer(dx_dt_neg, dy_dt_pos) * sigma_np
        delta_4 = outer(dx_dt_neg, dy_dt_neg) * sigma_nn
        delta_5 = outer(x, dy_dt_pos) * eta_sp
        delta_6 = outer(x, dy_dt_neg) * eta_sn
        delta_7 = outer(dx_dt_pos, y) * eta_ps
        delta_8 = outer(dx_dt_neg, y) * eta_ns

        W += eta * (
            delta_1
            + delta_2
            + delta_3
            + delta_4
            + delta_5
            + delta_6
            + delta_7
            + delta_8
        )

        kernel_params["W"] = self.regulate_fcn(W)
        kernel_params["prev_x"] = x
        kernel_params["prev_y"] = y

        return kernel_params


learning_rules = {
    etaPlainRule.name: etaPlainRule,
    OjaABCDRule.name: OjaABCDRule,
    ABCDRule.name: ABCDRule,
    etaABCDRule.name: etaABCDRule,
    ABCDRuleTrace.name: ABCDRuleTrace,
    ABCDRuleBias.name: ABCDRuleBias,
    etaABCDRuleBias.name: etaABCDRule,
    BCMRule.name: BCMRule,
    etaBCMRule.name: etaBCMRule,
    GDHLRule.name: GDHLRule,
    etaGDHLRule.name: etaGDHLRule,
}
