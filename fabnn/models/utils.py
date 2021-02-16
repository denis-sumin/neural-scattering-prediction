from tensorflow.keras.activations import elu, relu, selu
from tensorflow.keras.layers import LeakyReLU
from tensorflow.python.keras.initializers import Constant, TruncatedNormal


def get_activation(activation_params):
    activation_classes = {
        "relu": relu,
        "selu": selu,
        "elu": elu,
        "leaky_relu": LeakyReLU(activation_params.get("alpha", 0.3)),
    }
    return activation_classes[activation_params["function"]]


def get_initializer(initializer_params):
    if initializer_params["function"] == "truncated_normal":
        return TruncatedNormal(stddev=initializer_params["stddev"])
    elif initializer_params["function"] == "constant":
        return Constant(value=initializer_params["value"])
    else:
        return initializer_params["function"]
