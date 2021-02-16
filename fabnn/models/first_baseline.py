from functools import reduce
from operator import mul
from typing import Dict

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Dense, Input, Lambda
from tensorflow.keras.models import Model

from fabnn.layers.input_preprocess_channel import InputPreprocessChannel
from fabnn.models.utils import get_activation, get_initializer


def make_model(params: Dict):
    feature_shape = tuple(params["feature_shape"])
    levels = len(params["scale_levels"])

    feature_size = reduce(mul, feature_shape)

    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    inputs = [Input(shape=(reduce(mul, feature_shape),))]
    start_block = inputs[0]

    # optional input preprocess channel
    if params.get("preprocess_channel", {}).get("enabled", False):
        conf = params.get("preprocess_channel", {})
        input_preprocess = InputPreprocessChannel(
            polynom2_initializers=conf.get("polynom2_initializers", [0.00, -0.1]),
            polynom1_initializers=conf.get("polynom1_initializers", [0.0, 2.05]),
            bias_initializers=conf.get("bias_initializers", [0.47, -0.42]),
            mask_channel=feature_shape[-1] == 3,
        )
        start_block = input_preprocess(start_block)
        new_feature_shape = input_preprocess.compute_output_shape(feature_shape)
        feature_size = reduce(mul, new_feature_shape)

    start_block = Dense(
        feature_size,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(start_block)
    start_block = Dense(
        feature_size,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(start_block)

    blocks = [start_block]
    for i in range(1, levels):
        inputs.append(Input(shape=(reduce(mul, feature_shape),)))
        dense_in = inputs[i]
        if params.get("preprocess_channel", {}).get("enabled", False):
            dense_in = input_preprocess(dense_in)
        dense_in = Dense(
            feature_size,
            activation=None,
            use_bias=False,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(dense_in)

        layer = Dense(
            feature_size,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(blocks[i - 1])
        layer = Add()([dense_in, layer])
        layer = activation(layer)

        layer = Dense(
            feature_size,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(layer)
        layer = Add()([layer, blocks[i - 1]])  # residual connection
        layer = activation(layer)

        blocks.append(layer)

    prev_layer = blocks[-1]

    for i in range(3):
        output_size = 1 if i == 2 else feature_size
        prev_layer = Dense(
            output_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(prev_layer)

    sum_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="FinalOutputLayer")(
        prev_layer
    )

    return Model(inputs=inputs, outputs=sum_layer)
