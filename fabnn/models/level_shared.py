from functools import reduce
from operator import mul
from typing import Dict, Tuple

from nn_rendering_prediction.layers.input_preprocess_channel import InputPreprocessChannel
from nn_rendering_prediction.models.utils import get_activation, get_initializer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model


def make_one_level_model(feature_shape: Tuple, params: Dict):
    feature_size = reduce(mul, feature_shape)

    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    inputs = [
        Input(shape=[feature_size]),
        Input(shape=[feature_size]),
    ]
    prev_block, new_block = inputs

    dense_in = Dense(
        feature_size,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(new_block)

    layer = Dense(
        feature_size,
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(prev_block)

    layer = Add()([dense_in, layer])
    layer = activation(layer)

    layer = Dense(
        feature_size,
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(layer)

    layer = Add()([layer, prev_block])  # residual connection
    layer = activation(layer)

    return Model(inputs=inputs, outputs=layer)


def make_model(params: Dict):
    feature_shape = tuple(params["feature_shape"])
    levels = params["scale_levels"]

    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    input_ = Input(shape=(len(levels),) + feature_shape)
    combined_input = input_

    if params.get("preprocess_channel", {}).get("enabled", False):
        conf = params.get("preprocess_channel", {})
        input_preprocess = InputPreprocessChannel(
            polynom2_initializers=conf.get("polynom2_initializers", [0.00, -0.1]),
            polynom1_initializers=conf.get("polynom1_initializers", [0.0, 2.05]),
            bias_initializers=conf.get("bias_initializers", [0.47, -0.42]),
            mask_channel=feature_shape[-1] == 3,
        )
        combined_input = input_preprocess(combined_input)
        feature_shape = input_preprocess.compute_output_shape(feature_shape)

    input_levels = [
        Lambda(lambda x: x[:, i, :, :, :, :])(combined_input) for i in range(len(levels))
    ]
    feature_size = reduce(mul, feature_shape)

    level_block = make_one_level_model(feature_shape, params)

    blocks = []
    for idx, scale_level in enumerate(levels):
        if len(blocks) == 0:
            zero_layer = Lambda(lambda x: K.zeros_like(x))(input_levels[idx])
            zero_layer = Reshape(
                [
                    feature_size,
                ]
            )(zero_layer)
            prev_block = zero_layer
        else:
            prev_block = blocks[-1]

        assert scale_level[0] == scale_level[1]
        multiplier = scale_level[0] / 64
        multiplier = [multiplier, multiplier] + [
            1,
        ] * (input_levels[idx].shape[-1] - 2)
        new_level = Lambda(lambda x: x * multiplier, name="input_density_mult_{}".format(idx))(
            input_levels[idx]
        )

        new_level = Reshape(
            [
                feature_size,
            ]
        )(new_level)

        layer = level_block([prev_block, new_level])
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

    return Model(inputs=input_, outputs=sum_layer)
