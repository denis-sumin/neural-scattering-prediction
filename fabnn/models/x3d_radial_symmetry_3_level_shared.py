from functools import reduce
from operator import mul
from typing import Dict, Tuple

import tensorflow as tf
from nn_rendering_prediction.layers.channel_crossing import ChannelCrossing
from nn_rendering_prediction.layers.euclidian_distance_channel import EuclidianDistanceChannel
from nn_rendering_prediction.layers.input_preprocess_channel import InputPreprocessChannel
from nn_rendering_prediction.models.utils import get_activation, get_initializer
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model


def get_octants(input_):
    batch_dim, z_dim, y_dim, x_dim, channel_dim = input_.shape
    z_half_dim = int(z_dim) // 2 + z_dim % 2
    y_half_dim = int(y_dim) // 2 + y_dim % 2
    x_half_dim = int(x_dim) // 2 + x_dim % 2

    oct1 = Lambda(lambda x: x[:, :z_half_dim, :y_half_dim, :x_half_dim, :])(input_)
    oct2 = Lambda(
        lambda x: K.reverse(x[:, :z_half_dim, :y_half_dim, x_half_dim - x_dim % 2 :, :], axes=3)
    )(input_)
    oct3 = Lambda(
        lambda x: K.reverse(x[:, :z_half_dim, y_half_dim - y_dim % 2 :, :x_half_dim, :], axes=2)
    )(input_)
    oct4 = Lambda(
        lambda x: K.reverse(
            x[:, :z_half_dim, y_half_dim - y_dim % 2 :, x_half_dim - x_dim % 2 :, :], axes=(3, 2)
        )
    )(input_)

    oct5 = Lambda(
        lambda x: K.reverse(x[:, z_half_dim - z_dim % 2 :, :y_half_dim, :x_half_dim, :], axes=1)
    )(input_)
    oct6 = Lambda(
        lambda x: K.reverse(
            x[:, z_half_dim - z_dim % 2 :, :y_half_dim, x_half_dim - x_dim % 2 :, :], axes=(3, 1)
        )
    )(input_)
    oct7 = Lambda(
        lambda x: K.reverse(
            x[:, z_half_dim - z_dim % 2 :, y_half_dim - y_dim % 2 :, :x_half_dim, :], axes=(2, 1)
        )
    )(input_)
    oct8 = Lambda(
        lambda x: K.reverse(
            x[:, z_half_dim - z_dim % 2 :, y_half_dim - y_dim % 2 :, x_half_dim - x_dim % 2 :, :],
            axes=(3, 2, 1),
        )
    )(input_)

    return oct1, oct2, oct3, oct4, oct5, oct6, oct7, oct8


def make_symmetrical_block(feature_shape, params):
    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    input_ = Input(shape=feature_shape)

    feature_size = reduce(mul, feature_shape)
    layer = Reshape(
        [
            feature_size,
        ]
    )(input_)

    layer = Dense(
        feature_size,
        activation=activation,
        use_bias=True,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(layer)

    for _ in range(int(params.get("additional_dense_layer", False))):
        layer = Dense(
            feature_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(layer)

    return Model(inputs=input_, outputs=layer)


def make_symmetrical_end_block(feature_size, params):
    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    input_ = Input(shape=(feature_size,))
    layer = input_

    for i in range(3):
        layer = Dense(
            feature_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(layer)

    return Model(inputs=input_, outputs=layer)


def make_one_level_model(feature_shape: Tuple, params: Dict):
    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    z_half_dim = feature_shape[0] // 2 + feature_shape[0] % 2
    y_half_dim = feature_shape[1] // 2 + feature_shape[1] % 2
    x_half_dim = feature_shape[2] // 2 + feature_shape[2] % 2
    symmetrical_block_shape = (z_half_dim, y_half_dim, x_half_dim, feature_shape[3])
    block_dimension = reduce(mul, symmetrical_block_shape) * 8

    inputs = [
        Input(shape=[block_dimension]),
        Input(shape=feature_shape),
    ]
    prev_block, new_block = inputs

    oct1, oct2, oct3, oct4, oct5, oct6, oct7, oct8 = get_octants(new_block)

    sym_block = make_symmetrical_block(symmetrical_block_shape, params)
    r1 = sym_block(oct1)
    r2 = sym_block(oct2)
    r3 = sym_block(oct3)
    r4 = sym_block(oct4)
    r5 = sym_block(oct5)
    r6 = sym_block(oct6)
    r7 = sym_block(oct7)
    r8 = sym_block(oct8)
    new_block = Concatenate(axis=1)([r1, r2, r3, r4, r5, r6, r7, r8])

    dense_in = Dense(
        block_dimension,
        activation=None,
        use_bias=False,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(new_block)

    layer = Dense(
        block_dimension,
        activation=None,
        kernel_initializer=kernel_initializer,
        bias_initializer=bias_initializer,
    )(prev_block)

    layer = Add()([dense_in, layer])
    layer = activation(layer)

    layer = Dense(
        block_dimension,
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

    if params.get("euclidian_distance_channel", {}).get("enabled", False):
        conf = params.get("euclidian_distance_channel", {})
        distance_channel = EuclidianDistanceChannel(
            scale_levels=params["scale_levels"],
            patch_size=params["patch_size"],
            voxel_size=params["voxel_size"],
            add_voxel_volume_channel=conf.get("add_voxel_volume_channel", False),
        )
        combined_input = distance_channel(combined_input)
        feature_shape = distance_channel.compute_output_shape(feature_shape)

    if params.get("channel_crossing", {}).get("enabled", False):
        conf = params.get("preprocess_channel", {})
        channel_crossing = ChannelCrossing(conf.get("output_channels", feature_shape[-1]))
        combined_input = channel_crossing(combined_input)
        feature_shape = channel_crossing.compute_output_shape(feature_shape)

    input_levels = [
        Lambda(lambda x: x[:, i, :, :, :, :])(combined_input) for i in range(len(levels))
    ]

    level_block = make_one_level_model(feature_shape, params)

    z_half_dim = feature_shape[0] // 2 + feature_shape[0] % 2
    y_half_dim = feature_shape[1] // 2 + feature_shape[1] % 2
    x_half_dim = feature_shape[2] // 2 + feature_shape[2] % 2
    symmetrical_block_shape = (z_half_dim, y_half_dim, x_half_dim, feature_shape[3])
    feature_size = reduce(mul, symmetrical_block_shape) * 8

    blocks = []
    for idx, scale_level in enumerate(levels):
        if idx == 0:
            sh = K.shape(input_levels[idx])
            zero_layer = tf.fill(tf.stack([sh[0], feature_size]), 0.0)
            prev_block = zero_layer
        else:
            prev_block = blocks[-1]

        assert scale_level[0] == scale_level[1]
        channel_multipliers = {
            "scattering": scale_level[0] / (levels[-1][0] / 2),
            "absorption": scale_level[0] / (levels[-1][0] / 2),
            "mask": 1.0,
            "distance": -1.0 / (scale_level[0]),
        }
        multiplier = [channel_multipliers[key] for key in params["stencil_channels"]]
        new_level = Lambda(lambda x: x * multiplier, name="input_density_mult_{}".format(idx))(
            input_levels[idx]
        )

        layer = level_block([prev_block, new_level])
        blocks.append(layer)

    prev_layer = blocks[-1]

    octant_size = reduce(mul, symmetrical_block_shape)
    sym_end_block = make_symmetrical_end_block(octant_size, params)
    octants = []
    for idx in range(8):
        octant = Lambda(lambda x: x[:, octant_size * idx : octant_size * (idx + 1)])(prev_layer)
        octants.append(
            Reshape([z_half_dim, y_half_dim, x_half_dim, feature_shape[3]])(sym_end_block(octant))
        )

    octants_corners_concatenated = Concatenate(axis=1)(
        [Lambda(lambda x: x[:, -1, -1, -1, :])(o) for o in octants]
    )

    sum_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="FinalOutputLayer")(
        octants_corners_concatenated
    )

    return Model(inputs=input_, outputs=sum_layer)
