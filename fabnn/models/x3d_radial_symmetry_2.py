from functools import reduce
from operator import mul
from typing import Dict

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model

from fabnn.layers.channel_crossing import ChannelCrossing
from fabnn.layers.euclidian_distance_channel import EuclidianDistanceChannel
from fabnn.layers.input_preprocess_channel import InputPreprocessChannel
from fabnn.models.utils import get_activation, get_initializer


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

    if params.get("additional_dense_layer", False):
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
        output_size = 1 if i == 2 else feature_size
        layer = Dense(
            output_size,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(layer)

    return Model(inputs=input_, outputs=layer)


def make_model(params: Dict):
    feature_shape = tuple(params["feature_shape"])
    levels = len(params["scale_levels"])

    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    input_ = Input(shape=(levels,) + feature_shape)
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

    input_levels = [Lambda(lambda x: x[:, i, :, :, :, :])(combined_input) for i in range(levels)]

    z_half_dim = feature_shape[0] // 2 + feature_shape[0] % 2
    y_half_dim = feature_shape[1] // 2 + feature_shape[1] % 2
    x_half_dim = feature_shape[2] // 2 + feature_shape[2] % 2
    symmetrical_block_shape = (z_half_dim, y_half_dim, x_half_dim, feature_shape[3])

    oct1, oct2, oct3, oct4, oct5, oct6, oct7, oct8 = get_octants(input_levels[0])

    sym_block = make_symmetrical_block(symmetrical_block_shape, params)
    r1 = sym_block(oct1)
    r2 = sym_block(oct2)
    r3 = sym_block(oct3)
    r4 = sym_block(oct4)
    r5 = sym_block(oct5)
    r6 = sym_block(oct6)
    r7 = sym_block(oct7)
    r8 = sym_block(oct8)
    start_block = Concatenate(axis=1)([r1, r2, r3, r4, r5, r6, r7, r8])

    block_dimension = int(start_block.shape[1])

    blocks = [start_block]
    for i in range(1, levels):
        oct1, oct2, oct3, oct4, oct5, oct6, oct7, oct8 = get_octants(input_levels[i])
        sym_block = make_symmetrical_block(symmetrical_block_shape, params)
        r1 = sym_block(oct1)
        r2 = sym_block(oct2)
        r3 = sym_block(oct3)
        r4 = sym_block(oct4)
        r5 = sym_block(oct5)
        r6 = sym_block(oct6)
        r7 = sym_block(oct7)
        r8 = sym_block(oct8)
        dense_in = Concatenate(axis=1)([r1, r2, r3, r4, r5, r6, r7, r8])

        layer = Dense(
            block_dimension,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(blocks[i - 1])
        layer = Add()([dense_in, layer])
        layer = activation(layer)

        layer = Dense(
            block_dimension,
            activation=None,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(layer)
        layer = Add()([layer, blocks[i - 1]])  # residual connection
        layer = activation(layer)

        blocks.append(layer)

    prev_layer = blocks[-1]

    octant_size = int(r1.shape[1])
    sym_end_block = make_symmetrical_end_block(octant_size, params)
    octants = []
    for idx in range(8):
        octant = Lambda(lambda x: x[:, octant_size * idx : octant_size * (idx + 1)])(prev_layer)
        octants.append(sym_end_block(octant))

    octants = Concatenate(axis=1)(octants)

    sum_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="FinalOutputLayer")(
        octants
    )

    return Model(inputs=input_, outputs=sum_layer)
