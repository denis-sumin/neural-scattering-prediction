from functools import reduce
from operator import mul
from typing import Dict

from tensorflow.keras import backend as K
from tensorflow.keras.layers import Add, Concatenate, Dense, Input, Lambda, Reshape
from tensorflow.keras.models import Model

from fabnn.layers.input_preprocess_channel import InputPreprocessChannel
from fabnn.models.utils import get_activation, get_initializer


def get_quadrants(input_):
    batch_dim, z_dim, y_dim, x_dim, channel_dim = input_.shape
    y_half_dim = int(y_dim) // 2 + y_dim % 2
    x_half_dim = int(x_dim) // 2 + x_dim % 2
    q1 = Lambda(lambda x: x[:, :, :y_half_dim, :x_half_dim, :])(input_)
    q2 = Lambda(lambda x: K.reverse(x[:, :, :y_half_dim, x_half_dim - x_dim % 2 :, :], axes=3))(
        input_
    )
    q3 = Lambda(lambda x: K.reverse(x[:, :, y_half_dim - y_dim % 2 :, :x_half_dim, :], axes=2))(
        input_
    )
    q4 = Lambda(
        lambda x: K.reverse(
            x[:, :, y_half_dim - y_dim % 2 :, x_half_dim - x_dim % 2 :, :], axes=(3, 2)
        )
    )(input_)
    return q1, q2, q3, q4


def make_symmetrical_block(feature_shape, params: Dict):
    feature_size = reduce(mul, feature_shape)

    activation = get_activation(params["activation"])
    bias_initializer = get_initializer(params["bias_initializer"])
    kernel_initializer = get_initializer(params["kernel_initializer"])

    input_ = Input(shape=feature_shape)
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

    input_levels = [Lambda(lambda x: x[:, i, :, :, :, :])(combined_input) for i in range(levels)]

    y_half_dim = feature_shape[1] // 2 + feature_shape[1] % 2
    x_half_dim = feature_shape[2] // 2 + feature_shape[2] % 2
    symmetrical_block_shape = (feature_shape[0], y_half_dim, x_half_dim, feature_shape[3])

    q1, q2, q3, q4 = get_quadrants(input_levels[0])
    sym_block = make_symmetrical_block(symmetrical_block_shape, params)
    r1 = sym_block(q1)
    r2 = sym_block(q2)
    r3 = sym_block(q3)
    r4 = sym_block(q4)
    start_block = Concatenate(axis=1)([r1, r2, r3, r4])

    block_dimension = int(start_block.shape[1])

    blocks = [start_block]
    for i in range(1, levels):
        q1, q2, q3, q4 = get_quadrants(input_levels[i])
        sym_block = make_symmetrical_block(symmetrical_block_shape, params)
        r1 = sym_block(q1)
        r2 = sym_block(q2)
        r3 = sym_block(q3)
        r4 = sym_block(q4)
        dense_in = Concatenate(axis=1)([r1, r2, r3, r4])

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
    for i in range(3):
        prev_layer = Dense(
            block_dimension,
            activation=activation,
            use_bias=True,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        )(prev_layer)

    sum_layer = Lambda(lambda x: K.sum(x, axis=1, keepdims=True), name="FinalOutputLayer")(
        prev_layer
    )

    return Model(inputs=input_, outputs=sum_layer)
