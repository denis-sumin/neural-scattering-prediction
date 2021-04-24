from tensorflow.keras import backend as K
from tensorflow.keras.initializers import Constant
from tensorflow.keras.layers import Layer


class InputPreprocessChannel(Layer):
    """
    @params: initializers are a list of two elements for scattering + absorption channel
    """

    def __init__(
        self,
        polynom2_initializers,
        polynom1_initializers,
        bias_initializers,
        mask_channel: bool,
        **kwargs,
    ):
        self.polynom2_initializers = polynom2_initializers
        self.polynom1_initializers = polynom1_initializers
        self.bias_initializers = bias_initializers
        self.mask_channel = mask_channel
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.s_kernel = self.add_weight(
            name="s_polynom1",
            shape=(1,),
            initializer=Constant(self.polynom1_initializers[0]),
            trainable=True,
        )
        self.s_kernel2 = self.add_weight(
            name="s_polynom2",
            shape=(1,),
            initializer=Constant(self.polynom2_initializers[0]),
            trainable=True,
        )
        self.s_bias = self.add_weight(
            name="s_bias",
            shape=(1,),
            initializer=Constant(self.bias_initializers[0]),
            trainable=True,
        )
        self.a_kernel = self.add_weight(
            name="a_polynom1",
            shape=(1,),
            initializer=Constant(self.polynom1_initializers[1]),
            trainable=True,
        )
        self.a_kernel2 = self.add_weight(
            name="a_polynom2",
            shape=(1,),
            initializer=Constant(self.polynom2_initializers[1]),
            trainable=True,
        )
        self.a_bias = self.add_weight(
            name="a_bias",
            shape=(1,),
            initializer=Constant(self.bias_initializers[1]),
            trainable=True,
        )
        super(InputPreprocessChannel, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        result = self.process_scattering_channel(x[..., 0]) - self.process_absorption_channel(
            x[..., 1]
        )
        if self.mask_channel:
            result *= x[..., 2]
        return K.concatenate([x, K.expand_dims(result)], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 1,)  # adding a channel

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "bias_initializers": self.bias_initializers,
                "polynom1_initializers": self.polynom1_initializers,
                "polynom2_initializers": self.polynom2_initializers,
                "mask_channel": self.mask_channel,
            }
        )
        return config

    # polynomial of rank 2
    def process_scattering_channel(self, x):
        return (K.pow(x, 2) * self.s_kernel2) + (x * self.s_kernel) + self.s_bias

    # polynomial of rank 2
    def process_absorption_channel(self, x):
        return (K.pow(x, 2) * self.a_kernel2) + (x * self.a_kernel) + self.a_bias
