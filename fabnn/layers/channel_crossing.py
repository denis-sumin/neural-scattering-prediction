from tensorflow.keras import backend as K
from tensorflow.keras.initializers import TruncatedNormal
from tensorflow.keras.layers import Layer, Reshape


class ChannelCrossing(Layer):
    """
    @params: initializers are a list of two elements for scattering + absorption channel
    """

    def __init__(self, output_channels, kernel_initializer=None, **kwargs):
        if kernel_initializer is None:
            kernel_initializer = TruncatedNormal(stddev=1 / output_channels)
        self.output_channels = output_channels
        self.kernel_initializer = kernel_initializer
        super().__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.myinput_shape = input_shape
        self.kernel = self.add_weight(
            name="ChannelCrossWeight",
            shape=(input_shape[-1], self.output_channels),
            initializer=self.kernel_initializer,
            trainable=True,
        )
        super(ChannelCrossing, self).build(input_shape)  # Be sure to call this at the end

    def call(self, x, **kwargs):
        channel_voxel = Reshape((K.prod(self.myinput_shape[1:-1]), self.myinput_shape[-1]))(x)
        result = K.dot(channel_voxel, self.kernel)
        return Reshape(self.myinput_shape[1:-1] + (self.output_channels,))(result)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (self.output_channels,)  # altering channels

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "output_channels": self.output_channels,
                "kernel_initializer": self.kernel_initializer,
            }
        )
        return config
