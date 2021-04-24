from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer
from tensorflow.math import divide_no_nan


class AlbedoMappingChannel(Layer):
    def __init__(self, **kwargs):
        super(AlbedoMappingChannel, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        super(AlbedoMappingChannel, self).build(input_shape)  # Be sure to call this at the end

    def map_albedo(self, a):
        # Eq 4 [Elek, Sumin 2017]
        return (
            0.065773 * K.pow(a, 1.569383)
            + 0.201198 * K.pow(a, 6.802855)
            + 0.279264 * K.pow(a, 28.61815)
            + 0.251997 * K.pow(a, 142.0079)
            + 0.201767 * K.pow(a, 1393.165)
        )

    def call(self, x):
        albedo = divide_no_nan(x[..., 0], (x[..., 0] + x[..., 1]))
        result = self.map_albedo(albedo)
        return K.concatenate([x, K.expand_dims(result)], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (input_shape[-1] + 1,)  # adding a channel
