import numpy
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Layer


class EuclidianDistanceChannel(Layer):
    """
    @params:
    """

    def __init__(
        self,
        scale_levels,
        patch_size,
        voxel_size,
        add_voxel_volume_channel=False,
        z_centered=True,
        **kwargs,
    ):
        self.scale_levels = scale_levels
        self.kernel_size = numpy.array(patch_size)
        self.voxel_size = voxel_size
        self.incl_volume_channel = add_voxel_volume_channel
        self.z_centered = z_centered
        super(EuclidianDistanceChannel, self).__init__(**kwargs)

    def build(self, input_shape):

        euclidian_channel = numpy.empty(shape=(len(self.scale_levels), *self.kernel_size))
        if self.incl_volume_channel:
            voxel_volume_channel = numpy.ones(shape=(len(self.scale_levels), *self.kernel_size))

        for idx, scale_level in enumerate(self.scale_levels[:]):
            coords = self.euclidian_stencil(scale_level, self.kernel_size)
            euclidian_channel[idx] = coords

            if self.incl_volume_channel:
                voxel_volume_channel[idx] *= numpy.prod(
                    self.voxel_size * [scale_level[1], scale_level[0], scale_level[0]]
                )

        self.euclidian_channel = K.constant(euclidian_channel, name="euclidian_distance")
        if self.incl_volume_channel:
            self.volume_channel = K.constant(voxel_volume_channel, name="voxel_volume")

        super(EuclidianDistanceChannel, self).build(
            input_shape
        )  # Be sure to call this at the end

    def euclidian_stencil(self, scale_level, kernel_size):
        kernel_values = kernel_size.copy()
        kernel_values[0] *= scale_level[1]  # Z
        kernel_values[1] *= scale_level[0]  # Y
        kernel_values[2] *= scale_level[0]  # X

        kernel_val_half = numpy.array(kernel_values, dtype=numpy.float32) / 2
        # z not centered == planar patch
        if not self.z_centered:
            kernel_val_half[0] = 0
            kernel_values[0] -= 1
        # euclidian coordinates
        coords = numpy.meshgrid(
            numpy.linspace(
                -kernel_val_half[0], -kernel_val_half[0] + kernel_values[0], kernel_size[0]
            ),
            numpy.linspace(
                -kernel_val_half[1], -kernel_val_half[1] + kernel_values[1], kernel_size[1]
            ),
            numpy.linspace(
                -kernel_val_half[2], -kernel_val_half[2] + kernel_values[2], kernel_size[2]
            ),
            indexing="ij",
        )
        # convert to mm
        coords = coords * numpy.tile(
            self.voxel_size[..., numpy.newaxis, numpy.newaxis, numpy.newaxis], [1, *kernel_size]
        )
        coords = numpy.sqrt(numpy.sum(numpy.array(coords) ** 2, axis=0))
        # print(coords)
        return coords

    def fit_dimensionality(self, tensor, batch_size):
        tensor = K.expand_dims(tensor)  # channels
        tensor = K.expand_dims(tensor, axis=0)  # batches
        tensor = K.tile(tensor, (batch_size,) + (1,) * 5)  # repeat over batches
        return tensor

    def call(self, x):
        batch_size = K.shape(x)[0]
        tensors = [x, self.fit_dimensionality(self.euclidian_channel, batch_size)]
        if self.incl_volume_channel:
            tensors.append(self.fit_dimensionality(self.volume_channel, batch_size))
        return K.concatenate(tensors, axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (
            input_shape[-1] + 1 + self.incl_volume_channel,
        )  # adding a channel

    def get_config(self):
        config = super().get_config().copy()
        config.update(
            {
                "scale_levels": self.scale_levels,
                "kernel_size": self.kernel_size,
                "voxel_size": self.voxel_size,
                "incl_volume_channel": self.incl_volume_channel,
                "z_centered": self.z_centered,
            }
        )
        return config
