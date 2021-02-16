from itertools import repeat
from multiprocessing.dummy import Pool as ThreadPool

import numpy


def build_volume_sat(volume: numpy.ndarray):
    volume_sat = numpy.zeros(
        shape=(volume.shape[0] + 1, volume.shape[1] + 1, volume.shape[2] + 1, volume.shape[3]),
        dtype=numpy.float64,
    )

    cumsum = volume_sat[1:, 1:, 1:, :]
    with ThreadPool() as pool:
        # equivalent to volume.cumsum(0).cumsum(1).cumsum(2)
        pool.starmap(
            numpy.cumsum,
            zip(numpy.rollaxis(volume, 1), repeat(0), repeat(None), numpy.rollaxis(cumsum, 1)),
        )
        pool.starmap(
            numpy.cumsum,
            zip(numpy.rollaxis(cumsum, 0), repeat(0), repeat(None), numpy.rollaxis(cumsum, 0)),
        )
        pool.starmap(
            numpy.cumsum,
            zip(numpy.rollaxis(cumsum, 1), repeat(1), repeat(None), numpy.rollaxis(cumsum, 1)),
        )

    return volume_sat


def get_sum_from_sat(v, z1, z2, y1, y2, x1, x2):
    return (
        +v[z2, y2, x2]
        - v[z2, y2, x1]
        - v[z2, y1, x2]
        - v[z1, y2, x2]
        + v[z1, y1, x2]
        + v[z1, y2, x1]
        + v[z2, y1, x1]
        - v[z1, y1, x1]
    )


def downscale_local_mean_sat(volume_sat, patch_start, patch_size, scale_kernel):
    patch_area = scale_kernel[0] * scale_kernel[1] * scale_kernel[2]
    volume_sat_channels = volume_sat.shape[3]
    downscale_sat = numpy.zeros(shape=patch_size + (volume_sat_channels,), dtype=volume_sat.dtype)
    for i in range(patch_size[0]):
        for j in range(patch_size[1]):
            for k in range(patch_size[2]):

                z1 = patch_start[0] + i * scale_kernel[0]
                z2 = z1 + scale_kernel[0]

                y1 = patch_start[1] + j * scale_kernel[1]
                y2 = y1 + scale_kernel[1]

                x1 = patch_start[2] + k * scale_kernel[2]
                x2 = x1 + scale_kernel[2]

                downscale_sat[i, j, k] = get_sum_from_sat(volume_sat, z1, z2, y1, y2, x1, x2)
    return downscale_sat / patch_area
