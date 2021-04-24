import sys
from time import time

import numpy
from downscale import downscale_local_mean_sat, generate_scales_sat
from nn_rendering_prediction.data_storage import get_scale_kernels_numpy
from nn_rendering_prediction.summed_area_table import build_volume_sat

PATCH_SIZE = (3, 5, 5)
SCALE_LEVELS = (
    (1, 1),
    (2, 2),
    (4, 3),
    (6, 4),
    (8, 5),
    (12, 6),
    (16, 7),
    (24, 8),
    (32, 9),
    (64, 10),
)
padding_xy = 64 * 5

layers, height, width = 30, 100, 100

volume = numpy.random.rand(layers, height, width, 2).astype(numpy.float32)
volume_sat = build_volume_sat(volume)

volume_pad = numpy.pad(
    volume,
    ((0, 0), (padding_xy, padding_xy), (padding_xy, padding_xy), (0, 0)),
    mode="constant",
    constant_values=0.0,
)
volume_pad_sat = build_volume_sat(volume_pad)

patch_size = PATCH_SIZE

time_all_levels_sat_cpp = 0
time_sat_cpp = 0

scale_kernels_numpy = get_scale_kernels_numpy(SCALE_LEVELS)

for y in range(height):
    print("\r{} / {}".format(y, height), end="")
    sys.stdout.flush()
    for x in range(width):
        s = time()
        levels_sat_cpp = generate_scales_sat(
            scale_kernels_numpy, patch_size, volume_sat, y, x, True
        )
        time_all_levels_sat_cpp += time() - s

        for level_idx, (level_xy, level_z) in enumerate(SCALE_LEVELS):
            w = (patch_size[0] * level_z, patch_size[1] * level_xy, patch_size[2] * level_xy)
            patch = volume[: w[0], y - w[1] // 2 : y + w[1] // 2, x - w[2] // 2 : x + w[2] // 2]
            scale_kernel = (level_z, level_xy, level_xy, 1)

            s = time()
            level_sat_cpp = downscale_local_mean_sat(
                volume_pad_sat, x + padding_xy, y + padding_xy, patch_size, scale_kernel
            )
            time_sat_cpp += time() - s

            assert numpy.allclose(levels_sat_cpp[level_idx], level_sat_cpp), (
                levels_sat_cpp[level_idx][0, 0],
                level_sat_cpp[0, 0],
            )

print()
print("cpp generate_scales_sat_cpp", round(time_all_levels_sat_cpp, 4))
print("cpp downscale_local_mean_cpp", round(time_sat_cpp, 4))
