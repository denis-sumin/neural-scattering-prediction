import sys
from time import time

import numpy
import skimage.transform
from downscale import downscale_local_mean, downscale_local_mean_sat
from nn_rendering_prediction.summed_area_table import (
    build_volume_sat,
    downscale_local_mean_sat as downscale_local_mean_sat_py,
)

PATCH_SIZE = (3, 5, 5)
SCALE_LEVELS = ((2, 2), (4, 3), (6, 4), (8, 5), (12, 6), (16, 7), (24, 8), (32, 9), (64, 10))
padding_xy = 64 * 5

layers, height, width = 30, 10, 10

volume = numpy.random.rand(layers, height, width, 2).astype(numpy.float32)
volume = numpy.pad(
    volume, ((0, 0), (padding_xy, padding_xy), (padding_xy, padding_xy), (0, 0)), mode="edge"
)

volume_sat = build_volume_sat(volume)

patch_size = PATCH_SIZE

time_skimage = 0
time_cpp = 0
time_sat_py = 0
time_sat_cpp = 0

for y in range(padding_xy, height + padding_xy):
    print("\r{} / {}".format(y - padding_xy + 1, height), end="")
    sys.stdout.flush()
    for x in range(padding_xy, width + padding_xy):
        for level_xy, level_z in SCALE_LEVELS:
            w = (patch_size[0] * level_z, patch_size[1] * level_xy, patch_size[2] * level_xy)
            patch = volume[: w[0], y - w[1] // 2 : y + w[1] // 2, x - w[2] // 2 : x + w[2] // 2]
            scale_kernel = (level_z, level_xy, level_xy, 1)

            s = time()
            level_skimage = skimage.transform.downscale_local_mean(
                patch.astype(numpy.float64), scale_kernel
            )
            time_skimage += time() - s

            s = time()
            level_cpp = downscale_local_mean(patch, numpy.array(scale_kernel, dtype=numpy.uint32))
            time_cpp += time() - s

            assert numpy.allclose(level_skimage, level_cpp), (
                level_skimage[0, 0],
                level_cpp[0, 0],
            )

            s = time()
            patch_start = (0, y - w[1] // 2, x - w[2] // 2)
            level_sat_py = downscale_local_mean_sat_py(
                volume_sat, patch_start, patch_size, scale_kernel
            )
            time_sat_py += time() - s

            assert numpy.allclose(level_skimage, level_sat_py), (
                level_skimage[0, 0],
                level_sat_py[0, 0],
            )

            s = time()
            level_sat_cpp = downscale_local_mean_sat(volume_sat, x, y, patch_size, scale_kernel)
            time_sat_cpp += time() - s

            assert numpy.allclose(level_skimage, level_sat_cpp), (
                level_skimage[0, 0],
                level_sat_cpp[0, 0],
            )

print()
print("skimage.transform.downscale_local_mean", round(time_skimage, 4))
print("cpp downscale_local_mean", round(time_cpp, 4))
print("py downscale_local_mean_py", round(time_sat_py, 4))
print("cpp downscale_local_mean_cpp", round(time_sat_cpp, 4))
