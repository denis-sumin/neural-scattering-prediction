import gc
import sys
from functools import partial
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from random import randint, seed
from time import sleep, time

import numpy
from downscale import generate_scales_sat
from materials import Material, populate_materials
from nn_rendering_prediction.data_storage import Data
from nn_rendering_prediction.data_storage.data import get_scale_kernels_numpy
from nn_rendering_prediction.summed_area_table import build_volume_sat
from sat_tile_tree import SATTileTree2D
from utils import human_size, resolve_project_path

volume_filepath = sys.argv[1]
performance_test_samples = int(sys.argv[2])
volume_layers = int(sys.argv[3]) if len(sys.argv) >= 4 else None


materials_file = resolve_project_path(
    "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
)
d = numpy.load(volume_filepath, allow_pickle=True)

discrete_volume = d["halftoned_voxels"]
if volume_layers is not None:
    discrete_volume = discrete_volume[:volume_layers]
stencil_channels = 2
volume = numpy.empty(shape=discrete_volume.shape + (stencil_channels,), dtype=numpy.float32)

labels_ = d["labels"].item()
labels = dict()
for key, value in dict(labels_).items():
    if isinstance(value, Material):
        labels[key] = value.name
    else:
        labels[key] = value

materials = populate_materials(materials_file)
channel = 0
Data.map_discrete_voxels(discrete_volume, volume, labels, materials, channel, stencil_channels)

print("Volume created", volume.shape)

ts = time()
volume_sat = build_volume_sat(volume)
sat_time = time() - ts
print("SAT", sat_time, "seconds")
sat_size = (
    volume_sat.shape[0]
    * volume_sat.shape[1]
    * volume_sat.shape[2]
    * volume_sat.shape[3]
    * volume_sat.dtype.itemsize
)
print("SAT size", human_size(sat_size))

ts = time()
tree = SATTileTree2D(volume, 32, alignment_z_centered=False)
tree_time = time() - ts
print("SAT Tile Tree", tree_time, "seconds")
print("SAT Tile Tree size", human_size(tree.size()))

sat_from_tile_tree = tree.get_sat()

# print(volume_sat[3:5, 0, 3], volume_sat[0, 3:5, 3], volume_sat[3, 3:5, 0])
print(volume.shape, volume_sat.shape, sat_from_tile_tree.shape)
print("SAT are close", numpy.allclose(volume_sat[1:, 1:, 1:], sat_from_tile_tree))

pad_x, pad_y = 0, 0

patch_size = (3, 5, 5)
SCALE_LEVELS = [
    [1, 1],
    [2, 2],
    [4, 4],
    [6, 6],
    [8, 8],
    [12, 12],
    [16, 16],
    [32, 32],
    [64, 64],
    [128, 128],
]
scale_kernels_numpy = get_scale_kernels_numpy(SCALE_LEVELS)
scale_levels_list = list(SCALE_LEVELS)

time_sat = 0.0
time_tree = 0.0


print("Test against SAT")
seed(0)
diffs = []
for i in range(1000):
    x = randint(0, volume.shape[2])
    y = randint(0, volume.shape[1])

    ts = time()
    res_sat = generate_scales_sat(scale_kernels_numpy, patch_size, volume_sat, y, x, True)
    time_sat += time() - ts

    ts = time()
    res_tree = tree.generate_scales(scale_levels_list, patch_size, 0, y, x)
    time_tree += time() - ts

    # for i in range(len(scale_levels_list)):
    #     print(i, numpy.abs(res_sat[i] - res_tree[i]).mean())
    #
    assert numpy.allclose(res_sat, res_tree)

    # diff = numpy.abs(res_sat - res_tree)
    # diffs.append((diff.mean(), diff.max()))

# diff_mean_mean = sum((mean for mean, max_ in diffs)) / len(diffs)
# diff_max_mean = sum((max_ for mean, max_ in diffs)) / len(diffs)
# print('mean_mean', diff_mean_mean, 'max_mean', diff_max_mean)

print("SAT", time_sat, "seconds", "Tree", time_tree, "seconds")
print(time_tree / time_sat)

print("Test performance")
seed(0)
zyx_positions = [
    (0, randint(0, volume.shape[1]), randint(0, volume.shape[2]))
    for i in range(performance_test_samples)
]
print(len(zyx_positions))
ts = time()
a = list(
    map(
        lambda it: tree.generate_scales(  # noqa: F821
            scale_levels_list, patch_size, it[0], it[1], it[2]
        ),
        zyx_positions,
    )
)
print(time() - ts, "seconds")

ts = time()
a = tree.generate_scales_list(scale_levels_list, patch_size, zyx_positions)
print(time() - ts, "seconds")

for threads in (cpu_count() // 4, cpu_count() // 2, cpu_count()):
    ts = time()
    with Pool(threads) as p:
        p.starmap(partial(tree.generate_scales, scale_levels_list, patch_size), zyx_positions)
    print(threads, "threads", time() - ts)

del tree
gc.collect()
sleep(3)
