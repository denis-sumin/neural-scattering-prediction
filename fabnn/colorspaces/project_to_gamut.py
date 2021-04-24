import pickle
import sys
from functools import partial

import numpy
from colorspaces.cmykw_to_rgb import fm_mixture
from materials import populate_materials
from utils import dump_image, read_image

input_file = sys.argv[1]
output_file = sys.argv[2]
materials_file = sys.argv[3]
lookup_table_file = sys.argv[4]
lookup_table_sampling = int(sys.argv[5])

g = 0.4
materials = populate_materials(materials_file)
for name, m in materials.items():
    assert tuple(m.anisotropy) == (g,) * 3
materials_order = (0, 1, 2, 3, 4)
material_labels = tuple(zip(range(len(materials_order)), materials_order))

print("Load rgb -> cmykw table")
with open(lookup_table_file, "rb") as f:
    lookup_table = pickle.load(f)

image = read_image(input_file, convert_to_grayscale=False)
output_image = numpy.empty(shape=image.shape, dtype=image.dtype)

cache = dict()
print("Build image")
forward_mixture = partial(fm_mixture, materials, material_labels)
for i, row in enumerate(image):
    for j, p in enumerate(row):
        try:
            rgb_value = cache[tuple(p)]
        except KeyError:
            key = tuple(
                (
                    float(round(item * (lookup_table_sampling - 1)) / (lookup_table_sampling - 1))
                    for item in p
                )
            )
            cmykw = lookup_table[key]
            rgb_value = forward_mixture(cmykw)
            cache[tuple(p)] = rgb_value
        output_image[i, j] = rgb_value

dump_image(output_image, output_file)
