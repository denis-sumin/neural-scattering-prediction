import json
import os

import numpy

from fabnn.utils import resolve_project_path


class Material:

    channels = 3

    def __init__(self, albedo, anisotropy, density, name, wavelengths=None):
        if wavelengths:
            self.wavelengths = numpy.array(wavelengths)
            assert (
                100 < numpy.min(wavelengths) and numpy.max(wavelengths) < 1000
            ), "wavelengths must be in nanometer"
            self.channels = len(wavelengths)
        assert len(albedo) == self.channels, "Albedo should be a list of {} floats".format(
            self.channels
        )
        self.albedo = numpy.array(albedo)
        assert (
            len(anisotropy) == self.channels
        ), "Anisotropy should be a list of {} floats".format(self.channels)
        self.anisotropy = numpy.array(anisotropy)
        assert len(density) == self.channels, "Density should be a list of {} floats".format(
            self.channels
        )
        self.density = numpy.array(density)

        self.scattering = self.albedo * self.density
        self.absorption = self.density - self.scattering

        self.name = name

    def __repr__(self):
        return "Material('{}')".format(self.name)

    @property
    def as_dict(self):
        return dict(
            albedo=self.albedo,
            anisotropy=self.anisotropy,
            density=self.density,
            name=self.name,
        )

    def constrain_to_g(self, target_g):
        for i in range(3):
            s = self.albedo[i] * self.density[i]
            b = self.density[i] - s

            s = (s * (1 - self.anisotropy[i])) / (1 - target_g)
            self.anisotropy[i] = target_g
            self.density[i] = s + b
            self.albedo[i] = s / self.density[i]

    def constrained_albedo(self, target_g):
        result = self.albedo.copy()
        for i in range(3):
            s = self.albedo[i] * self.density[i]
            b = self.density[i] - s

            g = self.anisotropy[i]
            s_new = (s * (1 - g)) / (1 - target_g)
            d_new = s_new + b
            result[i] = s_new / d_new
        return result


def populate_materials(input_file_path: str):
    with open(input_file_path, "r") as f:
        data = json.load(f)

    labels = set()
    materials = dict()
    for item in data:
        label = item["label"]
        if label in labels:
            raise ValueError("Labels in materials file are not unique")
        else:
            labels.add(label)

        materials[label] = Material(
            albedo=item["albedo"],
            anisotropy=item["anisotropy"],
            density=item["density"],
            name=item["name"],
            wavelengths=item.get("wavelengths", None),
        )

    return materials


def check_material_labels(materials, material_labels):
    for label, value in material_labels:
        error_message = "Labels in the data file do not match the materials file: {}; {}".format(
            materials, material_labels
        )
        if isinstance(value, Material):
            assert label in materials and materials[label].name == value.name, error_message
        else:
            assert label in materials and materials[label].name == value, error_message


def update_material_labels(materials, material_labels):
    for label, material in materials.items():
        if (label, material.name) not in material_labels:
            material_labels.add((label, material.name))
    return material_labels


def project_layer_to_gamut(proj, layer):
    if numpy.array_equal(layer, numpy.zeros(shape=layer.shape, dtype=layer.dtype)):
        value = proj.project_image_to_gamut(numpy.zeros((1, 1, 3), dtype=layer.dtype))
        layer[:, :] = value[0, 0]
    else:
        layer[:] = proj.project_image_to_gamut(layer)
        small_random_rgb = numpy.random.rand(3) * 0.1 - 0.05
        layer[0, 0] += small_random_rgb


def convert_rgb_to_cmykw_cont(voxels, rgb_to_cmyk_table, table_sampling):
    # Round to indexes, shape ex. [255, 255, 255, 3]
    ind = numpy.rint((table_sampling - 1) * voxels).astype(numpy.int32)

    # Split into three arrays for the three indexes of the lut
    r = ind[:, :, :, 0]
    g = ind[:, :, :, 1]
    b = ind[:, :, :, 2]

    return rgb_to_cmyk_table[r, g, b]


def convert_cmykw_cont_to_cmykw_discrete(voxels):
    from dither import dither5_error_diffusion, dither5_error_diffusion_3d

    values = numpy.array(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ],
        dtype=numpy.float32,
    )

    error_diffusion = os.getenv("ERROR_DIFFUSION", "2d")
    print("Halftone to one material per voxel, use {} error diffusion".format(error_diffusion))
    if error_diffusion == "2d":
        dithered = dither5_error_diffusion(numpy.array(voxels, dtype=numpy.float32), values)
    elif error_diffusion == "3d":
        dithered = dither5_error_diffusion_3d(numpy.array(voxels, dtype=numpy.float32), values)
    else:
        raise ValueError("Unexpected error_diffusion type {}".format(error_diffusion))

    return dithered


def map_to_materials(voxels, rgb_to_cmykw_table, table_sampling):
    material_labels = tuple(zip(range(5), ("cyan", "magenta", "yellow", "black", "white")))

    voxels_cmyk_cont = convert_rgb_to_cmykw_cont(voxels, rgb_to_cmykw_table, table_sampling)
    voxels_cmyk_discrete = convert_cmykw_cont_to_cmykw_discrete(voxels_cmyk_cont)

    return voxels_cmyk_discrete, set(material_labels)


def map_to_materials_bw(voxels, materials):
    from dither import dither_uniform

    material_labels = tuple(zip(range(len(materials)), materials))

    halftone_materials = {
        "black": 0.3,
        "white": 0.997,
    }

    values = numpy.array(
        [halftone_materials[name] for _, name in material_labels], dtype=numpy.float32
    )

    dithered = dither_uniform(numpy.array(voxels, dtype=numpy.float32), values)
    return dithered, set(material_labels)


def get_correct_labels(materials_file=None):
    if materials_file is None:
        materials_file = resolve_project_path(
            "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
        )
    materials = populate_materials(materials_file)
    return {key: m.name for key, m in materials.items()}
