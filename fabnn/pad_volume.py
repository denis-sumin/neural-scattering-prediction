import sys

import numpy

from fabnn.utils import dump_discrete_material_voxels


def pad_volume(
    volume, padding_y, padding_x, labels=None, constant_name=None, constant_label=None
):
    if constant_label is None:
        if labels is None or constant_name is None:
            raise ValueError(
                "If constant_label is not set, labels and constant_name should be set"
            )
        for label, name in labels:
            if name == constant_name:
                constant_label = label
        if constant_label is None:
            raise ValueError("{} not found in labels: {}".format(constant_name, labels))
    return numpy.pad(
        volume,
        pad_width=((0, 0), (padding_y, padding_y), (padding_x, padding_x)),
        mode="constant",
        constant_values=constant_label,
    )


if __name__ == "__main__":
    input_discrete_voxels_filepath = sys.argv[1]
    padding_width = int(sys.argv[2])
    output_discrete_voxels_filepath = sys.argv[3]

    npz_file = numpy.load(input_discrete_voxels_filepath)
    discrete_volume = npz_file["halftoned_voxels"]
    labels = list(npz_file["labels"].item().items())
    rendered_image = npz_file["rendered_image"]

    discrete_volume = pad_volume(discrete_volume, padding_width, padding_width, labels, "white")

    dump_discrete_material_voxels(
        voxels=discrete_volume,
        labels=labels,
        materials_dict=None,
        rendered_image=rendered_image,
        filepath=output_discrete_voxels_filepath,
    )
