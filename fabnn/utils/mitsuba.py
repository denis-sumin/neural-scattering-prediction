import os
import struct
from typing import Mapping

import numpy
from utils import ensure_dir


def write_mitsuba_vol_file_header(
    f,
    res_x: int,
    res_y: int,
    res_z: int,
    channels: int,
    size_x_min: float,
    size_y_min: float,
    size_z_min: float,
    size_x_max: float,
    size_y_max: float,
    size_z_max: float,
):
    mitsuba_version = 3
    mitsuba_format = 1

    f.write(b"VOL")
    f.write(struct.pack("B", mitsuba_version))
    f.write(struct.pack("i", mitsuba_format))
    f.write(struct.pack("iii", res_x, res_y, res_z))
    f.write(struct.pack("i", channels))
    f.write(struct.pack("fff", size_x_min, size_y_min, size_z_min))
    f.write(struct.pack("fff", size_x_max, size_y_max, size_z_max))


def write_mitsuba_format_vol_file(
    filename: str,
    voxels: numpy.ndarray,
    sample_width=None,
    sample_height=None,
    sample_thickness=None,
    bbox_min=None,
    bbox_max=None,
) -> None:
    ensure_dir(os.path.dirname(filename))

    if bbox_min is not None and bbox_max is not None:
        size_x_min, size_y_min, size_z_min = bbox_min
        size_x_max, size_y_max, size_z_max = bbox_max
    elif sample_width is not None and sample_height is not None and sample_thickness is not None:
        size_x_min, size_x_max = -sample_width / 2.0, sample_width / 2.0
        size_y_min, size_y_max = -sample_height / 2.0, sample_height / 2.0
        size_z_min, size_z_max = -sample_thickness / 2.0, sample_thickness / 2.0
    else:
        raise ValueError(
            "Either bbox_min, bbox_max or "
            "sample_width, sample_height, sample_thickness "
            "should be specified"
        )

    if len(voxels.shape) == 4:
        res_z, res_y, res_x, channels = voxels.shape
    elif len(voxels.shape) == 3:
        res_z, res_y, res_x = voxels.shape
        channels = 1
    else:
        raise ValueError("voxels should have either 3 or 4 dimensions")

    assert voxels.dtype == numpy.float32, "voxels must be 32bit float"

    with open(filename, "wb") as f:
        write_mitsuba_vol_file_header(
            f,
            res_x,
            res_y,
            res_z,
            channels,
            size_x_min,
            size_y_min,
            size_z_min,
            size_x_max,
            size_y_max,
            size_z_max,
        )
        f.write(voxels.tobytes(order="C"))


def label_grid_to_mitsuba_volgrid(
    volgrid_filepath: str, label_grid, material_label_to_value_map: Mapping[int, float]
) -> None:
    ensure_dir(os.path.dirname(volgrid_filepath))

    size_x_min, size_y_min, size_z_min = label_grid.bbox_min
    size_x_max, size_y_max, size_z_max = label_grid.bbox_max

    res_z, res_y, res_x = label_grid.res_z, label_grid.res_y, label_grid.res_x
    channels = 1

    layer_continous = numpy.empty((res_y, res_x, channels), dtype=numpy.float32)

    with open(volgrid_filepath, "wb") as f:
        write_mitsuba_vol_file_header(
            f,
            res_x,
            res_y,
            res_z,
            channels,
            size_x_min,
            size_y_min,
            size_z_min,
            size_x_max,
            size_y_max,
            size_z_max,
        )
        for layer_discrete in label_grid.data:
            for label in material_label_to_value_map.keys():
                layer_continous[layer_discrete == label] = material_label_to_value_map[label]
            f.write(layer_continous.tobytes(order="C"))
