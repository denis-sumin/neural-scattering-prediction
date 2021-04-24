import gzip
import os
import struct
from typing import Iterable, Mapping, Optional, Sequence, Tuple

import numpy
import utils.mitsuba

from fabnn.materials import Material
from fabnn.utils import ensure_dir


class LabelGrid:

    PATCH_TYPES = ["labels", "scattering_absorption"]

    def __init__(
        self,
        file_path: Optional[str] = None,
        voxel_grid: Optional[numpy.ndarray] = None,
        bbox_min: Optional[Sequence] = None,
        bbox_max: Optional[Sequence] = None,
    ) -> None:
        self.data = None
        self.data_s_b = None
        self.materials = None
        self.res_x, self.res_y, self.res_z = 3 * (None,)
        self.bbox_min, self.bbox_max = 2 * (None,)
        self.voxel_size = None
        self.channels = ("r", "g", "b")

        if file_path is not None and voxel_grid is not None:
            raise ValueError("file_path and voxel_grid cannot be used simultaneously")

        if file_path is not None:
            self.load_from_file(file_path)

        if voxel_grid is not None:
            if bbox_min is not None and bbox_max is not None:
                self.data = voxel_grid
                self.res_z, self.res_y, self.res_x = voxel_grid.shape[:3]
                self.bbox_min = numpy.array(bbox_min)
                self.bbox_max = numpy.array(bbox_max)
            else:
                raise ValueError("voxel_grid is given, but bbox_min and bbox_max are not")

        if self.data is not None:
            voxel_size = (self.bbox_max - self.bbox_min) / numpy.array(
                [self.res_x, self.res_y, self.res_z]
            )
            self.voxel_size = tuple(voxel_size)

    def __repr__(self):
        return """ LabelGrid: data[{}] ({},{},{}), [{},{}]""".format(
            self.data.shape,
            self.res_x,
            self.res_y,
            self.res_z,
            self.bbox_min,
            self.bbox_max,
        )

    @staticmethod
    def __get_open_func(file_path: str):
        if file_path.endswith(".gz"):
            return gzip.open
        elif file_path.endswith(".bin"):
            return open
        else:
            raise ValueError("Labelgrid files are either .bin (no compression) or .gz")

    def convert_label_to_s_b(self, materials):
        dtype = numpy.float32
        self.materials = materials
        data_s_b = [
            numpy.empty(shape=self.data.shape + (2,), dtype=dtype)
            for _ in range(Material.channels)
        ]
        for c in range(Material.channels):
            for label, m in materials.items():
                data_s_b[c][self.data == label] = numpy.array(
                    (m.scattering[c], m.absorption[c]), dtype=dtype
                )
        self.data_s_b = numpy.array(data_s_b)

    def get_patch_around(
        self,
        coord_w: Sequence[float],
        patch_dims: Sequence[int],
        padding_label: int,
        patch_type: str,
    ) -> numpy.ndarray:
        """
        Get a patch of the grid around the give coord_w
        :param coord_w: (x_w, y_w, z_w)
        :param patch_dims: (x_dim, y_dim, z_dim)
        :param padding_label: material which is used to pad the volume
        :param patch_type: 'labels' or 'scattering_absorption':
                           if the data should contain material labels or their
                           scattering-absorption values
        :return:
        """
        coord_idx = self.world_to_index(coord_w)
        radius, remainder = numpy.divmod(numpy.array(patch_dims), 2)

        bbox_min = coord_idx - radius
        bbox_max = coord_idx + radius + remainder
        bbox_min_clip = bbox_min.clip(min=0)
        bbox_max_clip = bbox_max.clip(
            max=(self.data.shape[2], self.data.shape[1], self.data.shape[0])
        )
        res_start_coord = bbox_min_clip - bbox_min
        res_end_coord = bbox_max_clip - bbox_min_clip + res_start_coord

        if patch_type == "labels":
            res = (
                numpy.ones(
                    shape=(patch_dims[2], patch_dims[1], patch_dims[0]),
                    dtype=self.data.dtype,
                )
                * padding_label
            )

            res[
                res_start_coord[2] : res_end_coord[2],
                res_start_coord[1] : res_end_coord[1],
                res_start_coord[0] : res_end_coord[0],
            ] = self.data[
                bbox_min_clip[2] : bbox_max_clip[2],
                bbox_min_clip[1] : bbox_max_clip[1],
                bbox_min_clip[0] : bbox_max_clip[0],
            ]
        elif patch_type == "scattering_absorption":
            try:
                res = numpy.empty(
                    shape=(
                        Material.channels,
                        patch_dims[2],
                        patch_dims[1],
                        patch_dims[0],
                        2,
                    ),
                    dtype=self.data_s_b.dtype,
                )
                for c in range(Material.channels):
                    res[c, :, :, :] = numpy.array(
                        (
                            self.materials[padding_label].scattering[c],
                            self.materials[padding_label].absorption[c],
                        ),
                        dtype=self.data_s_b.dtype,
                    )
            except TypeError:  # either data_s_b or materials not initialized
                raise RuntimeError("scattering_absorption volume was not calculated")
            try:
                res[
                    :,
                    res_start_coord[2] : res_end_coord[2],
                    res_start_coord[1] : res_end_coord[1],
                    res_start_coord[0] : res_end_coord[0],
                ] = self.data_s_b[
                    :,
                    bbox_min_clip[2] : bbox_max_clip[2],
                    bbox_min_clip[1] : bbox_max_clip[1],
                    bbox_min_clip[0] : bbox_max_clip[0],
                ]
            except ValueError:
                print(self.data_s_b.shape, res.shape)
                print(bbox_min, bbox_min_clip, bbox_max, bbox_max_clip, res_start_coord)
                raise
        else:
            raise ValueError("patch_type should be one of:", self.PATCH_TYPES)

        return res

    def load_from_file(self, file_path: str) -> None:
        open_func = self.__get_open_func(file_path)
        with open_func(file_path, "rb") as f:
            res_x, res_y, res_z = struct.unpack("iii", f.read(struct.calcsize("iii")))
            bbox_min = struct.unpack("ddd", f.read(struct.calcsize("ddd")))
            bbox_max = struct.unpack("ddd", f.read(struct.calcsize("ddd")))
            label_grid = numpy.frombuffer(f.read(), dtype=numpy.uint8).reshape(
                (res_z, res_y, res_x), order="C"
            )

        self.res_x, self.res_y, self.res_z = res_x, res_y, res_z
        self.bbox_min = numpy.array(bbox_min)
        self.bbox_max = numpy.array(bbox_max)
        self.data = label_grid

    def write_to_file(self, file_path: str) -> None:
        ensure_dir(os.path.dirname(file_path))
        open_func = self.__get_open_func(file_path)

        bbox_min = tuple(self.bbox_min)
        bbox_max = tuple(self.bbox_max)

        with open_func(file_path, "wb") as f:
            f.write(struct.pack("iii", self.res_x, self.res_y, self.res_z))
            f.write(struct.pack("ddd", *bbox_min))
            f.write(struct.pack("ddd", *bbox_max))
            f.write(self.data.tobytes(order="C"))

    def write_mitsuba_vol_file(
        self,
        volgrid_albedo_filepath: str,
        albedo_label_to_value_map: Mapping[int, float],
        volgrid_density_filepath: str,
        density_label_to_value_map: Mapping[int, float],
        channel_idx: int,
        max_density: float,
    ):
        utils.mitsuba.label_grid_to_mitsuba_volgrid(
            volgrid_albedo_filepath, self, albedo_label_to_value_map
        )
        utils.mitsuba.label_grid_to_mitsuba_volgrid(
            volgrid_density_filepath, self, density_label_to_value_map
        )

    def world_to_index(self, coord_w: Iterable[float]) -> Tuple[int, int, int]:
        """
        :param coord_w: (x_w, y_w, z_w)
        :return: (x_idx, y_idx, z_idx)
        """
        coord_w = numpy.array(coord_w)
        if (self.bbox_min <= coord_w).all() and (coord_w <= self.bbox_max).all():
            dims_w = self.bbox_max - self.bbox_min
            dims_idx = numpy.array([self.res_x, self.res_y, self.res_z])
            coord_idx = numpy.floor((coord_w - self.bbox_min) / (dims_w / dims_idx))
            return tuple(coord_idx.astype(numpy.int))
        else:
            raise ValueError("coord is outside of the grid bbox")
