import os
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from multiprocessing.dummy import Pool as ThreadPool
from time import time
from typing import Tuple

import numpy
import pyopenvdb

from fabnn.label_grid import LabelGrid
from fabnn.utils import setup_console_logger

from .data import MODE_TRAIN, Data

logger = setup_console_logger(__name__)


def find_inner_voxel(
    coord_world, label_grid: LabelGrid, normal, max_distance_mm: float = 3 * 0.084
) -> Tuple[Tuple[int, int, int], bool]:
    """
    :param coord_world: coordinate X-Y-Z
    :param label_grid:
    :param normal:
    :param max_distance_mm: default: 3 300 dpi voxels
    :return: coordinate X-Y-Z
    """
    step = min(label_grid.voxel_size)
    n = numpy.array(normal, dtype=numpy.float64)
    n *= step / numpy.linalg.norm(n)

    coord_label_initial = label_grid.world_to_index(coord_world)
    coord_world_ = numpy.array(coord_world, dtype=numpy.float64)

    distance = 0
    while True:
        try:
            coord_label = label_grid.world_to_index(coord_world_)
        except ValueError:
            return coord_label_initial, False
        try:
            label_grid_label = label_grid.data[coord_label[2], coord_label[1], coord_label[0]]
        except IndexError:
            return coord_label_initial, False
        if label_grid_label in (5, 6):  # air, transparent
            coord_world_ -= n
            distance += step
        else:
            return coord_label, True
        if distance > max_distance_mm:
            return coord_label_initial, False


def calculate_discrete_volume_index(
    label_grid: LabelGrid,
    world_coord_index,
    normal_index,
    find_inner_material_voxels: bool,
    idx: int,
):
    coord_world = world_coord_index[idx]

    if find_inner_material_voxels:
        normal = normal_index[idx]
        coord_labels_index, found = find_inner_voxel(coord_world, label_grid, normal)
    else:
        coord_labels_index = label_grid.world_to_index(coord_world)
        found = True
    return coord_labels_index, found


class Data3D(Data):
    """
    Data implementation for planar 2.5D slabs
    """

    def __init__(self, *args, **kwargs):
        self.discrete_voxels_index = {}
        self.material_voxel_found_index = {}
        self.render_index = {}

        self.voxel_size = (25.4 / 600.0, 25.4 / 300.0, 0.027)
        super().__init__(*args, **kwargs)

        if not self.alignment_z_centered:
            raise ValueError("Data3D accepts only alignment_z_centered=True")

    def generate_patches(self, seed_: int = 0):
        if self.find_inner_material_voxels:
            for datafile_idx, (dataset_item_key, _) in enumerate(self.volume_files):
                if (
                    datafile_idx in self.discrete_voxels_index
                    and datafile_idx in self.material_voxel_found_index
                    and datafile_idx in self.render_index
                ):
                    continue

                dataset_item = self.data_items[dataset_item_key]
                volume_path = dataset_item["volume_path"]
                label_grid = LabelGrid(volume_path)
                (
                    render,
                    render_index,
                    discrete_voxels_index,
                    material_voxel_found_index,
                    distances,
                ) = self.get_render(
                    dataset_item_key=dataset_item_key, data_object=label_grid, make_index=True
                )
                self.discrete_voxels_index[datafile_idx] = discrete_voxels_index
                if self.mode == MODE_TRAIN:
                    self.material_voxel_found_index[datafile_idx] = material_voxel_found_index & (
                        distances < 0.2
                    )
                else:
                    self.material_voxel_found_index[datafile_idx] = material_voxel_found_index
                self.render_index[datafile_idx] = render_index

        super().generate_patches(seed_)

    def load_volume_s_b_with_render(self, idx=None, dataset_item_key=None):
        dataset_item = self.data_items[dataset_item_key]
        filename = dataset_item["filename"]

        volume_path = dataset_item["volume_path"]

        if self.verbose_logging:
            logger.info("Loading volume {}".format(filename))

        label_grid = LabelGrid(volume_path)
        discrete_volume = label_grid.data

        if not numpy.allclose(self.voxel_size, label_grid.voxel_size, atol=1e-03):
            raise RuntimeError(
                "Label grid voxel_size {} is different from the data object voxel_size {}".format(
                    tuple(label_grid.voxel_size), tuple(self.voxel_size)
                )
            )

        make_index = idx not in self.discrete_voxels_index or idx not in self.render_index
        if not self.stencils_only or make_index:
            (
                render,
                render_index,
                discrete_voxels_index,
                material_voxel_found_index,
                distances,
            ) = self.get_render(
                dataset_item_key=dataset_item_key, data_object=label_grid, make_index=make_index
            )
            if make_index:
                self.discrete_voxels_index[idx] = discrete_voxels_index
                self.material_voxel_found_index[idx] = material_voxel_found_index
                self.render_index[idx] = render_index
        else:
            render = None

        volumes_s_b_renders = []
        volume = numpy.empty(
            shape=discrete_volume.shape + (len(self.stencil_channels),), dtype=numpy.float32
        )

        if self.sat_object_class.__name__ not in self.timings:
            self.timings[self.sat_object_class.__name__] = {}

        for channel in range(self.material_channels):
            with ThreadPool() as pool:
                pool.starmap(
                    Data.map_discrete_voxels,
                    zip(
                        discrete_volume,
                        volume,
                        repeat(self.labels),
                        repeat(self.materials),
                        repeat(channel),
                        repeat(len(self.stencil_channels)),
                    ),
                )

            distance_field_vdb = dataset_item.get("proxy_object_filled_path", "")
            if distance_field_vdb.endswith("_filled.vdb"):
                distance_field_vdb = distance_field_vdb.replace("_filled", "_filled_training")
            sat_object = self.sat_object_class(
                volume=volume,
                patch_size=self.patch_size,
                scale_levels=self.scale_levels,
                stencil_channels=self.stencil_channels,
                verbose_logging=self.verbose_logging,
                tile_size=dataset_item["metadata"]["tile_size"],
                alignment_z_centered=True,
                timings=self.timings[self.sat_object_class.__name__],
                filled_vdb=distance_field_vdb,
                label_grid=label_grid,
            )

            volumes_s_b_renders.append((sat_object, render))
        del volume

        return volumes_s_b_renders

    def get_render(self, dataset_item_key=None, data_object=None, make_index=False):
        dataset_item = self.data_items[dataset_item_key]
        filled_path = dataset_item["proxy_object_filled_path"]
        render_path = dataset_item["render_path"]

        if filled_path is None and render_path is None:
            raise ValueError(
                "One of: `proxy_object_filled_path`, `render_filename` "
                "is needed for Data3D but neither is given"
            )

        index_cache_path_suffix = (
            "_index_cache_inner_voxel.npz"
            if self.find_inner_material_voxels
            else "_index_cache.npz"
        )
        index_cache_path = None
        if filled_path:
            index_cache_path = filled_path.replace("_filled.vdb", index_cache_path_suffix)
        if render_path:
            substring_index = render_path.find("_render_result")
            if substring_index > 0:
                index_cache_path_old = render_path[:substring_index] + index_cache_path_suffix
            else:
                raise RuntimeError("Render path does not containt substring render_result")
            # if filled_path:
            #     if os.path.exists(index_cache_path_old) and not os.path.exists(index_cache_path):
            #         os.rename(index_cache_path_old, index_cache_path)
            # else:
            index_cache_path = index_cache_path_old

        label_grid = data_object
        rendering_grid = pyopenvdb.read(render_path, "rendering") if render_path else None
        # normal_grid = pyopenvdb.read(
        #     filled_path if filled_path is not None else render_path, 'normalGrid')
        normal_grid = pyopenvdb.read(
            render_path if render_path is not None else filled_path, "normalGrid"
        )
        normal_grid_acc = normal_grid.getConstAccessor()

        if make_index:
            if index_cache_path and os.path.exists(index_cache_path):
                d = numpy.load(index_cache_path, allow_pickle=True)
                discrete_voxels_index = d["discrete_voxels_index"]
                material_voxel_found_index = d["material_voxel_found_index"]
                render_index = d["render_index"]
                try:
                    distances = d["distances"]
                except KeyError:  # temporary backward compatibility code
                    s = time()
                    with ThreadPool() as pool:
                        render_index_tuples = [
                            (int(r[0]), int(r[1]), int(r[2])) for r in render_index
                        ]
                        render_voxels_world = numpy.array(
                            pool.map(normal_grid.transform.indexToWorld, render_index_tuples)
                        )
                    discrete_voxels_world = (
                        label_grid.bbox_min + discrete_voxels_index * label_grid.voxel_size
                    )
                    distances = numpy.linalg.norm(
                        discrete_voxels_world - render_voxels_world, axis=1
                    )
                    logger.info(
                        "Calculating distances for {} took {} seconds".format(
                            dataset_item_key, time() - s
                        )
                    )
                    numpy.savez_compressed(
                        index_cache_path,
                        discrete_voxels_index=discrete_voxels_index,
                        material_voxel_found_index=material_voxel_found_index,
                        render_index=render_index,
                        distances=distances,
                    )
            else:
                logger.info("Building indexes for {}".format(dataset_item_key))

                if "find_inner_voxel" not in self.timings:
                    self.timings["find_inner_voxel"] = 0.0

                discrete_voxels_index = numpy.empty(
                    shape=(dataset_item["metadata"]["num_surface_voxels"], 3), dtype=numpy.int32
                )
                render_index = numpy.empty(
                    shape=(dataset_item["metadata"]["num_surface_voxels"], 3), dtype=numpy.int32
                )
                material_voxel_found_index = numpy.empty(
                    shape=(dataset_item["metadata"]["num_surface_voxels"],), dtype=numpy.bool
                )
                world_coord_index = numpy.empty(
                    shape=(dataset_item["metadata"]["num_surface_voxels"], 3), dtype=numpy.float64
                )
                normal_index = numpy.empty(
                    shape=(dataset_item["metadata"]["num_surface_voxels"], 3), dtype=numpy.float64
                )
                index_counter = 0
                if rendering_grid:
                    for item in rendering_grid.citerOnValues():
                        if item.count == 1:  # voxel value
                            grid_coord = item.min
                            render_index[index_counter] = grid_coord
                            world_coord_index[
                                index_counter
                            ] = rendering_grid.transform.indexToWorld(grid_coord)
                            normal_index[index_counter], active = normal_grid_acc.probeValue(
                                grid_coord
                            )
                            assert active
                            index_counter += 1
                else:
                    for item in normal_grid.citerOnValues():
                        if item.count == 1:  # voxel value
                            grid_coord = item.min
                            render_index[index_counter] = grid_coord
                            world_coord_index[index_counter] = normal_grid.transform.indexToWorld(
                                grid_coord
                            )
                            normal_index[index_counter] = item.value
                            index_counter += 1
                if label_grid is not None:
                    calculate_discrete_volume_index_func = partial(
                        calculate_discrete_volume_index,
                        label_grid,
                        world_coord_index,
                        normal_index,
                        self.find_inner_material_voxels,
                    )
                    ts = time()
                    with Pool() as pool:
                        indexes = pool.map(
                            calculate_discrete_volume_index_func, range(len(render_index))
                        )
                    for idx, (coord, found) in enumerate(indexes):
                        discrete_voxels_index[idx] = coord
                        material_voxel_found_index[idx] = found
                    find_inner_voxel_timing = time() - ts
                    if self.verbose_logging:
                        logger.info(
                            "find_inner_voxel timing: {} seconds".format(
                                round(find_inner_voxel_timing, 3)
                            )
                        )
                    self.timings["find_inner_voxel"] += find_inner_voxel_timing

                    discrete_voxels_world = (
                        label_grid.bbox_min + discrete_voxels_index * label_grid.voxel_size
                    )
                    distances = numpy.linalg.norm(
                        discrete_voxels_world - world_coord_index, axis=1
                    )
                else:
                    distances = None

                numpy.savez_compressed(
                    index_cache_path,
                    discrete_voxels_index=discrete_voxels_index,
                    material_voxel_found_index=material_voxel_found_index,
                    render_index=render_index,
                    distances=distances,
                )
        else:
            discrete_voxels_index = None
            material_voxel_found_index = None
            render_index = None
            distances = None

        if label_grid is None:
            del discrete_voxels_index
            discrete_voxels_index = None

        return (
            rendering_grid,
            render_index,
            discrete_voxels_index,
            material_voxel_found_index,
            distances,
        )

    @staticmethod
    def estimate_patch_count(metadata):
        return metadata["num_surface_voxels"]

    def convert_patch_index(self, datafile_idx, channel, pidx):
        _, metadata = self.volume_files[datafile_idx]

        assert pidx < Data3D.estimate_patch_count(metadata), "patch index within range"
        x, y, z = self.discrete_voxels_index[datafile_idx][pidx, :]
        return int(x), int(y), int(z)

    def convert_patch_index_to_render_coords(self, datafile_idx, channel, pidx):
        _, metadata = self.volume_files[datafile_idx]

        assert pidx < Data3D.estimate_patch_count(metadata), "patch index within range"
        x, y, z = self.render_index[datafile_idx][pidx, :]
        return int(x), int(y), int(z)

    def get_stencil_prediction(self, datafile_idx, channel, pidx):
        volumes_s_b_renders = self.s_b_sat_volumes[datafile_idx]
        sat_object, render = volumes_s_b_renders[channel]

        x, y, z = self.convert_patch_index(datafile_idx, channel, pidx)
        stencils = sat_object.get_stencils(x=x, y=y, z=z)

        x, y, z = self.convert_patch_index_to_render_coords(datafile_idx, channel, pidx)
        prediction = (
            render.getConstAccessor().getValue((int(x), int(y), int(z)))[channel]
            if not self.stencils_only
            else None
        )
        return stencils, prediction

    def get_stencil_prediction_list(self, datafile_idx, channel, pidxs):
        volumes_s_b_renders = self.s_b_sat_volumes[datafile_idx]
        sat_object, render = volumes_s_b_renders[channel]

        xyz_list = [self.convert_patch_index(datafile_idx, channel, pidx) for pidx in pidxs]
        zyx_list = [(z, y, x) for x, y, z in xyz_list]
        stencils = sat_object.get_stencils_list(zyx_list)

        if not self.stencils_only:
            xyz_list_render = [
                self.convert_patch_index_to_render_coords(datafile_idx, channel, pidx)
                for pidx in pidxs
            ]
            predictions = [
                render.getConstAccessor().getValue((int(x), int(y), int(z)))[channel]
                for x, y, z in xyz_list_render
            ]
        else:
            predictions = None

        return stencils, predictions
