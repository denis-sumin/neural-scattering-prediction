import math
from abc import abstractmethod
from collections import Iterable, defaultdict
from copy import copy, deepcopy
from functools import partial
from itertools import starmap
from multiprocessing import Pool, cpu_count
from multiprocessing.dummy import Pool as ThreadPool
from random import seed, shuffle
from threading import Lock
from time import time
from typing import Dict, Optional, Tuple

import numpy
import pandas as pd
import sklearn

from fabnn.dataset_utils import (
    get_dataset_item_render_path,
    get_dataset_item_volume_path,
    resolve_dataset_item_file_path,
)
from fabnn.downscale import generate_scales_sat
from fabnn.grid_converter import FloatGridConverterBox
from fabnn.materials import get_correct_labels, populate_materials
from fabnn.sat_tile_tree import SATTileTree, SATTileTree2D
from fabnn.summed_area_table import build_volume_sat
from fabnn.utils import clean_timings_dict, human_size, log_timing, setup_console_logger

from ..collect_dataset_metadata import check_dataset_item_files_md5
from . import DataInterface

logger = setup_console_logger(__name__)


def get_scale_kernels_numpy(scale_levels):
    return numpy.array(
        [(level_z, level_xy, level_xy, 1) for level_xy, level_z in scale_levels],
        dtype=numpy.uint16,
    )


def get_step_memory_size(volume_files, step):
    volume_sizes = {idx: item[1]["memory_size"] for idx, item in enumerate(volume_files)}
    return sum((volume_sizes[volume_idx] for volume_idx in step))


def get_steps_from_memory_limit(volumes_files_idx_order, volume_files, memory_limit):
    volume_sizes = {idx: item[1]["memory_size"] for idx, item in enumerate(volume_files)}
    if sum(volume_sizes.values()) <= memory_limit:
        return [volumes_files_idx_order]

    steps = []
    step, idx_to_add = [], 0
    while True:
        while True:
            volume_idx_to_add = (volumes_files_idx_order * 2)[idx_to_add]
            size_to_add = volume_sizes[volume_idx_to_add]
            if size_to_add > memory_limit:
                raise ValueError(
                    "memory_size {} of volume with idx {} exceeds the memory limit {}".format(
                        size_to_add, idx_to_add, memory_limit
                    )
                )
            if get_step_memory_size(volume_files, step) + size_to_add <= memory_limit:
                step.append(volume_idx_to_add)
                idx_to_add += 1
            else:
                break

        new_step = copy(step)
        while get_step_memory_size(volume_files, new_step) + size_to_add > memory_limit:
            new_step.pop(0)

        if idx_to_add - len(volumes_files_idx_order) >= len(step):
            break
        else:
            steps.append(step)
            step = new_step
    return steps


def prepare_one_volume_patches(
    channels,
    estimate_patch_count_func,
    limit_ratio,
    shuffle_patches,
    random_seed,
    idx,
    key_metadata,
):
    if isinstance(key_metadata, tuple):
        _, metadata = key_metadata
    elif isinstance(key_metadata, numpy.ndarray):
        material_voxel_found_index = key_metadata
        metadata = None
    else:
        raise RuntimeError("Unexpected type of key_metadata argument:", type(key_metadata))

    if metadata is not None:
        patches_range = numpy.arange(estimate_patch_count_func(metadata))
    else:
        patches_range = numpy.argwhere(material_voxel_found_index)[:, 0]

    volume_patches = numpy.array(
        numpy.meshgrid(idx, numpy.arange(channels), patches_range), dtype=numpy.uint32
    ).T.reshape(-1, 3)
    num_patches = max(1, int(len(volume_patches) * limit_ratio))
    if shuffle_patches:
        volume_patches = sklearn.utils.shuffle(volume_patches, random_state=random_seed)
    volume_patches = volume_patches[:num_patches]
    return volume_patches, num_patches


class SimpleSAT:
    def __init__(
        self,
        volume,
        patch_size,
        scale_levels,
        stencil_channels,
        verbose_logging,
        timings=None,
        tile_size=None,
        alignment_z_centered=True,
        filled_vdb=None,
        label_grid=None,
    ):
        self.timings = timings if timings is not None else {}

        self.patch_size = patch_size
        self.scale_kernels_numpy = get_scale_kernels_numpy(scale_levels)
        self.stencil_channels = stencil_channels
        self.verbose_logging = verbose_logging

        s = time()
        self.volume_sat = build_volume_sat(volume)
        log_timing(self.timings, "build_volume_sat", time() - s)

        if self.verbose_logging:
            sat_size = self.get_size()
            logger.info("volume_sat size: {}".format(human_size(sat_size)))

    def get_size(self) -> int:
        size = (
            self.volume_sat.shape[0]
            * self.volume_sat.shape[1]
            * self.volume_sat.shape[2]
            * self.volume_sat.shape[3]
            * self.volume_sat.dtype.itemsize
        )
        return size

    def get_stencils(self, x, y, z=0):
        stencils = generate_scales_sat(
            self.scale_kernels_numpy, self.patch_size, self.volume_sat, y, x, True
        )
        return stencils

    def get_stencils_list(self, zyx_list):
        stencils = numpy.array(
            [
                generate_scales_sat(
                    self.scale_kernels_numpy, self.patch_size, self.volume_sat, y, x, True
                )
                for z, y, x in zyx_list
            ]
        )
        return stencils


class TreeSAT:
    def __init__(
        self,
        volume,
        patch_size,
        scale_levels,
        stencil_channels,
        verbose_logging,
        timings=None,
        tile_size=32,
        alignment_z_centered=True,
        filled_vdb=None,
        label_grid=None,
    ):
        self.timings = timings if timings is not None else {}

        self.patch_size = patch_size
        self.scale_levels_list = list(scale_levels)
        self.stencil_channels = stencil_channels
        self.verbose_logging = verbose_logging

        self.activate_mask = (
            isinstance(self.stencil_channels, Iterable) and "mask" in self.stencil_channels
        )
        self.activate_distance = (
            isinstance(self.stencil_channels, Iterable) and "distance" in self.stencil_channels
        )

        s = time()
        self.volume_sb_tree = SATTileTree2D(volume[:, :, :, :2], tile_size, alignment_z_centered)
        log_timing(self.timings, "Create SATTileTree2D", time() - s)

        if self.verbose_logging:
            logger.info("sb_tree size: {}".format(human_size(self.volume_sb_tree.size())))
        if self.activate_mask:
            s = time()
            self.volume_mask_tree = SATTileTree(
                volume[:, :, :, 2:3], tile_size, alignment_z_centered
            )
            log_timing(self.timings, "Create mask AverageKDTree", time() - s)
            if self.verbose_logging:
                logger.info("mask_tree size: {}".format(human_size(self.volume_mask_tree.size())))
        else:
            self.volume_mask_tree = None

        if self.activate_distance:
            converter = FloatGridConverterBox()
            s = time()
            distance_field = converter.resampleToNumpy(
                filled_vdb,
                "distanceField",
                label_grid.bbox_min,
                label_grid.bbox_max,
                label_grid.voxel_size,
            )
            print("distance_field min and max:", distance_field.min(), distance_field.max())
            log_timing(self.timings, "Resample distanceField", time() - s)

            s = time()
            self.distance_tree = SATTileTree(
                distance_field[..., numpy.newaxis], tile_size, alignment_z_centered
            )
            log_timing(self.timings, "Create distance SATTileTree", time() - s)
            if self.verbose_logging:
                logger.info(
                    "distance_tree size: {}".format(human_size(self.distance_tree.size()))
                )

    def get_size(self) -> int:
        size = self.volume_sb_tree.size()
        if self.activate_mask:
            size += self.volume_mask_tree.size()
        if self.activate_distance:
            size += self.distance_tree.size()
        return size

    def get_stencils(self, x, y, z=0):
        s = time()
        stencils = self.volume_sb_tree.generate_scales(
            self.scale_levels_list, self.patch_size, z, y, x
        )
        log_timing(self.timings, "volume_sb_tree.generate_scales", time() - s)

        additional_channels = []
        if self.activate_mask:
            s = time()
            additional_channels.append(
                self.volume_mask_tree.generate_scales(
                    self.scale_levels_list, self.patch_size, z, y, x
                )
            )
            log_timing(self.timings, "volume_mask_tree.generate_scales", time() - s)

        if self.activate_distance:
            s = time()
            additional_channels.append(
                self.distance_tree.generate_scales(
                    self.scale_levels_list, self.patch_size, z, y, x
                )
            )
            log_timing(self.timings, "distance_tree.generate_scales", time() - s)

        if len(additional_channels):
            s = time()
            stencils = numpy.concatenate([stencils, *additional_channels], axis=-1)
            log_timing(self.timings, "numpy.concatenate", time() - s)
        return stencils

    def get_stencils_list(self, zyx_list):
        s = time()
        stencils = self.volume_sb_tree.generate_scales_list(
            self.scale_levels_list, self.patch_size, zyx_list
        )
        log_timing(self.timings, "volume_sb_tree.generate_scales", time() - s)

        additional_channels = []
        if self.activate_mask:
            s = time()
            additional_channels.append(
                self.volume_mask_tree.generate_scales_list(
                    self.scale_levels_list, self.patch_size, zyx_list
                )
            )
            log_timing(self.timings, "volume_mask_tree.generate_scales", time() - s)
        if self.activate_distance:
            s = time()
            additional_channels.append(
                self.distance_tree.generate_scales_list(
                    self.scale_levels_list, self.patch_size, zyx_list
                )
            )
            log_timing(self.timings, "distance_tree.generate_scales", time() - s)

        if len(additional_channels):
            s = time()
            # numpy.stack is faster than numpy.concatenate([stencils, stencils_mask], axis=-1)
            arrays = []
            for channel in [
                stencils,
            ] + additional_channels:
                arrays += [channel[..., i] for i in range(channel.shape[-1])]
            stencils = numpy.stack(arrays, axis=-1)
            log_timing(self.timings, "numpy.concatenate", time() - s)
        return stencils


MODE_PREDICT = "MODE_PREDICT"
MODE_TRAIN = "MODE_TRAIN"


class Data(DataInterface):
    """
    Base implementation of the Data interface
    """

    def __init__(
        self,
        data_items,
        dataset_base_dir: str,
        materials_file,
        mode: str,
        patch_size,
        scale_levels,
        find_inner_material_voxels: bool,
        alignment_z_centered: bool,
        shuffle_patches: bool,
        stencil_channels,
        ignore_md5_checks: bool = False,
        limit_patches_number: Optional[int] = None,
        rotations: Optional[int] = None,
        sat_object_class_name: str = "SimpleSAT",
        sliding_window_length: Optional[int] = None,
        sliding_window_memory_limit: Optional[int] = None,
        stencils_only: bool = False,
        timings: Dict = None,
        verbose_logging: bool = False,
    ):
        super().__init__()
        self.mode = mode
        self.timings = timings if timings is not None else {}

        self.dataset_base_dir = dataset_base_dir
        self.data_items = data_items

        for dataset_item in self.data_items.values():
            dataset_item["volume_path"] = get_dataset_item_volume_path(
                dataset_base_dir, dataset_item
            )
            dataset_item["render_path"] = get_dataset_item_render_path(
                dataset_base_dir, dataset_item
            )
            dataset_item["proxy_object_filled_path"] = resolve_dataset_item_file_path(
                self.dataset_base_dir, dataset_item, "proxy_object_filled_path", required=False
            )

        self.limit_patches_number = limit_patches_number
        self.sliding_window_length = sliding_window_length
        self.sliding_window_memory_limit = sliding_window_memory_limit
        self.rotations = rotations if rotations is not None else 1

        self.volume_files = None
        self.s_b_sat_volumes = {}
        self.load_indexes = None
        self.patches = []
        self.labels = get_correct_labels(materials_file)

        if sat_object_class_name == "SimpleSAT":
            self.sat_object_class = SimpleSAT
        elif sat_object_class_name == "TreeSAT":
            self.sat_object_class = TreeSAT
        else:
            raise ValueError("`sat_object_class_name` should be one of: SimpleSAT, TreeSAT")

        self.alignment_z_centered = alignment_z_centered
        self.patch_size = patch_size
        self.scale_levels = scale_levels
        self.stencil_channels = stencil_channels
        self.find_inner_material_voxels = find_inner_material_voxels

        max_level_z = max((item[1] for item in scale_levels))
        self.thickness = max_level_z * patch_size[0]

        self.materials = populate_materials(materials_file)
        material_channels = [m.channels for m in self.materials.values()]
        assert max(material_channels) == min(
            material_channels
        ), "number of channels in materials file mismatch"  # all elements equal
        self.material_channels = material_channels[0]

        self.stencils_only = stencils_only
        self.ignore_md5_checks = ignore_md5_checks
        self.shuffle_patches = shuffle_patches

        self.verbose_logging = verbose_logging
        self.load_volumes_lock = Lock()

        s = time()
        self.preload_volumes()
        logger.info("Loading files took {} seconds".format(round(time() - s, 2)))
        s = time()
        self.generate_patches()
        logger.info(
            "Generated {} patches in {} seconds".format(len(self.patches), round(time() - s, 2))
        )

    def __iter__(self):
        for i in range(len(self)):
            yield self[i]

    def __len__(self):
        return len(self.patches)

    def get_and_clean_timings(self):
        timings = deepcopy(self.timings)
        clean_timings_dict(self.timings)
        return timings

    def log_timing(self, key, timing):
        log_timing(self.timings, key, timing)

    def load_volumes_set(self, volume_indexes):
        if set(volume_indexes) == set(self.s_b_sat_volumes.keys()):
            return

        s = time()
        with self.load_volumes_lock:
            self.log_timing("get_batch.1-acquire-volumes-lock", time() - s)
            s = time()
            for remove_datafile_index in set(self.s_b_sat_volumes.keys()).difference(
                volume_indexes
            ):
                if self.verbose_logging:
                    logger.info("Unloading {}".format(remove_datafile_index))
                vol = self.s_b_sat_volumes.pop(remove_datafile_index)
                del vol
            self.log_timing("load_volumes_set.2-unload-prev-step-volumes", time() - s)

            s = time()
            for datafile_idx in set(volume_indexes).difference(self.s_b_sat_volumes.keys()):
                ts = time()
                dataset_item_key, metadata = self.volume_files[datafile_idx]
                self.s_b_sat_volumes[datafile_idx] = self.load_volume_s_b_with_render(
                    idx=datafile_idx, dataset_item_key=dataset_item_key
                )
                if self.verbose_logging:
                    logger.info("Loading took {} seconds".format(time() - ts))
            self.log_timing("load_volumes_set.3-load-volumes", time() - s)

    @staticmethod
    @abstractmethod
    def estimate_patch_count(metadata):
        pass

    @abstractmethod
    def convert_patch_index(
        self, datafile_idx: int, channel: int, pidx: int
    ) -> Tuple[int, int, int]:
        """ convert between patch index and native X,Y[,Z] representation"""

    @abstractmethod
    def get_stencil_prediction(self, datafile_idx, channel, pidx):
        pass

    @abstractmethod
    def get_stencil_prediction_list(self, datafile_idx, channel, pidxs):
        pass

    @abstractmethod
    def load_volume_s_b_with_render(self, idx=None, dataset_item_key=None):
        pass

    @abstractmethod
    def get_render(self, dataset_item_key=None, data_object=None, **kwargs):
        pass

    def preload_volumes(self):
        s = time()
        self.volume_files = []
        for idx, (dataset_item_key, dataset_item) in enumerate(self.data_items.items()):
            volume_md5_matches, render_md5_matches = check_dataset_item_files_md5(
                self.dataset_base_dir, dataset_item
            )
            md5_match = volume_md5_matches and render_md5_matches
            if not self.ignore_md5_checks and not md5_match:
                raise ValueError(
                    "Dataset file metadata is outdated for item {}".format(dataset_item_key)
                )

            self.volume_files.append((dataset_item_key, dataset_item.get("metadata", {})))
        self.log_timing("preload_volumes", time() - s)

    def generate_patches(self, seed_: int = 0):
        seed(seed_)

        channels = self.material_channels

        s = time()
        patches_number = 0
        for idx, (key, metadata) in enumerate(self.volume_files):
            patches_number += self.estimate_patch_count(metadata) * channels
        if self.limit_patches_number is not None and patches_number > self.limit_patches_number:
            limit_ratio = self.limit_patches_number / patches_number
        else:
            limit_ratio = 1.0

        prepare_one_volume_patches_func = partial(
            prepare_one_volume_patches,
            channels,
            self.__class__.estimate_patch_count,
            limit_ratio,
            self.shuffle_patches,
            seed_,
        )

        if self.mode == MODE_TRAIN and self.find_inner_material_voxels:
            try:
                with Pool(min(cpu_count(), 16)) as pool:
                    res = pool.starmap(
                        prepare_one_volume_patches_func, self.material_voxel_found_index.items()
                    )
            except AssertionError:
                res = starmap(
                    prepare_one_volume_patches_func, self.material_voxel_found_index.items()
                )
        else:
            try:
                with Pool(min(cpu_count(), 16)) as pool:
                    res = pool.starmap(
                        prepare_one_volume_patches_func, enumerate(self.volume_files)
                    )
            except AssertionError:
                res = starmap(prepare_one_volume_patches_func, enumerate(self.volume_files))
        patches_per_volume = {
            idx: (volume_patches, num_patches)
            for idx, (volume_patches, num_patches) in zip(range(len(self.volume_files)), res)
        }
        self.log_timing("generate_patches.1-prepare-patches-pre-volume", time() - s)

        volumes_files_idx_order = list(range(len(self.volume_files))) * self.rotations
        if self.shuffle_patches:
            shuffle(volumes_files_idx_order)

        s = time()
        if self.sliding_window_length is None or self.sliding_window_length > len(
            self.volume_files
        ):
            self.sliding_window_length = len(self.volume_files)

        if self.sliding_window_length < len(volumes_files_idx_order):
            steps = []
            for idx_start in range(len(volumes_files_idx_order)):
                step = tuple(
                    (volumes_files_idx_order * 2)[
                        idx_start : idx_start + self.sliding_window_length
                    ]
                )
                if self.sliding_window_memory_limit:
                    if (
                        get_step_memory_size(self.volume_files, step)
                        > self.sliding_window_memory_limit
                    ):
                        raise ValueError(
                            "A step of a fixed size {} exceeds memory limit {}".format(
                                self.sliding_window_length,
                                human_size(self.sliding_window_memory_limit),
                            )
                        )
                steps.append(step)
        else:
            if self.sliding_window_memory_limit:
                steps = get_steps_from_memory_limit(
                    volumes_files_idx_order=volumes_files_idx_order,
                    volume_files=self.volume_files,
                    memory_limit=self.sliding_window_memory_limit,
                )
            else:
                steps = [tuple(volumes_files_idx_order)]
        if self.verbose_logging:
            logger.info(
                "Sliding window step sizes are: {} volumes".format([len(step) for step in steps])
            )

        volume_idx_counters = defaultdict(int)
        for step in steps:
            for idx in step:
                volume_idx_counters[idx] += 1
        self.log_timing("generate_patches.2-prepare-steps", time() - s)

        self.load_indexes = {}

        s = time()
        step_start_idx = 0
        patches_per_step = []
        for step in steps:
            step_patches = []
            for volume_idx in step:
                volume_patches, num_patches = patches_per_volume[volume_idx]
                split_idx = math.ceil(num_patches / volume_idx_counters[volume_idx])
                add, remain = volume_patches[:split_idx], volume_patches[split_idx:]
                step_patches.append(add)
                patches_per_volume[volume_idx] = remain, num_patches
            step_patches = numpy.concatenate(step_patches)
            self.load_indexes[(step_start_idx, step_start_idx + len(step_patches) - 1)] = step
            step_start_idx += len(step_patches)
            patches_per_step.append(step_patches)

        if self.shuffle_patches:
            with ThreadPool() as pool:
                patches_per_step = pool.map(
                    partial(sklearn.utils.shuffle, random_state=seed_), patches_per_step
                )

        if self.verbose_logging:
            for step_idx, step_patches in enumerate(patches_per_step):
                logger.debug("Step volumes: {}".format(steps[step_idx]))
                logger.debug("Step first 100 patches: {}".format(step_patches[:100]))

        self.log_timing("generate_patches.3-create-patches-sequence", time() - s)

        s = time()
        self.patches = pd.DataFrame(
            numpy.concatenate(patches_per_step),
            columns=("datafile_idx", "channel_idx", "patch_idx"),
        ).astype(
            {
                "datafile_idx": "uint16",
                "channel_idx": "uint8",
                "patch_idx": "uint32",
            }
        )
        self.log_timing("generate_patches.4-store-sequence-in-pd.Dataframe", time() - s)
        logger.info(
            "Patches dataframe uses {}".format(human_size(self.patches.memory_usage().sum()))
        )

    def __getitem__(self, idx):
        datafile_idx = self.patches.at[idx, "datafile_idx"]
        c = self.patches.at[idx, "channel_idx"]
        pidx = self.patches.at[idx, "patch_idx"]

        patch = datafile_idx, c, pidx

        s = time()
        with self.load_volumes_lock:
            self.log_timing("__getitem__.1-acquire-volumes-lock", time() - s)
            if datafile_idx not in self.s_b_sat_volumes:
                s = time()
                for index_first, index_last in sorted(self.load_indexes.keys()):
                    if index_first <= idx <= index_last:
                        step_volumes = self.load_indexes[(index_first, index_last)]
                        for remove_datafile_index in set(self.s_b_sat_volumes.keys()).difference(
                            step_volumes
                        ):
                            if self.verbose_logging:
                                logger.info("Unloading {}".format(remove_datafile_index))
                            vol = self.s_b_sat_volumes.pop(remove_datafile_index)
                            del vol
                        break
                self.log_timing("__getitem__.2-unload-prev-step-volumes", time() - s)
                dataset_item_key, metadata = self.volume_files[datafile_idx]
                s = time()
                self.s_b_sat_volumes[datafile_idx] = self.load_volume_s_b_with_render(
                    idx=datafile_idx, dataset_item_key=dataset_item_key
                )
                if self.verbose_logging:
                    logger.info("Loading took {} seconds".format(time() - s))
                self.log_timing("__getitem__.3-load-volume", time() - s)

        s = time()
        stencils, prediction = self.get_stencil_prediction(*patch)
        self.log_timing("__getitem__.4-create-stencils", time() - s)

        return patch, stencils, prediction

    def get_batch(self, start, size, restore_permutation=True, output_locations=False):
        size = min(len(self), start + size) - start
        stencils = []
        predictions = []
        locations = [] if output_locations else None
        permutation = []

        batch_first = start
        batch_size = size
        batch_last = batch_first + batch_size - 1

        while True:
            step_found = False
            for step_start, step_end in sorted(self.load_indexes.keys()):
                if step_start <= start <= step_end:
                    step_found = True
                    break

            if not step_found:
                print("Data.load_indexes.keys()", list(self.load_indexes.keys()))
                raise ValueError(
                    "Requested batch range ({}, {}) is outside of the data range "
                    "(0, {}).".format(start, start + size, len(self.patches))
                )

            end = min(batch_last, step_end)
            step_volumes = self.load_indexes[(step_start, step_end)]

            self.load_volumes_set(step_volumes)

            s = time()
            batch_patches = self.patches.iloc[start : end + 1]
            for (datafile_idx, channel), pidx_df in batch_patches.groupby(
                ["datafile_idx", "channel_idx"]
            ):

                patches = pidx_df["patch_idx"].values
                stencils_this, predictions_this = self.get_stencil_prediction_list(
                    datafile_idx, channel, patches
                )

                stencils.append(stencils_this)
                if output_locations:
                    locations.extend([(datafile_idx, channel, p) for p in patches])
                if not self.stencils_only:
                    predictions.extend(predictions_this)
                if restore_permutation:
                    permutation.extend(pidx_df.index.values + (start - batch_first))
            self.log_timing("get_batch.get_stencil_prediction_list", time() - s)

            if end < batch_last:
                start = end + 1
            else:
                break

        if output_locations:
            locations = numpy.array(locations, dtype=numpy.uint32)
        stencils = numpy.concatenate(stencils, axis=0)

        if self.stencils_only:
            predictions = [None for _ in range(len(stencils))]
        predictions = numpy.array(predictions)

        if restore_permutation:
            permutation = numpy.argsort(permutation)
            if output_locations:
                locations = locations[permutation]
            predictions = predictions[permutation]
            stencils = stencils[permutation]

        return stencils, predictions, locations

    @staticmethod
    def map_discrete_voxels(
        discrete_volume, volume, labels, materials, channel, stencil_channels
    ):
        for label, name in labels.items():
            a = materials[label].albedo[channel]
            d = materials[label].density[channel]
            s = a * d
            b = d - s
            if stencil_channels == 2:
                s_b = numpy.array((s, b), dtype=volume.dtype)
            elif stencil_channels == 3:
                mask_value = float(not (name == "air"))
                s_b = numpy.array((s, b, mask_value), dtype=volume.dtype)
            else:
                raise ValueError("stencil_channels should be 2 or 3 in Data")
            volume[discrete_volume == label] = s_b


def make_batch(
    data: Data, start: int, size: int, build_size: int, output_locations: bool = False
):
    stencils, predictions, locations = data.get_batch(
        start, build_size, output_locations=output_locations
    )
    if output_locations:
        batches = [
            (
                stencils[batch_start : batch_start + size],
                predictions[batch_start : batch_start + size],
                locations[batch_start : batch_start + size],
            )
            for batch_start in range(0, build_size, size)
        ]
    else:
        batches = [
            (
                stencils[batch_start : batch_start + size],
                predictions[batch_start : batch_start + size],
            )
            for batch_start in range(0, build_size, size)
        ]
    return batches


def one_batch_stencils_swap_axes(stencils):
    size, scale_levels = stencils.shape[:2]
    stencils = stencils.reshape((size, scale_levels, -1))
    reordered_stencils = numpy.moveaxis(stencils, 1, 0)
    return [numpy.squeeze(a) for a in numpy.vsplit(reordered_stencils, scale_levels)]


def make_batch_swap_axes(
    data: Data, start: int, size: int, build_size: int, output_locations: bool = False
):
    stencils, predictions, locations = data.get_batch(
        start, build_size, output_locations=output_locations
    )

    reordered_stencils_split_into_batches = (
        one_batch_stencils_swap_axes(stencils[batch_start : batch_start + size])
        for batch_start in range(0, min(build_size, len(stencils)), size)
    )
    predictions_split_into_batches = (
        predictions[batch_start : batch_start + size]
        for batch_start in range(0, build_size, size)
    )

    if output_locations:
        locations_split_into_batches = (
            locations[batch_start : batch_start + size]
            for batch_start in range(0, build_size, size)
        )
        return [
            item
            for item in zip(
                reordered_stencils_split_into_batches,
                predictions_split_into_batches,
                locations_split_into_batches,
            )
        ]
    else:
        return [
            item
            for item in zip(reordered_stencils_split_into_batches, predictions_split_into_batches)
        ]
