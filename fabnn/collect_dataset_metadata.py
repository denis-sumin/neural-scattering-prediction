import argparse
import os
from copy import copy
from functools import partial
from multiprocessing import Pool, cpu_count
from typing import Callable, Dict

import numpy
import pyopenvdb
import yaml

from fabnn.dataset_utils import (
    get_dataset_item_render_path,
    get_dataset_item_volume_path,
    is_2D_file,
    is_3D_file,
)
from fabnn.materials import Material, get_correct_labels
from fabnn.utils import md5sum_path, read_image, resolve_project_path, setup_console_logger

logger = setup_console_logger(__name__)


def estimate_memory_size_one(
    dataset_base_dir, key, data_item_dict, verbose_logging, test_size
) -> int:
    data_item_dict = copy(data_item_dict)
    data_item_dict["metadata"]["tile_size"] = test_size

    # prevent circular import with hacky dynamic import
    from nn_rendering_prediction.data_storage import Data3D, DataPlanar

    dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, data_item_dict)
    if is_2D_file(dataset_volume_file_path):
        data_class = DataPlanar
    elif is_3D_file(dataset_volume_file_path):
        data_class = Data3D
    else:
        raise ValueError(
            "{} is neither 2.5D pipeline file nor 3D".format(dataset_volume_file_path)
        )

    train_data = data_class(
        alignment_z_centered=True,
        data_items={key: data_item_dict},
        dataset_base_dir=dataset_base_dir,
        find_inner_material_voxels=False,
        ignore_md5_checks=True,
        limit_patches_number=1,
        materials_file=resolve_project_path(
            "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
        ),
        mode="MODE_PREDICT",
        patch_size=[3, 5, 5],
        sat_object_class_name="TreeSAT",
        scale_levels=[
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
        ],
        shuffle_patches=False,
        stencil_channels=["scattering", "absorption", "mask"],
        stencils_only=True,
        verbose_logging=verbose_logging,
    )
    _ = train_data[0]  # trigger loading of volumes
    volumes_s_b_renders = train_data.s_b_sat_volumes[0]
    size = 0
    for channel in range(3):
        sat_object, render = volumes_s_b_renders[channel]
        size += sat_object.get_size()
    del train_data
    return size


def estimate_memory_sizes(
    dataset_base_dir, key, data_item_dict, allow_multiprocessing, verbose_logging
) -> Dict[int, int]:
    test_sizes = [8, 16, 24, 32, 40, 48, 56, 64]
    limit_omp = 4
    os.environ["OMP_NUM_THREADS"] = str(limit_omp)

    try:
        h, w = data_item_dict["metadata"]["height"], data_item_dict["metadata"]["width"]
        multiprocessing_ = h * w < 1000 * 1000
    except KeyError:
        multiprocessing_ = False
    multiprocessing_ = multiprocessing_ and allow_multiprocessing
    if multiprocessing_:
        with Pool(max(1, cpu_count() // limit_omp)) as pool:
            test_sizes_results = pool.map(
                partial(
                    estimate_memory_size_one,
                    dataset_base_dir,
                    key,
                    data_item_dict,
                    verbose_logging,
                ),
                test_sizes,
            )
    else:
        test_sizes_results = map(
            partial(
                estimate_memory_size_one, dataset_base_dir, key, data_item_dict, verbose_logging
            ),
            test_sizes,
        )

    test_results = {tile_size: size for tile_size, size in zip(test_sizes, test_sizes_results)}
    os.environ["OMP_NUM_THREADS"] = str(cpu_count())
    return test_results


def check_dataset_item_labels(dataset_base_dir, dataset_item, materials_file):
    correct_labels = get_correct_labels(materials_file)

    dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
    if is_2D_file(dataset_volume_file_path):
        d = numpy.load(dataset_volume_file_path, allow_pickle=True)
        labels = d["labels"].item()
    elif is_3D_file(dataset_volume_file_path):
        return  # no labels embedded in 3D files
    else:
        raise ValueError(
            "{} is neither 2.5D pipeline file nor 3D".format(dataset_volume_file_path)
        )

    for key, value in dict(labels).items():
        if isinstance(value, Material):
            if value.name != correct_labels[key]:
                raise ValueError(
                    "Check of the labels failed: {} {}".format(correct_labels[key], value.name)
                )
        else:
            if value != correct_labels[key]:
                raise ValueError(
                    "Check of the labels failed: {} {}".format(correct_labels[key], value)
                )


def check_dataset_item_files_md5(dataset_base_dir, dataset_item, return_sums=False):
    dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
    saved_volume_md5 = dataset_item.get("metadata", {}).get("processed_volume_md5")
    dataset_volume_file_md5 = md5sum_path(dataset_volume_file_path)
    volume_md5_matches = dataset_volume_file_md5 == saved_volume_md5

    dataset_render_file_path = get_dataset_item_render_path(dataset_base_dir, dataset_item)
    saved_render_md5 = dataset_item.get("metadata", {}).get("processed_render_md5")
    if dataset_render_file_path:
        dataset_render_file_md5 = md5sum_path(dataset_render_file_path)
    else:
        dataset_render_file_md5 = None
    render_md5_matches = dataset_render_file_md5 == saved_render_md5

    if return_sums:
        return (
            volume_md5_matches,
            render_md5_matches,
            dataset_volume_file_md5,
            dataset_render_file_md5,
        )
    else:
        return volume_md5_matches, render_md5_matches


def get_dataset_item_dimensions_2D(dataset_base_dir, dataset_item, test_only=False):
    dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
    d = numpy.load(dataset_volume_file_path, allow_pickle=True)
    discrete_volume = d["halftoned_voxels"]
    if test_only:
        dim_y, dim_x = discrete_volume.shape[1:3]
        pad_y, pad_x = 0, 0
    else:
        dataset_render_file_path = get_dataset_item_render_path(dataset_base_dir, dataset_item)
        if dataset_render_file_path.endswith("npz"):
            render = d["rendered_image"]
        else:
            render = read_image(dataset_render_file_path)
        dim_y, dim_x = render.shape[:2]
        assert (discrete_volume.shape[1] - dim_y) % 2 == 0
        assert (discrete_volume.shape[2] - dim_x) % 2 == 0

        pad_y = (discrete_volume.shape[1] - dim_y) // 2
        pad_x = (discrete_volume.shape[2] - dim_x) // 2

    return {
        "width": dim_x,
        "height": dim_y,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def get_dataset_item_dimensions_3D(dataset_base_dir, dataset_item, test_only=False):
    dataset_render_file_path = get_dataset_item_render_path(dataset_base_dir, dataset_item)
    render_grid = pyopenvdb.read(dataset_render_file_path, "rendering")
    num_surface_voxels = render_grid.activeVoxelCount()
    return {
        "num_surface_voxels": num_surface_voxels,
    }


def process_items_list(
    dataset_base_dir,
    items,
    save_dataset: Callable,
    test_only=False,
    recalculate_metadata=False,
    allow_multiprocessing=False,
    verbose_logging=False,
    materials_file=None,
):
    for key, dataset_item in items.items():
        logger.info("Processing {}".format(key))
        dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
        dataset_render_file_path = get_dataset_item_render_path(dataset_base_dir, dataset_item)

        if is_2D_file(dataset_volume_file_path):
            get_dimensions = get_dataset_item_dimensions_2D
        elif is_3D_file(dataset_volume_file_path):
            get_dimensions = get_dataset_item_dimensions_3D
        else:
            raise ValueError(
                "{} is neither 2.5D pipeline file nor 3D".format(dataset_volume_file_path)
            )

        (
            volume_md5_matches,
            render_md5_matches,
            dataset_volume_file_md5,
            dataset_render_file_md5,
        ) = check_dataset_item_files_md5(dataset_base_dir, dataset_item, return_sums=True)

        if not dataset_render_file_path and not test_only:
            raise ValueError("No rendering found for dataset item with key {}".format(key))

        metadata_dict = dataset_item.get("metadata", {})
        if (
            not recalculate_metadata
            and "memory_size" in metadata_dict
            and "tile_size" in metadata_dict
            and volume_md5_matches
        ):
            if not render_md5_matches:
                dim_metadata = get_dimensions(dataset_base_dir, dataset_item, test_only=test_only)
                metadata_dict.update(dim_metadata)
                metadata_dict.update(
                    {
                        "processed_render_md5": dataset_render_file_md5,
                    }
                )
                save_dataset()
            if metadata_dict.get("copied_from_source", False):
                save_dataset()
        else:
            dim_metadata = get_dimensions(dataset_base_dir, dataset_item, test_only=test_only)
            check_dataset_item_labels(dataset_base_dir, dataset_item, materials_file)

            metadata_dict = {
                "processed_volume_md5": dataset_volume_file_md5,
                "processed_render_md5": dataset_render_file_md5,
            }
            metadata_dict.update(dim_metadata)
            dataset_item["metadata"] = metadata_dict

            memory_sizes = estimate_memory_sizes(
                dataset_base_dir, key, dataset_item, allow_multiprocessing, verbose_logging
            )
            best_combination = sorted(memory_sizes.items(), key=lambda x: x[1])[0]
            tile_size, memory_size = best_combination

            metadata_dict.update(
                {
                    "memory_sizes": memory_sizes,
                    "tile_size": tile_size,
                    "memory_size": memory_size,
                }
            )

            dataset_item["metadata"] = metadata_dict
            save_dataset()


def merge_dataset_items(source_items, destination_items):
    for key, dataset_item in destination_items.items():
        if key not in source_items:
            continue
        source_dataset_item = source_items[key]
        items_match = True
        for field_name in ("filename", "render_filename"):
            if source_dataset_item[field_name] != dataset_item[field_name]:
                items_match = False
                break
        if items_match and "metadata" in source_dataset_item:
            dataset_item["metadata"] = source_dataset_item["metadata"]
            dataset_item["metadata"]["copied_from_source"] = True


def get_parser():
    parser = argparse.ArgumentParser(
        description="Estimate optimal parameters for a SAT Tile Tree over a dataset yml"
    )
    parser.add_argument(
        "-d",
        "--dataset",
        dest="dataset",
        type=str,
        required=True,
        help="dataset yml file to update",
    )
    parser.add_argument(
        "-s",
        "--source-dataset",
        dest="source_dataset",
        type=str,
        default=None,
        help="dataset yml to copy metadata from",
    )
    parser.add_argument(
        "-r", "--recalculate", help="Recalculate image if it exists", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )
    parser.add_argument(
        "-m",
        "--allow-multiprocessing",
        help="Allow multiprocessing in estimate_memory_sizes",
        action="store_true",
    )
    return parser


def main():
    parser = get_parser()
    args = parser.parse_args()

    dataset_base_dir = os.path.abspath(os.path.split(args.dataset)[0])
    with open(args.dataset, "r") as f:
        dataset = yaml.full_load(f)

    train_data_items = dataset["items"]
    validate_data_items = dataset.get("items_validate", {})
    materials_file = dataset["materials_file"]

    if args.source_dataset:
        with open(args.source_dataset, "r") as f:
            source_dataset = yaml.full_load(f)
        merge_dataset_items(source_dataset.get("items", {}), train_data_items)
        merge_dataset_items(source_dataset.get("items_validate", {}), validate_data_items)

    def save_dataset():
        with open(args.dataset, "w") as f:
            f.write(yaml.safe_dump(dataset))

    process_items_list(
        dataset_base_dir=dataset_base_dir,
        items=train_data_items,
        save_dataset=save_dataset,
        recalculate_metadata=args.recalculate,
        allow_multiprocessing=args.allow_multiprocessing,
        verbose_logging=args.verbose,
        materials_file=materials_file,
    )

    process_items_list(
        dataset_base_dir=dataset_base_dir,
        items=validate_data_items,
        save_dataset=save_dataset,
        recalculate_metadata=args.recalculate,
        allow_multiprocessing=args.allow_multiprocessing,
        verbose_logging=args.verbose,
        materials_file=materials_file,
    )


if __name__ == "__main__":
    main()
