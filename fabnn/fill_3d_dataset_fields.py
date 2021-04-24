import argparse
import os
from typing import Callable

import yaml

from fabnn.dataset_utils import get_dataset_item_render_path, get_dataset_item_volume_path
from fabnn.utils import setup_console_logger

logger = setup_console_logger(__name__)


def process_items_list(
    dataset_base_dir, items, save_dataset: Callable, recheck: bool = False, verbose_logging=False
):
    for key, dataset_item in items.items():
        logger.info("Processing {}".format(key))

        if (
            "obj_file_path" in dataset_item
            and "object_rotation" in dataset_item
            and "object_scale" in dataset_item
            and "proxy_grid_dpi" in dataset_item
            and "proxy_object_filled_path" in dataset_item
            and "proxy_object_solution_path" in dataset_item
            and "surface_sampling_density" in dataset_item
            and not recheck
        ):
            continue

        dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
        dataset_render_file_path = get_dataset_item_render_path(dataset_base_dir, dataset_item)

        base_file_path = dataset_volume_file_path.replace("_discrete_volume.gz", "")
        while base_file_path[-1].isdigit():
            base_file_path = base_file_path[:-1]

        obj_file_path = dataset_item.get("obj_file_path", base_file_path + "filled_remeshed.obj")
        proxy_object_filled_path = dataset_item.get(
            "proxy_object_filled_path", base_file_path + "filled.vdb"
        )
        proxy_object_solution_path = dataset_item.get(
            "proxy_object_solution_path",
            dataset_render_file_path.replace("render_result", "solution"),
        )

        for file_path in (obj_file_path, proxy_object_filled_path, proxy_object_solution_path):
            if not os.path.exists(file_path):
                raise RuntimeError("Failed to find file: {}, key: {}".format(file_path, key))

        try:
            task_file = dataset_item["task_file"]
            task_file_key = dataset_item["task_file_key"]
        except KeyError as e:
            logger.error("{} is not set for dataset item {}".format(e, key))
            raise

        with open(task_file, "r") as f:
            task_file_data = yaml.full_load(f)
        try:
            task_data = task_file_data[task_file_key]
        except KeyError:
            logger.error("Task file {} does not contain key {}".format(task_file, task_file_key))
            raise

        dataset_item["obj_file_path"] = obj_file_path
        dataset_item["object_rotation"] = task_data["rotation"]
        dataset_item["object_scale"] = task_data["scale"]
        dataset_item["proxy_grid_dpi"] = task_data["proxy_grid_dpi"]
        dataset_item["proxy_object_filled_path"] = proxy_object_filled_path
        dataset_item["proxy_object_solution_path"] = proxy_object_solution_path
        dataset_item["surface_sampling_density"] = task_data["sampling_density"]
        save_dataset()


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
        "-r", "--recalculate", help="Recalculate image if it exists", action="store_true"
    )
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
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

    def save_dataset():
        with open(args.dataset, "w") as f:
            f.write(yaml.safe_dump(dataset))

    process_items_list(
        dataset_base_dir=dataset_base_dir,
        items=train_data_items,
        recheck=args.recalculate,
        save_dataset=save_dataset,
        verbose_logging=args.verbose,
    )

    process_items_list(
        dataset_base_dir=dataset_base_dir,
        items=validate_data_items,
        recheck=args.recalculate,
        save_dataset=save_dataset,
        verbose_logging=args.verbose,
    )


if __name__ == "__main__":
    main()
