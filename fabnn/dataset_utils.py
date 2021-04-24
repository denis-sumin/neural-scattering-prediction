import os

import numpy


def resolve_dataset_item_file_path(dataset_base_dir, dataset_item, item_attr, required=True):
    dataset_file_path = dataset_item.get(item_attr)
    if dataset_file_path is None:
        if required:
            raise ValueError(
                "{} dataset field is missing but set to `required`".format(item_attr)
            )
        else:
            return None
    if not os.path.isabs(dataset_file_path):
        dataset_file_path = os.path.join(dataset_base_dir, dataset_file_path)
    return dataset_file_path


def get_dataset_item_volume_path(dataset_base_dir, dataset_item):
    return resolve_dataset_item_file_path(
        dataset_base_dir, dataset_item, "filename", required=True
    )


def get_dataset_item_render_path(dataset_base_dir, dataset_item):
    dataset_render_file_path = dataset_item.get("render_filename")
    if dataset_render_file_path is not None:
        if not os.path.isabs(dataset_render_file_path):
            dataset_render_file_path = os.path.join(dataset_base_dir, dataset_render_file_path)
    else:
        dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
        if dataset_volume_file_path.endswith("npz"):
            d = numpy.load(dataset_volume_file_path, allow_pickle=True)
            try:
                render = d["rendered_image"]
                h, w = render.shape[:2]
            except (IndexError, KeyError, ValueError):
                pass
            else:
                dataset_render_file_path = dataset_volume_file_path
    return dataset_render_file_path


def is_2D_file(filepath):
    return filepath.endswith("npz") and not is_3D_sa_grid_file(filepath)


def is_3D_file(filepath):
    return is_3D_label_grid_file(filepath) or is_3D_sa_grid_file(filepath)


def is_3D_label_grid_file(filepath):
    return filepath.endswith("discrete_volume.gz")


def is_3D_sa_grid_file(filepath):
    return filepath.endswith("sagrid.npz")


def classify_dataset_class(dataset_base_dir: str, dataset: dict) -> bool:
    """
    returns a bool if the dataset is 2D or not (3D)
    """
    is_2D = None

    mix_error = "Mixed 2D/3D items in dataset - cannot classify {}"
    for key, data_item_dict in dataset.items():
        dataset_volume_file_path = get_dataset_item_volume_path(dataset_base_dir, data_item_dict)
        if is_2D_file(dataset_volume_file_path):
            if is_2D is None:
                is_2D = True
            if not is_2D:
                raise ValueError(mix_error.format(dataset_volume_file_path))
        elif is_3D_file(dataset_volume_file_path):
            if is_2D is None:
                is_2D = False
            if is_2D:
                raise ValueError(mix_error.format(dataset_volume_file_path))
        else:
            raise ValueError(
                "Cannot classify {} into 2D/3D pipeline".format(dataset_volume_file_path)
            )

    return is_2D
