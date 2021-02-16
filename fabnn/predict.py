import math
import os
from typing import Callable, Dict, Generator

import numpy
import pandas as pd
import pyopenvdb

from fabnn.colorspaces import CIEXYZ_primaries, xyz2linear_rgb
from fabnn.data_storage import Data3D, DataInterface, DataPlanar
from fabnn.data_storage.data import MODE_PREDICT, make_batch, make_batch_swap_axes
from fabnn.dataset_utils import classify_dataset_class
from fabnn.materials import populate_materials
from fabnn.models import models_collection

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def make_batches_gen(
    data: DataInterface,
    batch_size: int,
    locations: list,
    make_batch_function: Callable = make_batch,
) -> Generator:
    data_size = len(data)
    i = 0
    while True:
        batch, _, batch_locations = make_batch_function(
            data, i, batch_size, build_size=batch_size, output_locations=True
        )[0]
        locations.extend(batch_locations)
        yield batch
        i += batch_size
        if i >= data_size:
            break


def predict_volume_appearance(
    dataset_base_dir: str,
    dataset: dict,
    model_filename,
    model_config,
    materials_filename,
    stencils_only,
    timings: Dict = None,
    ignore_md5_checks: bool = False,
    verbose_logging: bool = False,
):
    model_params = model_config["model_params"]

    if not os.path.isabs(materials_filename):
        materials_filename = os.path.join(dataset_base_dir, materials_filename)

    # backward compatibility
    if type(model_params["stencil_channels"]) == int:
        model_params["stencil_channels"] = ["scattering", "absorption", "mask"][
            : model_params["stencil_channels"]
        ]

    model_arch_name = model_params["model_arch_name"]
    patch_size = model_params["patch_size"]
    scale_levels = model_params["scale_levels"]
    stencil_channels = model_params["stencil_channels"]

    is_2D_dataset = classify_dataset_class(dataset_base_dir, dataset)

    data_class = DataPlanar if is_2D_dataset else Data3D
    alignment_z_centered = model_params.get("alignment_z_centered", data_class == Data3D)
    data = data_class(
        alignment_z_centered=alignment_z_centered,
        data_items=dataset,
        dataset_base_dir=dataset_base_dir,
        find_inner_material_voxels=model_params["find_inner_material_voxels"],
        ignore_md5_checks=ignore_md5_checks,
        materials_file=materials_filename,
        mode=MODE_PREDICT,
        patch_size=patch_size,
        sat_object_class_name="TreeSAT",
        scale_levels=scale_levels,
        shuffle_patches=False,
        sliding_window_length=1,
        stencil_channels=stencil_channels,
        stencils_only=stencils_only,
        timings=timings,
        verbose_logging=verbose_logging,
    )

    batch_size = int(os.getenv("BATCH_SIZE", 10000))

    model_make_function = models_collection[model_arch_name]
    model = model_make_function(params=model_params)
    model.load_weights(model_filename)

    make_batch_function = (
        make_batch_swap_axes
        if model_arch_name in ("planar_first", "first_baseline")
        else make_batch
    )

    locations = []
    predictions = model.predict_generator(
        generator=make_batches_gen(data, batch_size, locations, make_batch_function),
        steps=math.ceil(len(data) / batch_size),
        verbose=1,
    )

    predicted_images = {}
    predicted_images_accessors = {}
    gt_renderings = {}
    predicted_masks = None
    predicted_masks_accessors = {}

    materials = populate_materials(materials_filename)
    material_channels = [m.channels for m in materials.values()]
    assert max(material_channels) == min(
        material_channels
    ), "number of channels in materials file mismatch"  # all elements equal
    materials_channel_count = material_channels[0]

    if materials_channel_count != 3:
        # spectral mode
        material_wavelengths = [m.wavelengths for m in materials.values()]
        for i in range(1, len(material_wavelengths)):
            assert numpy.array_equal(
                material_wavelengths[i - 1], material_wavelengths[i]
            ), "wavelength definition mismatch in materials file"
        # spectral prediction
        X, Y, Z = CIEXYZ_primaries(material_wavelengths[0] * 10)  # convert nm to Angstrom

    if is_2D_dataset:
        for (datafile_idx, channel, pidx), pixel_prediction in zip(locations, predictions):
            dataset_item_key, metadata = data.volume_files[datafile_idx]
            if dataset_item_key not in predicted_images.keys():
                if not stencils_only:
                    gt_renderings[dataset_item_key] = data.get_render(
                        dataset_item_key=dataset_item_key
                    )

                predicted_images[dataset_item_key] = numpy.empty(
                    shape=(metadata["height"], metadata["width"], 3), dtype=numpy.float32
                )
            x, y, z = data.convert_patch_index(datafile_idx, channel, pidx)
            predicted_images[dataset_item_key][y, x, channel] = pixel_prediction
    else:  # is 3D dataset
        material_voxel_found_index_masks = {
            dataset_item_key: data.material_voxel_found_index[datafile_idx]
            for datafile_idx, (dataset_item_key, _) in enumerate(data.volume_files)
        }
        predicted_masks = {}

        locations_predictions = (
            pd.DataFrame(
                numpy.concatenate([numpy.array(locations), numpy.array(predictions)], axis=1),
                columns=["datafile_idx", "channel", "pidx", "value"],
            )
            .astype(
                {
                    "datafile_idx": "uint16",
                    "channel": "uint8",
                    "pidx": "uint32",
                }
            )
            .sort_values(by=["datafile_idx", "pidx", "channel"])
        )
        num_channels = int(locations_predictions["channel"].max()) + 1

        for lp_idx in range(0, len(locations_predictions), num_channels):
            datafile_idx = int(locations_predictions.at[lp_idx, "datafile_idx"])
            pidx = int(locations_predictions.at[lp_idx, "pidx"])

            dataset_item_key, metadata = data.volume_files[int(datafile_idx)]
            if dataset_item_key not in predicted_images.keys():
                predicted_images[dataset_item_key] = pyopenvdb.Vec3SGrid((-1, -1, -1))
                predicted_images_accessors[dataset_item_key] = predicted_images[
                    dataset_item_key
                ].getAccessor()

                predicted_masks[dataset_item_key] = pyopenvdb.BoolGrid(False)
                predicted_masks_accessors[dataset_item_key] = predicted_masks[
                    dataset_item_key
                ].getAccessor()

                render, _, _, _, _ = data.get_render(dataset_item_key=dataset_item_key)
                gt_renderings[dataset_item_key] = render
                if render is not None:
                    predicted_images[dataset_item_key].transform = render.transform
                else:
                    dataset_item = data.data_items[dataset_item_key]
                    filled_path = dataset_item["proxy_object_filled_path"]
                    if filled_path is not None:
                        normal_grid = pyopenvdb.read(filled_path, "normalGrid")
                        predicted_images[dataset_item_key].transform = normal_grid.transform
                    else:
                        raise RuntimeError(
                            "One of: `render_filename`, `proxy_object_filled_path` "
                            "should be set to extract transform for the predicted grid"
                        )

                predicted_images[dataset_item_key].name = "prediction"
            x, y, z = data.convert_patch_index_to_render_coords(datafile_idx, 0, pidx)

            if num_channels == 3:
                value = (
                    locations_predictions.at[lp_idx + 0, "value"],
                    locations_predictions.at[lp_idx + 1, "value"],
                    locations_predictions.at[lp_idx + 2, "value"],
                )
            else:
                # convolve the renderings with the XYZ color matching functions
                value = [0.0, 0.0, 0.0]
                for i, wavelength in enumerate(material_wavelengths[0]):
                    value[0] += X[i] * locations_predictions.at[lp_idx + i, "value"]
                    value[1] += Y[i] * locations_predictions.at[lp_idx + i, "value"]
                    value[2] += Z[i] * locations_predictions.at[lp_idx + i, "value"]
                value = tuple(xyz2linear_rgb([[value]])[0][0])
            predicted_images_accessors[dataset_item_key].setValueOn((x, y, z), value)
            mask_value = material_voxel_found_index_masks[dataset_item_key][pidx]
            predicted_masks_accessors[dataset_item_key].setValueOn((x, y, z), mask_value)

    return predicted_images, gt_renderings, predicted_masks
