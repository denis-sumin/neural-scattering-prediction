import argparse
import json
import math
import os
from typing import Generator

import matplotlib.pyplot as plt
import numpy

# from nn_rendering_prediction.data_storage.data import one_batch_stencils_swap_axes # make_batch, make_batch_swap_axes, MODE_PREDICT
from nn_rendering_prediction.models import models_collection
from surface_albedo_mapping_ import apply_albedo_mapping_analytical_forward
from utils import ensure_dir

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def one_batch_stencils_swap_axes(stencils):
    size, scale_levels = stencils.shape[:2]
    stencils = stencils.reshape((size, scale_levels, -1))
    reordered_stencils = numpy.moveaxis(stencils, 1, 0)
    return [numpy.squeeze(a) for a in numpy.vsplit(reordered_stencils, scale_levels)]


def make_batches_gen(data: numpy.array, batch_size: int, model_params: dict) -> Generator:
    data_size = data.shape[0]

    patch_size = model_params["patch_size"]
    scale_levels = model_params["scale_levels"]
    stencil_channels = model_params["stencil_channels"]

    i = 0
    while True:
        batch = data[i : min(i + batch_size, data_size)]
        batch = numpy.tile(
            batch[:, None, None, None, None, :], (1, len(scale_levels), *patch_size, 1)
        )
        if model_params["model_arch_name"] in ("planar_first", "first_baseline"):
            batch = one_batch_stencils_swap_axes(batch)
        yield batch
        i += batch_size
        if i >= data_size:
            break


def predict_homogeneous_stencils(model_filename, model_config, scattering_absorption_values):
    model_params = model_config["model_params"]

    # backward compatibility
    if type(model_params["stencil_channels"]) == int:
        model_params["stencil_channels"] = ["scattering", "absorption", "mask"][
            : model_params["stencil_channels"]
        ]

    model_arch_name = model_params["model_arch_name"]

    batch_size = int(os.getenv("BATCH_SIZE", 10000))

    model_make_function = models_collection[model_arch_name]
    model = model_make_function(params=model_params)
    model.load_weights(model_filename)

    predictions = model.predict_generator(
        generator=make_batches_gen(scattering_absorption_values, batch_size, model_params),
        steps=math.ceil(scattering_absorption_values.shape[0] / batch_size),
        verbose=1,
    )
    print(predictions.shape)
    return predictions


def get_args():
    parser = argparse.ArgumentParser(
        description="A script to evaluate networks over scattering + absorption value ranges of homogeneous media"
    )
    parser.add_argument(
        "--scattering-range",
        nargs=3,
        type=float,
        default=[0.1, 30, 0.1],
        help="[start, stop, stepsize]",
    )
    parser.add_argument(
        "--absorption-range",
        nargs=3,
        type=float,
        default=[0.1, 30, 0.1],
        help="[start, stop, stepsize]",
    )
    parser.add_argument("networks", type=str, nargs="+")
    parser.add_argument(
        "-w", "--weights", dest="model_weights", type=str, default="model_weights.h5"
    )
    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, default="model_metadata.json"
    )
    parser.add_argument(
        "-o", "--output", required=True, help="Path to the output folder", type=str
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()

    scattering_range = numpy.arange(*args.scattering_range)
    absorption_range = numpy.arange(*args.absorption_range)

    s_coords, a_coords = numpy.meshgrid(scattering_range, absorption_range)
    input_values = numpy.stack([s_coords, a_coords], axis=-1)

    extent = (*args.absorption_range[:-1], *args.scattering_range[:-1])

    albedo_values = input_values[:, :, 0] / (input_values[:, :, 0] + input_values[:, :, 1])
    analytical = apply_albedo_mapping_analytical_forward(albedo_values)

    predictions = dict()

    plt.figure()
    fig, axes = plt.subplots(1, len(args.networks) + 1)

    for idx, model_folder in enumerate(args.networks):
        model_config_filename = os.path.join(model_folder, args.model_config)
        model_weights_filename = os.path.join(model_folder, args.model_weights)

        with open(model_config_filename, "r") as json_file:
            model_config = json.load(json_file)

        predictions[model_folder] = predict_homogeneous_stencils(
            model_weights_filename, model_config, input_values.reshape(-1, 2)
        ).reshape(*input_values.shape[:-1])

        # flip extent and image to follow matplotlib conventions
        axes[idx].imshow(predictions[model_folder].T, extent=extent, origin="lower")

    axes[-1].imshow(analytical.T, extent=extent, origin="lower")

    output_file = os.path.join(ensure_dir(os.path.dirname(args.output)), "plot.png")
    plt.savefig(output_file)
