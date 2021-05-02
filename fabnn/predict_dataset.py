import argparse
import json
import os
from pprint import pprint

import matplotlib.pyplot as plt
import numpy
import pyopenvdb
import tensorflow as tf
import yaml

from fabnn.dataset_utils import (
    get_dataset_item_render_path,
    get_dataset_item_volume_path,
    is_2D_file,
)
from fabnn.predict import predict_volume_appearance
from fabnn.utils import (
    dump_image,
    ensure_dir,
    linear_to_sRGB,
    md5sum_path,
    read_image,
    setup_console_logger,
)
from fabnn.utils.difference_metrics import get_colormap, get_difference_metric

logger = setup_console_logger(__name__)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(
    sess
)  # set this TensorFlow session as the default session for Keras  # noqa


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True)
    parser.add_argument("-n", "--nn", dest="model_folder", type=str, required=True)
    parser.add_argument(
        "-w", "--weights", dest="model_weights", type=str, default="model_weights.h5"
    )
    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, default="model_metadata.json"
    )
    parser.add_argument(
        "-o", "--output", dest="base_output_directory", type=str, default="results/nn_prediction"
    )
    parser.add_argument(
        "-r", "--recalculate", help="Recalculate image if it exists", action="store_true"
    )
    parser.add_argument(
        "--skip-md5-check",
        dest="skip_md5_check",
        help="Skip recalculation if model weights file has been updated",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )

    return parser.parse_args()


def main():
    args = get_args()

    dataset_base_dir = os.path.abspath(os.path.split(args.dataset)[0])
    dataset_name = os.path.splitext(os.path.split(args.dataset)[1])[0]

    with open(args.dataset) as f:
        dataset_data = yaml.full_load(f)

    model_folder = os.path.normpath(args.model_folder)

    with open(os.path.join(model_folder, args.model_config), "r") as json_file:
        model_config = json.load(json_file)

    _, nn_model_name = os.path.split(model_folder)
    results_output_directory = os.path.join(
        args.base_output_directory, dataset_name, nn_model_name
    )

    model_data = {
        "name": nn_model_name,
        "config": model_config,
    }

    model_weights_filepath = os.path.join(model_folder, args.model_weights)
    model_weights_file_md5 = md5sum_path(model_weights_filepath)
    output_yaml_file = os.path.join(results_output_directory, "summary.yml")

    if os.path.exists(output_yaml_file):
        with open(output_yaml_file, "r") as yaml_file:
            previous_prediction_data = yaml.full_load(yaml_file)
    else:
        previous_prediction_data = None

    results_data = {}
    dataset = dataset_data.get("items", {})

    predictions = {}
    prediction_masks = {}
    gt_renderings = {}
    to_predict_dataset = {}

    predicted_items = set()

    for item_key, dataset_item in dataset.items():
        dataset_item["is_2D"] = is_2D_file(
            get_dataset_item_volume_path(dataset_base_dir, dataset_item)
        )

        output_prediction_path = os.path.join(
            results_output_directory,
            "{}.{}".format(item_key, "exr" if dataset_item["is_2D"] else "vdb"),
        )
        dataset_item["output_prediction_path"] = output_prediction_path

        previous_prediction_md5 = (
            (
                previous_prediction_data.get("items", {})
                .get(item_key, {})
                .get("model_weights_file_md5")
            )
            if previous_prediction_data
            else None
        )
        md5_recalculate = (
            not args.skip_md5_check
            and previous_prediction_data
            and previous_prediction_md5 is not None
            and previous_prediction_md5 != model_weights_file_md5
        )

        if not os.path.exists(output_prediction_path) or args.recalculate or md5_recalculate:
            to_predict_dataset[item_key] = dataset_item
        else:
            render_filename = get_dataset_item_render_path(dataset_base_dir, dataset_item)
            if dataset_item["is_2D"]:
                predictions[item_key] = read_image(output_prediction_path)
                gt_renderings[item_key] = read_image(render_filename)
            else:
                predictions[item_key] = pyopenvdb.read(output_prediction_path, "prediction")
                gt_renderings[item_key] = pyopenvdb.read(render_filename, "rendering")

    prediction_timings = {}
    if len(to_predict_dataset.keys()):
        predictions_new, gt_renderings_new, prediction_masks_new = predict_volume_appearance(
            dataset=to_predict_dataset,
            dataset_base_dir=dataset_base_dir,
            materials_filename=dataset_data["materials_file"],
            model_config=model_config,
            model_filename=model_weights_filepath,
            stencils_only=False,
            timings=prediction_timings,
            verbose_logging=args.verbose,
        )
        for item_key, prediction in predictions_new.items():
            dataset_item = dataset[item_key]
            folder_path, file_name = os.path.split(dataset_item["output_prediction_path"])
            ensure_dir(folder_path)
            if dataset_item["is_2D"]:
                dump_image(image=prediction, filepath=dataset_item["output_prediction_path"])
            else:
                pyopenvdb.write(dataset_item["output_prediction_path"], [prediction])
            predicted_items.add(item_key)
        predictions.update(predictions_new)
        if prediction_masks_new:
            prediction_masks.update(prediction_masks_new)
        gt_renderings.update(gt_renderings_new)

    if args.verbose:
        print("Prediction timings:")
        pprint(prediction_timings)

    # That's temporary for the deadline - normalize diff images
    max_diff_pos, max_diff_neg = 0.0, 0.0

    for item_key, dataset_item in dataset.items():
        prediction = predictions[item_key]
        gt_rendering = gt_renderings[item_key]

        if not dataset_item["is_2D"]:
            prediction_grid = prediction
            prediction_acc = prediction_grid.getConstAccessor()
            try:
                prediction_mask_accessor = prediction_masks[item_key].getConstAccessor()
            except KeyError:
                prediction_mask_accessor = None

            coords_list = []
            diff_list = []

            for item in gt_rendering.citerOnValues():
                if item.count == 1:
                    target = item.value
                    prediction, prediction_active = prediction_acc.probeValue(item.min)
                    if prediction_mask_accessor:
                        mask_value = prediction_mask_accessor.getValue(item.min)
                    else:
                        mask_value = True
                    if prediction_active and mask_value:
                        coords_list.append(item.min)
                        diff_list.append([target, prediction])
            if diff_list:
                array = numpy.array(diff_list)
            else:
                array = numpy.zeros((1, 2, 3))
            gt_rendering = array[:, 0, :]
            prediction = array[:, 1, :]

            diff = prediction - gt_rendering
            diff_vis_scaling = 4.0
            diff_vis_array = plt.get_cmap(get_colormap("error"))(
                ((diff_vis_scaling * diff) / 2.0 + 0.5).mean(axis=1)
            )

            diff_positive = numpy.maximum(diff, 0.0)
            diff_negative = numpy.minimum(diff, 0.0)
            max_diff_pos = max(max_diff_pos, diff_positive.mean(axis=1).max())
            max_diff_neg = max(max_diff_neg, (-1.0 * diff_negative.mean(axis=1)).max())

            diff_grid = pyopenvdb.Vec3SGrid((-1, -1, -1))
            diff_grid.transform = prediction_grid.transform
            diff_grid.name = "diff_red-green"
            diff_grid_accessor = diff_grid.getAccessor()
            for coord, diff_vis_value in zip(coords_list, diff_vis_array):
                diff_grid_accessor.setValueOn(
                    coord, (diff_vis_value[0], diff_vis_value[1], diff_vis_value[2])
                )

            diff_dE = get_difference_metric("ciede2000")(gt_rendering, prediction)
            diff_dE_vis_scaling = 20.0
            diff_dE_vis_array = plt.get_cmap(get_colormap("ciede2000"))(
                diff_dE / diff_dE_vis_scaling
            )

            diff_dE_grid = pyopenvdb.Vec3SGrid((-1, -1, -1))
            diff_dE_grid.transform = prediction_grid.transform
            diff_dE_grid.name = "diff_dE2000_20max"
            diff_dE_grid_accessor = diff_dE_grid.getAccessor()
            for coord, diff_vis_value in zip(coords_list, diff_dE_vis_array[0]):
                diff_dE_grid_accessor.setValueOn(
                    coord, (diff_vis_value[0], diff_vis_value[1], diff_vis_value[2])
                )
            pyopenvdb.write(
                dataset_item["output_prediction_path"], [prediction_grid, diff_grid, diff_dE_grid]
            )

        rmse_linear = float(get_difference_metric("rms")(gt_rendering, prediction))
        rmse_srgb = float(
            get_difference_metric("rms")(linear_to_sRGB(gt_rendering), linear_to_sRGB(prediction))
        )
        if dataset_item["is_2D"]:
            ssim_linear = float(get_difference_metric("ssim")(gt_rendering, prediction))
            ssim_srgb = float(
                get_difference_metric("ssim")(
                    linear_to_sRGB(gt_rendering), linear_to_sRGB(prediction)
                )
            )
        else:
            ssim_linear = 0.0
            ssim_srgb = 0.0

        print(item_key, "RMSE:", rmse_linear)
        print(item_key, "RMSE SRGB:", rmse_srgb)
        print(item_key, "SSIM:", ssim_linear)
        print(item_key, "SSIM SRGB:", ssim_srgb)

        volume_filename = get_dataset_item_volume_path(dataset_base_dir, dataset_item)
        render_filename = get_dataset_item_render_path(dataset_base_dir, dataset_item)


        results_data[item_key] = {
            "volume_filename": os.path.abspath(volume_filename),
            "render_filename": os.path.abspath(render_filename),
            "prediction_filename": os.path.abspath(output_prediction_path),
            "base_image_name": dataset_item.get("base_image_name", ""),
            "rmse_linear": rmse_linear,
            "rmse_srgb": rmse_srgb,
            "ssim_linear": ssim_linear,
            "ssim_srgb": ssim_srgb,
            "model_weights_file_md5": model_weights_file_md5,
        }

    print("max_diff_pos=", max_diff_pos, "max_diff_neg=", max_diff_neg)

    with open(output_yaml_file, "w") as f:
        yaml.dump({"model": model_data, "items": results_data}, f)


if __name__ == "__main__":
    main()
