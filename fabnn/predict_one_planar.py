import argparse
import json
import os

import numpy
import tensorflow as tf

from fabnn.predict import predict_volume_appearance
from fabnn.utils import dump_image, linear_to_sRGB, read_image, resolve_project_path
from fabnn.utils.difference_metrics import rms, ssim

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device the operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(
    sess
)  # set this TensorFlow session as the default session for Keras  # noqa


def predict_one_planar(
    discrete_voxels_file, materials_filename, model_filename, model_config_filename
):
    npz_file = numpy.load(discrete_voxels_file)
    voxels = npz_file["halftoned_voxels"]

    with open(model_config_filename, "r") as json_file:
        model_config = json.load(json_file)

    input_folder, input_filename = os.path.split(discrete_voxels_file)
    dataset_key = "single_volume"

    predictions, _, _ = predict_volume_appearance(
        dataset_base_dir=input_folder,
        dataset={
            dataset_key: {
                "filename": input_filename,
                "metadata": {
                    "height": voxels.shape[1],
                    "width": voxels.shape[2],
                    "tile_size": 32,
                    "pad_x": 0,
                    "pad_y": 0,
                },
            }
        },
        ignore_md5_checks=True,
        model_filename=model_filename,
        model_config=model_config,
        materials_filename=materials_filename,
        stencils_only=True,
    )
    prediction = predictions[dataset_key]

    return prediction


def get_args():
    parser = argparse.ArgumentParser(description="")

    parser.add_argument("-i", "--input", dest="discrete_voxels_file", type=str, required=True)
    parser.add_argument("--gt", dest="gt_rendering_path", type=str, default=None)
    parser.add_argument("-n", "--nn", dest="model_folder", type=str, required=True)
    parser.add_argument(
        "-w", "--weights", dest="model_weights", type=str, default="model_weights.h5"
    )
    parser.add_argument(
        "-c", "--config", dest="model_config", type=str, default="model_metadata.json"
    )
    parser.add_argument(
        "-m",
        "--materials",
        dest="materials_file",
        type=str,
        default=resolve_project_path(
            "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
        ),
    )
    parser.add_argument("-o", "--output", dest="output_image_path", type=str, required=True)
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )

    return parser.parse_args()


def main():
    args = get_args()

    discrete_voxels_file = args.discrete_voxels_file
    gt_rendering_path = args.gt_rendering_path
    output_image_path = args.output_image_path

    if gt_rendering_path:
        gt_rendering = read_image(gt_rendering_path)
    else:
        try:
            npz_file = numpy.load(discrete_voxels_file)
            gt_rendering = npz_file["rendered_image"]
            if (
                gt_rendering == None
            ):  # when None is loaded from numpy, it is "not None", but "== None"   # noqa
                gt_rendering = None
        except Exception:
            gt_rendering = None

    prediction = predict_one_planar(
        discrete_voxels_file=discrete_voxels_file,
        materials_filename=args.materials_file,
        model_filename=os.path.join(args.model_folder, args.model_weights),
        model_config_filename=os.path.join(args.model_folder, args.model_config),
    )
    dump_image(image=prediction, filepath=output_image_path)

    if gt_rendering is not None:
        print("RMSE:", rms(gt_rendering, prediction))
        print("RMSE SRGB:", rms(linear_to_sRGB(gt_rendering), linear_to_sRGB(prediction)))
        print("SSIM:", ssim(gt_rendering, prediction))
        print("SSIM SRGB:", ssim(linear_to_sRGB(gt_rendering), linear_to_sRGB(prediction)))


if __name__ == "__main__":
    main()
