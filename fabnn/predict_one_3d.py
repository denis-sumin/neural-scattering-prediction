import argparse
import json
import os

import numpy
import pyopenvdb
import tensorflow as tf

from fabnn.predict import predict_volume_appearance
from fabnn.utils import linear_to_sRGB, resolve_project_path
from fabnn.utils.difference_metrics import rms

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# config.log_device_placement = True  # to log device placement (on which device an operation ran)
sess = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(
    sess
)  # set this TensorFlow session as the default session for Keras  # noqa


def predict_one_3d(
    discrete_voxels_file: str,
    filled_vdb_path: str,
    materials_filename: str,
    model_filename: str,
    model_config_filename: str,
):

    if not os.path.abspath(discrete_voxels_file):
        discrete_voxels_file = os.path.abspath(discrete_voxels_file)

    if not os.path.isabs(filled_vdb_path):
        filled_vdb_path = os.path.abspath(filled_vdb_path)

    input_folder, input_filename = os.path.split(discrete_voxels_file)
    dataset_key = "single_volume"

    with open(model_config_filename, "r") as json_file:
        model_config = json.load(json_file)

    predictions, _, _ = predict_volume_appearance(
        dataset_base_dir=input_folder,
        dataset={
            dataset_key: {
                "filename": input_filename,
                "proxy_object_filled_path": filled_vdb_path,
                "metadata": {
                    "num_surface_voxels": pyopenvdb.read(
                        filled_vdb_path, "normalGrid"
                    ).activeVoxelCount(),
                    "tile_size": 24,
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
    parser.add_argument("-f", "--filled", dest="filled_vdb_path", type=str, required=True)
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
    parser.add_argument("-o", "--output", dest="output_vdb_path", type=str, required=True)
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )

    return parser.parse_args()


def main():
    args = get_args()

    discrete_voxels_file = args.discrete_voxels_file
    gt_rendering_path = args.gt_rendering_path
    output_vdb_path = args.output_vdb_path

    if gt_rendering_path:
        gt_rendering = pyopenvdb.read(gt_rendering_path, "rendering")
    else:
        gt_rendering = None

    prediction = predict_one_3d(
        discrete_voxels_file=discrete_voxels_file,
        filled_vdb_path=args.filled_vdb_path,
        materials_filename=args.materials_file,
        model_filename=os.path.join(args.model_folder, args.model_weights),
        model_config_filename=os.path.join(args.model_folder, args.model_config),
    )
    pyopenvdb.write(output_vdb_path, [prediction])

    if gt_rendering is not None:
        prediction_acc = prediction.getConstAccessor()
        diff_list = []
        for item in gt_rendering.citerOnValues():
            if item.count == 1:
                target = item.value
                prediction, prediction_active = prediction_acc.probeValue(item.min)
                if prediction_active:
                    diff_list.append([target, prediction])
        if diff_list:
            array = numpy.array(diff_list)
        else:
            array = numpy.zeros((1, 2, 3))
        gt_rendering = array[:, 0, :]
        prediction = array[:, 1, :]

        print("RMSE:", rms(gt_rendering, prediction))
        print("RMSE SRGB:", rms(linear_to_sRGB(gt_rendering), linear_to_sRGB(prediction)))


if __name__ == "__main__":
    main()
