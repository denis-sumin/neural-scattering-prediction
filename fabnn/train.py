import argparse
import json
import math
import os
import shutil
import socket
from copy import deepcopy
from datetime import datetime
from pprint import pformat
from typing import Callable, Dict, Generator

import numpy
import psutil
import tensorflow as tf
import yaml
from tensorflow.python.keras.callbacks import LambdaCallback, ModelCheckpoint

from fabnn.callbacks import TensorBoard
from fabnn.data_storage import Data, Data3D, DataPlanar
from fabnn.data_storage.data import MODE_TRAIN, make_batch, make_batch_swap_axes
from fabnn.dataset_utils import classify_dataset_class
from fabnn.models import models_collection
from fabnn.utils import (
    ensure_dir,
    get_git_revision_short_hash,
    human_size,
    setup_console_logger,
    update_timings_dict,
)

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU
# sess = tf.compat.v1.Session(config=config)
# tf.compat.v1.keras.backend.set_session(sess)  # set this TensorFlow session as the default session for Keras  # noqa

TIMINGS_TRAIN = dict()
TIMINGS_VALIDATE = dict()

logger = setup_console_logger(__name__)


def make_batches_gen(
    data: Data,
    data_name: str,
    batch_size: int,
    batch_build_length: int = None,
    initial_data_batch_start_index: int = 0,
    initial_data_shuffle_random_seed: int = 0,
    make_batch_function: Callable = make_batch,
    reshuffle: bool = False,
) -> Generator:
    data_size = len(data)
    i = initial_data_batch_start_index
    k = initial_data_shuffle_random_seed
    if k != 0:
        data.generate_patches(seed_=k)
    if batch_build_length is None:
        batch_build_length = batch_size
    else:
        batch_build_length = max(batch_build_length, batch_size)
        batch_build_length = int(round(batch_build_length / batch_size) * batch_size)
    while True:
        batches = make_batch_function(data, i, batch_size, batch_build_length)
        for batch in batches:
            yield batch
        i += batch_build_length
        if i >= data_size:
            i = 0
            k += 1
            logger.info("{}: Full round over the dataset complete.".format(data_name))
            if reshuffle:
                data.generate_patches(seed_=k)
                data_size = len(data)


def train(
    batch_build_length: int,
    batch_size: int,
    metadata: dict,
    model_config: Dict,
    output_dir: str,
    test_data: Data,
    train_data: Data,
    initial_epoch: int = 0,
    initial_weights: str = "",
    tensorboard_verbose: bool = False,
    train_epochs: int = 10,
    train_steps_per_epoch_limit: int = None,
    validation_freq: int = 1,
    verbose_logging: bool = False,
):
    if initial_epoch > 0 and initial_weights:
        raise ValueError("Only one of: initial_epoch or initial_weights - can be set at once")

    model_arch_name = model_config["model_arch_name"]
    weights_best_filename = os.path.join(output_dir, "model_weights.h5")
    checkpoint_filename = os.path.join(output_dir, "model_checkpoint.h5")
    logdir = os.path.join(output_dir, "tf_logs")

    if initial_epoch > 0:
        if os.path.exists(checkpoint_filename):
            logger.info("Loading model: {}".format(checkpoint_filename))
            model = tf.keras.models.load_model(checkpoint_filename)
        else:
            raise RuntimeError(
                "{} not found, you should start from scratch".format(checkpoint_filename)
            )
    else:  # initial_epoch == 0
        patch_size = model_config["patch_size"]
        feature_shape = (
            patch_size[0],
            patch_size[1],
            patch_size[2],
            len(model_config["stencil_channels"]),
        )

        model_make_function = models_collection[model_arch_name]

        model_params = deepcopy(model_config)
        model_params.update(
            {
                "feature_shape": feature_shape,
                "voxel_size": tuple(train_data.voxel_size),
            }
        )

        model = model_make_function(params=model_params)
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])

        ensure_dir(output_dir)
        with open(os.path.join(output_dir, "model.json"), "w") as json_file:
            json_file.write(model.to_json())

        metadata["model_params"] = model_params
        with open(os.path.join(output_dir, "model_metadata.json"), "w") as json_file:
            json.dump(metadata, json_file, indent=4, sort_keys=True)

        if initial_weights:
            logger.info("Loading weights: {}".format(initial_weights))
            model.load_weights(initial_weights)

        if os.path.exists(logdir):
            shutil.rmtree(logdir)

    checkpoint_best_callback = ModelCheckpoint(
        filepath=weights_best_filename,
        monitor="val_loss",
        verbose=1,
        save_best_only=True,
        save_weights_only=True,
        mode="auto",
        period=1,
    )

    checkpoint_all_callback = ModelCheckpoint(
        filepath=checkpoint_filename,
        verbose=1,
        save_best_only=False,
        save_weights_only=False,
        mode="auto",
        period=1,
    )

    tensorboard_callback = TensorBoard(
        log_dir=logdir,
        histogram_freq=25 if tensorboard_verbose else 0,
        write_grads=tensorboard_verbose,
        write_images=tensorboard_verbose,
        profile_batch=0,
    )

    def process_timings(epoch, logs):
        this_epoch_train_timings = train_data.get_and_clean_timings()
        this_epoch_test_timings = test_data.get_and_clean_timings()
        update_timings_dict(TIMINGS_TRAIN, this_epoch_train_timings)
        update_timings_dict(TIMINGS_VALIDATE, this_epoch_test_timings)
        logger.info("Epoch {}".format(epoch))
        logger.info("Data.train timings: {}".format(pformat(this_epoch_train_timings)))
        logger.info("Data.test timings: {}".format(pformat(this_epoch_test_timings)))

    process_timings_callback = LambdaCallback(
        on_epoch_begin=lambda epoch, logs: print(), on_epoch_end=process_timings
    )

    make_batch_function = (
        make_batch_swap_axes
        if model_arch_name in ("planar_first", "first_baseline")
        else make_batch
    )

    train_steps_per_epoch = math.ceil(len(train_data) / batch_size)
    if train_steps_per_epoch_limit is not None:
        train_steps_per_epoch = min(train_steps_per_epoch, train_steps_per_epoch_limit)

    initial_train_data_batch_start_index = (
        initial_epoch * train_steps_per_epoch * batch_size
    ) % len(train_data)
    initial_data_shuffle_random_seed = (
        initial_epoch * train_steps_per_epoch * batch_size
    ) // len(train_data)

    callbacks = [
        checkpoint_best_callback,
        checkpoint_all_callback,
        tensorboard_callback,
        process_timings_callback,
    ]

    model.fit_generator(
        generator=make_batches_gen(
            data=train_data,
            data_name="train_data",
            batch_size=batch_size,
            batch_build_length=batch_build_length,
            initial_data_batch_start_index=initial_train_data_batch_start_index,
            initial_data_shuffle_random_seed=initial_data_shuffle_random_seed,
            make_batch_function=make_batch_function,
            reshuffle=True,
        ),
        steps_per_epoch=train_steps_per_epoch,
        validation_data=make_batches_gen(
            data=test_data,
            data_name="validation_data",
            batch_size=batch_size,
            make_batch_function=make_batch_function,
            reshuffle=False,
        ),
        validation_freq=validation_freq,
        validation_steps=math.ceil(len(test_data) / batch_size),
        epochs=train_epochs,
        verbose=2,
        callbacks=callbacks,
        initial_epoch=initial_epoch,
    )


def get_args():
    parser = argparse.ArgumentParser(description="Run an full optimization on a given task file")

    parser.add_argument("-d", "--dataset", dest="dataset", type=str, required=True)
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument("-o", "--output", dest="output_model_directory", type=str, required=True)
    parser.add_argument(
        "-t",
        "--train-patches",
        dest="train_patches",
        type=int,
        default=None,
        help="Number of patches to take from the train data",
    )
    parser.add_argument(
        "-e",
        "--train-epoch-limit",
        dest="train_steps_per_epoch_limit",
        type=int,
        default=None,
        help="Maximum number of batches per epoch in the training",
    )
    parser.add_argument(
        "-b",
        "--batch-size",
        dest="batch_size",
        type=int,
        default=2000,
        help="Number of patches in the batch",
    )
    parser.add_argument(
        "-bb",
        "--batch-build-size",
        dest="batch_build_length",
        type=int,
        default=2000,
        help="Number of patches in the batch",
    )
    parser.add_argument(
        "--validation-patches",
        dest="validation_patches",
        type=int,
        default=None,
        help="Number of patches to take from the validation data",
    )
    parser.add_argument("--memory-limit", dest="memory_limit", type=float, default=None)
    parser.add_argument("--memory-volumes", dest="memory_volumes", type=int, default=None)
    parser.add_argument("--rotations", dest="rotations", type=int, default=None)
    parser.add_argument("--initial-epoch", dest="initial_epoch", type=int, default=0)
    parser.add_argument("--initial-weights", dest="initial_weights", type=str, default=None)
    parser.add_argument("--epochs", dest="train_epochs", type=int, default=2000)
    parser.add_argument("--validation-freq", dest="validation_freq", type=int, default=1)
    parser.add_argument(
        "-tt",
        "--tensorboard-verbose",
        help="Additional outputs to tensorboard",
        action="store_true",
    )
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )

    return parser.parse_args()


def main():
    args = get_args()

    with open(args.model_config, "r") as json_file:
        model_config = json.load(json_file)

    # backward compatibility
    if type(model_config["stencil_channels"]) == int:
        model_config["stencil_channels"] = ["scattering", "absorption", "mask"][
            : model_config["stencil_channels"]
        ]

    dataset_base_dir = os.path.abspath(os.path.split(args.dataset)[0])
    with open(args.dataset, "r") as f:
        dataset = yaml.full_load(f)

    train_data_items = dataset["items"]
    validate_data_items = dataset["items_validate"]
    materials_file = dataset["materials_file"]

    train_data_class = (
        DataPlanar if classify_dataset_class(dataset_base_dir, train_data_items) else Data3D
    )
    validate_data_class = (
        DataPlanar if classify_dataset_class(dataset_base_dir, validate_data_items) else Data3D
    )

    if args.memory_limit:
        sliding_window_memory_limit = psutil.virtual_memory().total * args.memory_limit
        logger.info(
            "Sliding window memory limit is set to {}".format(
                human_size(sliding_window_memory_limit)
            )
        )
    else:
        sliding_window_memory_limit = None

    train_data = train_data_class(
        alignment_z_centered=model_config.get("alignment_z_centered", train_data_class == Data3D),
        data_items=train_data_items,
        dataset_base_dir=dataset_base_dir,
        find_inner_material_voxels=model_config["find_inner_material_voxels"],
        limit_patches_number=args.train_patches,
        materials_file=materials_file,
        mode=MODE_TRAIN,
        patch_size=model_config["patch_size"],
        rotations=args.rotations,
        sat_object_class_name="TreeSAT",
        scale_levels=model_config["scale_levels"],
        shuffle_patches=True,
        sliding_window_length=args.memory_volumes,
        sliding_window_memory_limit=sliding_window_memory_limit,
        stencil_channels=model_config["stencil_channels"],
        verbose_logging=args.verbose,
    )

    test_data = validate_data_class(
        alignment_z_centered=model_config.get("alignment_z_centered", train_data_class == Data3D),
        data_items=validate_data_items,
        dataset_base_dir=dataset_base_dir,
        find_inner_material_voxels=model_config["find_inner_material_voxels"],
        limit_patches_number=args.validation_patches,
        materials_file=materials_file,
        mode=MODE_TRAIN,
        patch_size=model_config["patch_size"],
        rotations=1,
        sat_object_class_name="TreeSAT",
        scale_levels=model_config["scale_levels"],
        shuffle_patches=True,  # should be enabled if we limit the number of validation patches
        sliding_window_length=1,
        stencil_channels=model_config["stencil_channels"],
        verbose_logging=args.verbose,
    )

    logger.info("Training on {} patches".format(len(train_data)))
    logger.info("Validation will be performed on {} patches".format(len(test_data)))

    model_name = os.path.split(args.output_model_directory)[1]

    batch_build_length = args.batch_build_length
    batch_size = args.batch_size
    train_steps_per_epoch_limit = args.train_steps_per_epoch_limit

    metadata = {
        "batch_build_length": batch_build_length,
        "batch_size": batch_size,
        "dataset_path": args.dataset,
        "git_commit": get_git_revision_short_hash(),
        "hostname": socket.gethostname(),
        "memory-volumes": args.memory_volumes,
        "model_name": model_name,
        "rotations": args.rotations,
        "train_data_items": dataset["items"],
        "train_start_time": datetime.now().strftime("%Y-%m-%d-%H:%M:%S"),
        "train_steps_per_epoch_limit": train_steps_per_epoch_limit,
        "training_patches_number": len(train_data),
        "validate_data_items": dataset["items_validate"],
        "validation_patches_number": len(test_data),
        "sliding-window-volumes-average": round(
            numpy.array(
                [len(volume_idxs) for data_idxs, volume_idxs in train_data.load_indexes.items()]
            ).mean(),
            2,
        ),
    }

    try:
        train(
            batch_build_length=batch_build_length,
            batch_size=batch_size,
            metadata=metadata,
            model_config=model_config,
            output_dir=args.output_model_directory,
            test_data=test_data,
            train_data=train_data,
            # the arguments below are optional
            initial_epoch=args.initial_epoch,
            initial_weights=args.initial_weights,
            tensorboard_verbose=args.tensorboard_verbose,
            train_epochs=args.train_epochs,
            train_steps_per_epoch_limit=train_steps_per_epoch_limit,
            validation_freq=args.validation_freq,
            verbose_logging=args.verbose,
        )
    except KeyboardInterrupt:
        logger.info("Data.train timings: {}".format(pformat(TIMINGS_TRAIN)))
        logger.info("Data.test timings: {}".format(pformat(TIMINGS_VALIDATE)))
        raise


if __name__ == "__main__":
    main()
