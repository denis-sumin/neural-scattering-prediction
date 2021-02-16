import argparse
import json
import os
from collections import defaultdict
from itertools import repeat
from multiprocessing import cpu_count
from multiprocessing.dummy import Pool
from time import time
from typing import Generator

import numpy
import yaml
from nn_rendering_prediction.data_storage import DataPlanar
from utils import resolve_project_path

if "CUDA_VISIBLE_DEVICES" not in os.environ:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

BATCH_SIZE = int(os.getenv("BATCH_SIZE", 2000))
TRAINING_RATIO = float(os.getenv("TRAINING_RATIO", 0.8))

TIMINGS = defaultdict(int)


def make_batch(data: DataPlanar, start: int, size: int, pool: Pool):
    size = min(len(data), start + size) - start
    stencils = []
    predictions = []

    for step_start, step_end in sorted(data.load_indexes.keys()):
        if step_start <= start <= step_end:
            break

    def add_stencils_predictions(start_, i):
        _, stencil, prediction = data[start_ + i]
        stencils.append(stencil)
        predictions.append(prediction)

    if start + size - 1 <= step_end:
        pool.starmap(add_stencils_predictions, zip(repeat(start), range(size)))
    else:
        size1 = step_end - start + 1
        pool.starmap(add_stencils_predictions, zip(repeat(start), range(size1)))
        size2 = size - size1
        start2 = step_end + 1
        pool.starmap(add_stencils_predictions, zip(repeat(start2), range(size2)))
        # print('---', start, size, 'split into 2:', start, size1, start2, size2)

    scale_levels = stencils[0].shape[0]

    batch = numpy.concatenate(stencils).reshape((size, scale_levels, -1))
    # print(batch.shape, batch.base is stencils)
    reordered_batch = numpy.moveaxis(batch, 1, 0)
    # print(reordered_batch.shape, reordered_batch.base is batch)

    return [numpy.squeeze(a) for a in numpy.vsplit(reordered_batch, scale_levels)], numpy.array(
        predictions
    )


def make_batches_gen(data: DataPlanar, batch_size: int, reshuffle: bool = False) -> Generator:
    data_size = len(data)
    i = 0
    k = 0
    with Pool(min(cpu_count(), 5)) as pool:
        while True:
            ts = time()
            yield make_batch(data, i, batch_size, pool)
            TIMINGS[("make_batch", data)] += time() - ts

            i += batch_size
            if i >= data_size:
                i = 0
                k += 1
                if reshuffle:
                    data.generate_patches(seed_=k)
                    data_size = len(data)


def get_args():
    parser = argparse.ArgumentParser(description="Run an full optimization on a given task file")

    parser.add_argument("-d", "--data-dir", dest="data_dir", type=str, required=True)
    parser.add_argument("-r", "--render-dir", dest="render_dir", type=str, default=None)
    parser.add_argument("-c", "--config", dest="model_config", type=str, required=True)
    parser.add_argument(
        "-m",
        "--materials",
        dest="materials_file",
        type=str,
        default=resolve_project_path(
            "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
        ),
    )
    parser.add_argument(
        "-t",
        "--train-patches",
        dest="train_patches",
        type=int,
        default=None,
        help="Number of patches to take from the train data",
    )
    parser.add_argument("--train-set", dest="train_set_name", type=str, default="data_train")
    parser.add_argument("--test-set", dest="test_set_name", type=str, default="data_test")
    parser.add_argument("--memory-volumes", dest="memory_volumes", type=int, default=None)
    parser.add_argument("--rotations", dest="rotations", type=int, default=None)
    parser.add_argument("--initial-epoch", dest="initial_epoch", type=int, default=0)
    parser.add_argument(
        "-v", "--verbose", help="More debug prints and logging messages", action="store_true"
    )

    return parser.parse_args()


def main():
    args = get_args()

    if args.train_patches is None:
        training_patches_number = None
    else:
        training_patches_number = args.train_patches
    batch_size = BATCH_SIZE

    with open(args.model_config, "r") as json_file:
        model_config = json.load(json_file)

    sets = {args.train_set_name: [], args.test_set_name: []}
    for set_name, set_data in sets.items():
        with open(os.path.join(args.data_dir, set_name + ".yml"), "r") as f:
            dataset_config = yaml.full_load(f)
            for group in dataset_config["groups"]:
                group_items = group["items"]
                for item in group_items:
                    set_data.append(item["filename"])

    train_data = DataPlanar(
        data_dir=args.data_dir,
        data_items=sets[args.train_set_name],
        limit_patches_number=training_patches_number,
        materials_file=args.materials_file,
        patch_size=model_config["patch_size"],
        render_dir=args.render_dir,
        rotations=args.rotations,
        scale_levels=model_config["scale_levels"],
        sliding_window_length=args.memory_volumes,
        stencil_channels=model_config["stencil_channels"],
        verbose_logging=args.verbose,
    )

    print("Training on {} patches".format(len(train_data)))

    ts = time()
    for idx, (batch_x, batch_y) in enumerate(
        make_batches_gen(train_data, batch_size=batch_size, reshuffle=True)
    ):
        if not idx % 1000:
            print(idx, len(batch_x), batch_x[0].shape, batch_y.shape)
    timing = time() - ts
    print("{} batches prepared in {} seconds".format(idx, timing))


if __name__ == "__main__":
    main()
