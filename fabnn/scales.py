import os
import sys
from multiprocessing import Process, Queue, cpu_count

import numpy
from downscale import downscale_local_mean, downscale_local_mean_sat

CPU_THREADS = os.getenv("CPU_THREADS", cpu_count())


def generate_scales(scale_levels, patch_size, volume, y, x):
    max_level_xy = max((item[0] for item in scale_levels))
    max_level_z = max((item[1] for item in scale_levels))
    layers, height, width = volume.shape[:3]
    patch_size_z, patch_size_y, patch_size_x = patch_size
    if (
        x - patch_size_x * max_level_xy // 2 < 0
        or x + patch_size_x * max_level_xy // 2 > width
        or y - patch_size_y * max_level_xy // 2 < 0
        or y + patch_size_y * max_level_xy // 2 > height
        or patch_size_z * max_level_z > layers
    ):
        raise ValueError(
            "Insufficient padding to generate all scales "
            "{} {} {} {} {}".format(x, width, y, height, layers)
        )

    patch_scales = [
        volume[
            : patch_size[0],
            y - patch_size[1] // 2 : y + patch_size[1] // 2 + 1,
            x - patch_size[2] // 2 : x + patch_size[2] // 2 + 1,
        ].flatten()
    ]
    scale_levels = tuple([s for s in scale_levels if s != (1, 1)])

    for level_xy, level_z in scale_levels:
        w = (patch_size[0] * level_z, patch_size[1] * level_xy, patch_size[2] * level_xy)
        patch = volume[: w[0], y - w[1] // 2 : y + w[1] // 2, x - w[2] // 2 : x + w[2] // 2]
        scale_kernel = numpy.array((level_z, level_xy, level_xy, 1), dtype=numpy.uint32)
        patch_scales.append(downscale_local_mean(patch, scale_kernel).flatten())
    return patch_scales


def generate_scales_sat(scale_levels, patch_size, volume_sat, y, x):
    patch_scales = numpy.empty(shape=(len(scale_levels), *patch_size, 2), dtype=numpy.float32)
    for level_idx, (level_xy, level_z) in enumerate(scale_levels):
        scale_kernel = (level_z, level_xy, level_xy, 1)

        patch_scales[level_idx, :] = downscale_local_mean_sat(
            volume_sat, x, y, patch_size, scale_kernel
        )
    return patch_scales


class ScaleWorker(Process):
    def __init__(self, tasks_queue, results_queue, data=None, volume=None):
        self.tasks_queue = tasks_queue
        self.results_queue = results_queue

        if data is not None and volume is not None:
            raise ValueError("You should use either data or volume mode")
        if data is not None:
            self.data = data
            self.run = self.__run_on_data
        elif volume is not None:
            self.volume = volume
            self.run = self.__run_on_volume
        else:
            raise ValueError("Neither data nor volume was specified")
        Process.__init__(self)

    def __run_on_data(self):
        while True:
            idx = self.tasks_queue.get()
            patch_scales, value = self.data[idx]
            self.results_queue.put((patch_scales, value))

    def __run_on_volume(self):
        while True:
            (y, x), patch_size, scale_levels = self.tasks_queue.get()
            patch_scales = generate_scales_sat(scale_levels, patch_size, self.volume, y, x)
            self.results_queue.put(((y, x), patch_scales))


class ScaleWorkerManager:
    def __init__(self, data=None, volume=None):
        self.data = data
        self.volume = volume
        self.results_queue = Queue()
        self.tasks_queue = Queue()
        self.workers = []

    def __enter__(self):
        print("Threads", CPU_THREADS)
        for i in range(CPU_THREADS):
            w = ScaleWorker(
                self.tasks_queue, self.results_queue, data=self.data, volume=self.volume
            )
            w.start()
            self.workers.append(w)
        return self

    def __exit__(self, *args):
        for w in self.workers:
            w.terminate()

    def compute_data(self, batch_tasks):
        tasks_counter = 0

        for batch_task in batch_tasks:
            self.tasks_queue.put(batch_task)
            tasks_counter += 1

        batch_descriptors, batch_values = [], []

        for i in range(tasks_counter):
            descriptors, value = self.results_queue.get()
            batch_descriptors.append(descriptors)
            batch_values.append(value)

        return batch_descriptors, batch_values

    def compute_volume(self, patch_size, scale_levels, tasks):
        tasks_counter = 0
        for y, x in tasks:
            self.tasks_queue.put(((y, x), patch_size, scale_levels))
            tasks_counter += 1

        tasks, descriptors = [], []
        ticks = set((int(i / 20 * tasks_counter) for i in range(20)))
        for i in range(tasks_counter):
            if i in ticks:
                self.__print_progress(i / tasks_counter)
            (y, x), descriptor = self.results_queue.get()
            tasks.append((y, x))
            descriptors.append(descriptor)
        print()

        return tasks, descriptors

    def __print_progress(self, ratio):
        print("\rComputing patches: {} %".format(int(round(ratio * 100))), end="")
        sys.stdout.flush()
