import math
import pickle
import sys
from functools import partial
from multiprocessing import Process, Queue, cpu_count
from time import time

import numpy
from colorspaces.color_cube import build_color_slices_image
from materials import populate_materials
from scipy.spatial import KDTree
from surface_albedo_mapping_ import apply_albedo_mapping_analytical_forward
from utils import gamma_to_linear, linear_to_gamma


def fm_mixture(materials, material_labels, x):

    sc = numpy.zeros(shape=(3,), dtype=numpy.float32)
    bc = numpy.zeros(shape=(3,), dtype=numpy.float32)

    for i, name in sorted(material_labels):
        s = materials[name].albedo * materials[name].density
        b = materials[name].density - s
        sc += x[i] * s
        bc += x[i] * b

    return apply_albedo_mapping_analytical_forward(sc / (sc + bc))


class Worker_cmykw_to_rgb(Process):
    def __init__(self, tasks, results, materials, material_labels):
        self.tasks = tasks
        self.results = results
        self.forward_mixture = partial(fm_mixture, materials, material_labels)
        Process.__init__(self)

    def run(self):
        while True:
            cmykw = self.tasks.get()
            rgb = self.forward_mixture(numpy.array(cmykw, dtype=numpy.float32))
            self.results.put((cmykw, rgb))


def zero_one_gen(x):
    return (i / (x - 1) for i in range(x))


def zero_one_gen_pow2(x):
    return ((i / (x - 1)) ** 2 for i in range(x))


def gen_cmykw_to_rgb_table(
    steps,
    materials,
    material_labels,
    min_fraction=None,
    c_fractions_gen=zero_one_gen_pow2,
    m_fractions_gen=zero_one_gen_pow2,
    y_fractions_gen=zero_one_gen_pow2,
    k_fractions_gen=zero_one_gen_pow2,
):
    round_cmykw = round(math.log10(steps))
    eps = 0.00001

    tasks_queue, results_queue = Queue(), Queue()

    num_threads = cpu_count()
    print("Threads", num_threads)
    workers = list()
    for i in range(num_threads):
        w = Worker_cmykw_to_rgb(tasks_queue, results_queue, materials, material_labels)
        w.start()
        workers.append(w)

    counter = 0
    for c in c_fractions_gen(steps):
        if min_fraction and c != 0.0 and c < min_fraction:
            continue
        for m in m_fractions_gen(steps):
            if min_fraction and m != 0.0 and m < min_fraction:
                continue
            for y in y_fractions_gen(steps):
                if min_fraction and y != 0.0 and y < min_fraction:
                    continue
                for k in k_fractions_gen(steps):
                    if min_fraction and k != 0.0 and k < min_fraction:
                        continue
                    if (c + m + y + k) > 1.0:
                        continue
                    elif abs((c + m + y + k) - 1.0) < eps:
                        w = 0.0
                    else:
                        w = round(1.0 - c - m - y - k, round_cmykw)
                    if min_fraction and w != 0.0 and w < min_fraction:
                        continue
                    tasks_queue.put((c, m, y, k, w))
                    counter += 1
                    if not counter % (steps * steps * steps):
                        print("Sent {} to queue".format(counter))

    print("Tasks: ", counter)
    tasks_number = counter

    table = dict()
    while counter > 0:
        cmykw, rgb = results_queue.get()
        table[cmykw] = rgb
        counter -= 1
        if not counter % (steps * steps * steps):
            print(
                "Received and processed results {} %".format(
                    round((tasks_number - counter) / tasks_number * 100.0)
                )
            )

    for w in workers:
        w.terminate()

    return table


class Worker_nearest_rgb(Process):
    def __init__(self, tasks, results, tree, cmykw_list):
        self.tasks = tasks
        self.results = results
        self.tree = tree
        self.cmykw_list = cmykw_list
        Process.__init__(self)

    def run(self):
        while True:
            rgb = self.tasks.get()
            dist, idx = self.tree.query(linear_to_gamma(numpy.array(rgb, dtype=numpy.float32)))
            self.results.put((rgb, self.cmykw_list[idx]))


def build_rgb_to_cmykw_table(cmykw_to_rgb_table, rgb_step, kd_tree):
    cmykw_list = list(cmykw_to_rgb_table.keys())
    rgb_table = dict()

    tasks_queue, results_queue = Queue(), Queue()
    num_threads = cpu_count()
    print("Threads", num_threads)
    workers = list()
    for i in range(num_threads):
        w = Worker_nearest_rgb(tasks_queue, results_queue, kd_tree, cmykw_list)
        w.start()
        workers.append(w)

    counter = 0

    for r in zero_one_gen(rgb_step):
        for g in zero_one_gen(rgb_step):
            for b in zero_one_gen(rgb_step):
                tasks_queue.put((r, g, b))
                counter += 1
                if not counter % (rgb_step * rgb_step):
                    print("Sent {} to queue".format(counter))

    print("Tasks: ", counter)
    tasks_number = counter

    while counter > 0:
        rgb, cmykw = results_queue.get()
        rgb_table[rgb] = cmykw
        counter -= 1
        if not counter % (rgb_step * rgb_step):
            print(
                "Received and processed results {} %".format(
                    round((tasks_number - counter) / tasks_number * 100.0)
                )
            )

    for w in workers:
        w.terminate()

    return rgb_table


def main():
    materials_file = sys.argv[1]
    cmykw_to_rgb_file = sys.argv[2]
    cmykw_bins_per_axis = int(sys.argv[3])
    rgb_to_cmykw_file = sys.argv[4]
    rgb_bins_per_axis = int(sys.argv[5])

    g = 0.4
    materials = populate_materials(materials_file)
    for name, m in materials.items():
        assert tuple(m.anisotropy) == (g,) * 3
    materials_order = ("cyan", "magenta", "yellow", "black", "white")
    material_labels = tuple(zip(range(len(materials_order)), materials_order))

    cmykw_to_rgb_table = gen_cmykw_to_rgb_table(cmykw_bins_per_axis, materials, material_labels)
    # build_rgb_to_cmykw_table(table, 11)
    with open(cmykw_to_rgb_file, "wb") as f:
        pickle.dump(cmykw_to_rgb_table, f, pickle.HIGHEST_PROTOCOL)
    # exit()
    # print('Loading cmykw->rgb table')
    # with open(filename, 'rb') as f:
    #     cmykw_to_rgb_table = pickle.load(f)

    print("Build kd-tree")
    rgbs = list(cmykw_to_rgb_table.values())
    kd_tree = KDTree(linear_to_gamma(rgbs))

    print("Building rgb->cmykw table")
    rgb_table = build_rgb_to_cmykw_table(cmykw_to_rgb_table, rgb_bins_per_axis, kd_tree)
    with open(rgb_to_cmykw_file, "wb") as f:
        pickle.dump(rgb_table, f, pickle.HIGHEST_PROTOCOL)

    exit()

    image = gamma_to_linear(build_color_slices_image(25, 8))
    preview1 = numpy.empty(shape=image.shape, dtype=numpy.float32)
    preview2 = numpy.empty(shape=image.shape, dtype=numpy.float32)

    s = time()
    cache = dict()
    print("Build image 1")
    for i, row in enumerate(image):
        for j, p in enumerate(row):
            try:
                rgb_value = cache[tuple(p)]
            except KeyError:
                dist, idx = kd_tree.query(linear_to_gamma(p))
                rgb_value = rgbs[idx]
                cache[tuple(p)] = rgb_value
            preview1[i, j] = rgb_value
    print(time() - s)

    s = time()
    cache = dict()
    print("Build image 2")
    for i, row in enumerate(image):
        for j, p in enumerate(row):
            try:
                rgb_value = cache[tuple(p)]
            except KeyError:
                key = tuple(
                    (
                        float(round(item * (rgb_bins_per_axis - 1)) / (rgb_bins_per_axis - 1))
                        for item in p
                    )
                )
                cmykw = rgb_table[key]
                rgb_value = fm_mixture(materials, material_labels, cmykw)
                cache[tuple(p)] = rgb_value
            preview2[i, j] = rgb_value
    print(time() - s)

    import cv2

    cv2.imshow("Image", linear_to_gamma(cv2.cvtColor(image, cv2.COLOR_RGB2BGR)))
    cv2.imshow(
        "Search in cmykw table",
        linear_to_gamma(cv2.cvtColor(preview1, cv2.COLOR_RGB2BGR)),
    )
    cv2.imshow(
        "Search in rgb table",
        linear_to_gamma(cv2.cvtColor(preview2, cv2.COLOR_RGB2BGR)),
    )
    cv2.waitKey()


if __name__ == "__main__":
    main()
