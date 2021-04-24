# coding=utf-8
import hashlib
import io
import logging
import math
import os
import shutil
import subprocess
import sys
from copy import deepcopy
from datetime import datetime
from functools import reduce
from itertools import chain, islice
from typing import Callable, Optional

import numpy
import OpenEXR
from scipy.ndimage.filters import gaussian_filter

from .difference_metrics import get_difference_metric

# make all newly created files and directories to have mask 002
# for files it means: rw-r--r--
# for directories: rwxr-xr-x
os.umask(0o02)


def create_image(
    profile: numpy.ndarray, height: int, width: Optional[int] = None
) -> numpy.ndarray:
    if width is None:
        width = len(profile)

    image = numpy.zeros((height, width), dtype=numpy.float32)
    for row in image:
        row[:] = numpy.array(profile).reshape(row.shape)
    return image


def calculate_profile(image: numpy.ndarray, cut: int = 50) -> (numpy.ndarray, numpy.ndarray):
    try:
        height, width, channels = image.shape
    except ValueError:
        height, width = image.shape
        channels = 1

    if channels > 1:
        raise ValueError("Expected to get a 1-channel image")

    mean_profile = list()
    image_t = image.transpose()
    for row in image_t:
        if cut == 0:
            v = row.mean()
        else:
            v = row[cut:-cut].mean()
        mean_profile.append(v)
    mean_profile = numpy.array(mean_profile)

    image_mean = numpy.zeros(shape=(height, width), dtype=numpy.float32)
    for row in image_mean:
        row[:] = mean_profile

    return image_mean, mean_profile


def filter_hf_noise(hf: numpy.ndarray, guide: numpy.ndarray) -> numpy.ndarray:
    window = 5
    hf_filtered = hf.copy()
    for i, v in enumerate(hf_filtered):
        half_window = int(window / 2)
        local_average = numpy.average(guide[i - half_window : i + half_window])
        window_sum = sum(numpy.abs(hf_filtered[i - half_window : i + half_window]))
        if window_sum < local_average * 0.1:
            hf_filtered[i] = 0.0
    return hf_filtered


def unsharp_mask(image: numpy.ndarray, sigma: float, mult: float = 1.0) -> numpy.ndarray:
    return numpy.clip(image + mult * (image - gaussian_filter(image, sigma)), 0.0, 1.0)


def rms_contrast(image: numpy.ndarray) -> float:
    l_avg = numpy.average(image)
    elements_count = reduce(lambda x, y: x * y, image.shape)
    sum_square_differences = numpy.power(image - l_avg, 2).sum()
    return math.sqrt(sum_square_differences / elements_count)


def sRGB_to_linear(s: numpy.ndarray):
    a = 0.055
    return numpy.where(s <= 0.04045, s / 12.92, ((s + a) / (1 + a)) ** 2.4)


def linear_to_sRGB(s: numpy.ndarray):
    a = 0.055
    return numpy.where(s <= 0.0031308, 12.92 * s, (1 + a) * s ** (1 / 2.4) - a)


def _gamma_to_linear(s: float) -> float:
    return s ** 2.2


def gamma_to_linear(x: numpy.ndarray):
    return numpy.power(x, 2.2)


def _linear_to_gamma(s: float) -> float:
    return s ** (1 / 2.2)


def linear_to_gamma(x: numpy.ndarray):
    return numpy.power(x, 1 / 2.2)


# About the LEF color space: https://infoscience.epfl.ch/record/99814/files/scdialcs_.pdf


def rgb2lef(rgb: numpy.ndarray) -> numpy.ndarray:
    m = numpy.array(
        (
            (2 / 3, 2 / 3, 2 / 3),
            (1, -1 / 2, -1 / 2),
            (0, math.sqrt(3) / 2, -math.sqrt(3) / 2),
        )
    )
    return numpy.matmul(m, rgb)


def lef2rgb(lef: numpy.ndarray) -> numpy.ndarray:
    m = numpy.array(
        (
            (1 / 2, 2 / 3, 0),
            (1 / 2, -1 / 3, 1 / math.sqrt(3)),
            (1 / 2, -1 / 3, -1 / math.sqrt(3)),
        )
    )
    return numpy.matmul(m, lef)


def today_string():
    return datetime.today().date().strftime("%Y-%m-%d")


def ensure_dir(dirpath, clean: bool = False):
    if clean and os.path.exists(dirpath):
        shutil.rmtree(dirpath)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath)
    return dirpath


def dump_image(image: numpy.ndarray, filepath: str, filename: str = None) -> None:
    import cv2

    if filename:
        filepath = os.path.join(filepath, filename)
    ensure_dir(os.path.dirname(filepath))

    if len(image.shape) == 2 and image.shape[0] == 1 or len(image.shape) == 1:
        image = create_image(image, 200)

    if filepath.endswith("png"):
        image = linear_to_sRGB(image)
        image = image.clip(0.0, 1.0)
        image = (image * 255 + 0.5).astype(numpy.uint8)
    elif filepath.endswith("exr"):
        image = image.astype(numpy.float32)

        if len(image.shape) == 2:
            image = numpy.tile(image[..., numpy.newaxis], (1, 1, 3))
        w, h, d = image.shape
        assert d == 3 or d == 4

        # get the channels
        red = numpy.array(image[:, :, 0]).data
        green = numpy.array(image[:, :, 1]).data
        blue = numpy.array(image[:, :, 2]).data
        if d == 4:
            alpha = numpy.array(image[:, :, 3]).data

        # Write the three color channels to the output file
        out = OpenEXR.OutputFile(filepath, OpenEXR.Header(h, w))
        dict = {"R": red, "G": green, "B": blue}
        if d == 4:
            dict["A"] = str(alpha)
        out.writePixels(dict)
        out.close()
        del out
        return

    if len(image.shape) == 3 and image.shape[2] == 3:
        cv2.imwrite(filepath, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    else:
        cv2.imwrite(filepath, image)


def dump_array(arr: numpy.ndarray, filepath: str, filename: str = None) -> None:
    if filename:
        filepath = os.path.join(filepath, filename)
    ensure_dir(os.path.dirname(filepath))
    numpy.savez_compressed(filepath, data=arr)


def load_discrete_material_voxels(file):
    if isinstance(file, str):
        data = numpy.load(file, allow_pickle=True)
    elif isinstance(file, numpy.lib.npyio.NpzFile) or isinstance(file, dict):
        data = file
    else:
        raise ValueError("File should be either string or NpzFile")
    try:
        voxels = data["halftoned_voxels"]
        try:
            labels = set([(k, v["name"]) for k, v in data["labels"].item().items()])
        except TypeError:
            labels = set(data["labels"].item().items())
        rendered_image = data["rendered_image"]
    except KeyError:
        raise ValueError(
            "Data should contain fields: halftoned_voxels, " "labels, rendered_image"
        )
    return voxels, labels, rendered_image


def dump_discrete_material_voxels(
    voxels: numpy.ndarray,
    labels,
    materials_dict: Optional[dict],
    rendered_image: Optional[numpy.ndarray],
    filepath: str,
    filename: str = None,
) -> None:
    if filename:
        filepath = os.path.join(filepath, filename)
    ensure_dir(os.path.dirname(filepath))
    numpy.savez_compressed(
        filepath,
        halftoned_voxels=voxels,
        labels=dict(labels),
        materials=materials_dict,
        rendered_image=rendered_image,
    )


def image_has_one_channel(image: numpy.ndarray) -> bool:
    return len(image.shape) == 2 or image.shape[2] == 1


def read_image(
    path_to_image,
    convert_to_grayscale: bool = False,
    alpha: bool = False,
    clip_0_1: bool = False,
):
    import cv2

    path_to_image = os.path.normpath(path_to_image)
    if not os.path.exists(path_to_image):
        raise ValueError("{} does not exist".format(path_to_image))

    if path_to_image.endswith("png"):
        image = cv2.imread(path_to_image, cv2.IMREAD_UNCHANGED if alpha else cv2.IMREAD_COLOR)
    elif path_to_image.endswith("exr"):
        image = cv2.imread(path_to_image, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if clip_0_1 and (image.min() < 0.0 or image.max() > 1.0):
            print("Clipping input image to 0..1 range")
            image = numpy.clip(image, 0.0, 1.0)
    else:
        raise ValueError("Only .png and .exr images are supported")

    if image is None:
        raise ValueError("Failed to read the image")

    if path_to_image.endswith("png"):
        image = image.astype(numpy.float32) / 255.0
        image = sRGB_to_linear(image)

    if len(image.shape) == 2:
        image.reshape(image.shape + (1,))

    if len(image.shape) == 3:
        height, width, channels = image.shape
    elif len(image.shape) == 2:
        height, width = image.shape
        channels = 1
    else:
        raise ValueError("Image shape should have 2 or 3 integers")

    if channels >= 3:
        if convert_to_grayscale:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            image.reshape(image.shape + (1,))
        else:
            image = cv2.cvtColor(image, cv2.COLOR_BGRA2RGBA if alpha else cv2.COLOR_BGR2RGB)
    elif channels == 1:
        pass
    else:
        raise ValueError("Support only image with 1 or 3+ channels")

    return image


def convert_r_g_b_to_rgb(r: numpy.ndarray, g: numpy.ndarray, b: numpy.ndarray) -> numpy.ndarray:
    assert r.shape == g.shape == b.shape, "R, G, B have different dimensions"
    assert r.dtype == g.dtype == b.dtype, "R, G, B have different data types"
    assert image_has_one_channel(r), "Number of channels in R is not 1"
    assert image_has_one_channel(g), "Number of channels in G is not 1"
    assert image_has_one_channel(b), "Number of channels in B is not 1"

    height, width = r.shape[0], r.shape[1]
    rgb = numpy.zeros((height, width, 3), r.dtype)
    rgb[..., 0] = r
    rgb[..., 1] = g
    rgb[..., 2] = b
    return rgb


def numpy2vdb(grid, ijk, volume):
    volume = numpy.ascontiguousarray(volume.swapaxes(0, 2)).astype(numpy.float64)
    grid.copyFromArray(volume, ijk)


def vdb2numpy(grid):
    rgb_volume = numpy.empty(shape=grid.evalActiveVoxelDim() + (3,), dtype=numpy.float64)
    bbox_min, _ = grid.evalActiveVoxelBoundingBox()
    grid.copyToArray(rgb_volume, ijk=bbox_min)
    return numpy.ascontiguousarray(rgb_volume.swapaxes(0, 2)).astype(numpy.float32)


def inch_to_m(v):
    return v * 0.0254


def inch_to_mm(v):
    return inch_to_m(v) * 1000


def mm_to_inch(v):
    return v / 25.4


def debug_mode_enabled():
    return bool(int(os.getenv("DEBUG", 0)))


# TODO: use a logging system with levels (DEBUG-INFO-WARNING-ERROR)
def mitsuba_debug_mode_enabled():
    return bool(int(os.getenv("DEBUG_MITSUBA", 0)))


def resolve_project_path(subpath: str):
    sources_path = os.path.dirname(os.path.realpath(__file__))

    return os.path.abspath(os.path.join(sources_path, "..", "..", subpath))


def getenvcast(varname: str, default, type_: Callable):
    value = os.getenv(varname)
    if value is not None:
        return type_(value)
    else:
        return default


class Capturing:
    def __init__(self, f):
        self._file = f

    def __enter__(self):
        self._stderr = sys.stderr
        self._stdout = sys.stdout
        sys.stderr = self._file
        sys.stdout = self._file
        return self

    def __exit__(self, *args):
        sys.stderr = self._stderr
        sys.stdout = self._stdout


class TempFiles:
    def __init__(self, temp_path, filename_prefix):
        self.created_files = set()
        self.temp_path = os.path.abspath(temp_path)
        self.filename_prefix = filename_prefix

    def __enter__(self):
        return self

    def __exit__(self, *args):
        for filename in list(self.created_files):
            try:
                os.remove(filename)
                self.created_files.remove(filename)
            except (FileNotFoundError, OSError) as e:
                print("Failed to delete file {}. {}".format(filename, e))

    def add(self, path):
        self.created_files.add(path)

    def get_path(self, filename):
        return os.path.join(self.temp_path, self.filename_prefix + filename)

    def remove(self, path):
        self.created_files.remove(path)


class FileOrStdOutWriter:
    """
    This context manager provides a file-like object:
    either an opened file (if file_path is given) or stdout.
    """

    def __init__(self, file_path: Optional[str] = None):
        self.file_path = file_path

    def __enter__(self):
        if self.file_path:
            self.writer = open(self.file_path, "w", 1)
        else:
            self.writer = sys.stdout
        return self.writer

    def __exit__(self, *args):
        if self.file_path:
            self.writer.close()


def chunks(iterable, size=10):
    iterator = iter(iterable)
    for first in iterator:
        yield chain([first], islice(iterator, size - 1))


def get_git_revision_short_hash():
    return str(
        subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=resolve_project_path("")
        ),
        "utf-8",
    ).strip()


suffixes = ["B", "KB", "MB", "GB", "TB", "PB"]


def human_size(nbytes):
    i = 0
    while nbytes >= 1024 and i < len(suffixes) - 1:
        nbytes /= 1024.0
        i += 1
    f = ("%.2f" % nbytes).rstrip("0").rstrip(".")
    return "%s %s" % (f, suffixes[i])


def setup_console_logger(name, verbose=False):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.handlers = []
    logger.addHandler(ch)

    return logger


def md5sum(fd, seek_begin: bool = True):
    if seek_begin:
        fd.seek(0)
    md5 = hashlib.md5()
    content = fd.read()
    md5.update(content)
    if seek_begin:
        fd.seek(0)
    return md5.hexdigest()


def md5sum_path(path):
    with io.open(path, mode="rb") as fd:
        return md5sum(fd)


def clean_timings_dict(timings_dict):
    for k, v in timings_dict.items():
        if isinstance(v, float):
            timings_dict[k] = 0.0
        elif isinstance(v, dict):
            clean_timings_dict(v)
        else:
            raise ValueError("Values of a timing dict are expected to be either floats or dicts")


def update_timings_dict(base, new):
    for k, v in new.items():
        if k in base:
            if type(base[k]) != type(v):
                raise ValueError(
                    "base and new timings dict have different types "
                    "({} and {}) for key {}".format(type(base[k]), type(v), k)
                )
            if isinstance(v, float):
                base[k] += v
            elif isinstance(v, dict):
                update_timings_dict(base[k], v)
            else:
                raise ValueError(
                    "Values of a timing dict are expected " "to be either floats or dicts"
                )
        else:
            base[k] = deepcopy(v)


def log_timing(timings_dict, key, time_value):
    try:
        timings_dict[key] += time_value
    except KeyError:
        timings_dict[key] = time_value
