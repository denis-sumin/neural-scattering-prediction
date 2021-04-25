import os
from distutils.core import Extension, setup
from sys import platform
from typing import List

import numpy as np

EIGEN_INCLUDE_DIR = os.path.expanduser(os.getenv("EIGEN_INCLUDE_DIR", "/usr/include/eigen3/"))


def openmp_compile_args(other_compile_args: List[str]) -> List[str]:
    if platform == "darwin":
        return other_compile_args + ["-Xpreprocessor", "-fopenmp"]
    else:
        return other_compile_args + ["-fopenmp"]


def openmp_link_args(other_link_args: List[str]) -> List[str]:
    if platform == "darwin":
        return other_link_args + ["-lomp"]
    else:
        return other_link_args + ["-lgomp"]


ext_downscale = Extension(
    "downscale",
    sources=["fabnn/downscale.cpp"],
    include_dirs=[
        np.get_include(),
        "third-party/opencv",
    ],
    extra_compile_args=["-std=c++11"],
    # extra_link_args=['-lopencv_core'],
)

ext_sat_tile_tree = Extension(
    "sat_tile_tree",
    sources=[
        "fabnn/tree/sat_tile_tree.cpp",
    ],
    include_dirs=[
        np.get_include(),
        "pybind11/include",
        EIGEN_INCLUDE_DIR,
    ],
    extra_compile_args=openmp_compile_args(["-std=c++17"]),
    extra_link_args=openmp_link_args([]),
)

ext_grid_converter = Extension(
    "grid_converter",
    sources=[
        "fabnn/prepare_training_data/grid_converter.cpp",
    ],
    include_dirs=[
        np.get_include(),
        "pybind11/include",
        EIGEN_INCLUDE_DIR,
    ],
    extra_compile_args=openmp_compile_args(["-std=c++17"]),
    extra_link_args=openmp_link_args(["-lopenvdb", "-lHalf", "-ltbb"]),
)

setup(
    name="fabnn",
    ext_modules=[
        ext_downscale,
        ext_sat_tile_tree,
        ext_grid_converter,
    ],
)
