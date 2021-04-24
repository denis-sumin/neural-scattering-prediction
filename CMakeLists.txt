cmake_minimum_required(VERSION 3.9)
project(FabricationEnhancement)

set(CMAKE_CXX_STANDARD 17)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/data/cmake")

find_package(OpenMP)
if (OpenMP_FOUND)
    include_directories(${OpenMP_C_INCLUDE_DIRS})
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(WARNING "OpenMP is not supported by the compiler")
endif()

find_package(CUDA)
find_package(Eigen3 3.0 REQUIRED)
find_package(OpenCV)
find_package(IlmBase REQUIRED)
find_package(OpenVDB)
find_package(PythonLibs 3.7)
find_package(ZLIB REQUIRED)

include_directories(
    ${EIGEN3_INCLUDE_DIR}
    ${OPENVDB_INCLUDE_DIR}
    ${PYTHON_INCLUDE_DIRS}
    ${ZLIB_INCLUDE_DIRS}
)
link_directories(${IlmBase_LIBRARY_DIRS})

add_subdirectory(pybind11)

pybind11_add_module(sat_tile_tree sources/nn_rendering_prediction/tree/sat_tile_tree.cpp)
target_link_libraries(sat_tile_tree PRIVATE ${OpenMP_C_LIBRARIES})
