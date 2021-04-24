#include <algorithm>
#include <iostream>
#include <stdexcept>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION  // Disable old Numpy API
#include "numpy/arrayobject.h"
#include <opencv2/core/core.hpp>
#include <Python.h>


/* Errors */
static PyObject *DownscaleError;


static PyObject* downscale_local_mean(PyObject* self, PyObject* args) {
    PyArrayObject *in_arr = NULL, *factors = NULL, *out_arr = NULL;

    /* Parse arguments from Python:
       "O!" - check type of object
       &in_arr  - to variable in_arr
     */
    if (!PyArg_ParseTuple(args, "O!O!",
                          &PyArray_Type, &in_arr, &PyArray_Type, &factors))
        return nullptr;

//    if (PyArray_NDIM(values) != 1) {
//        PyErr_SetString(DitherError, "Values array must have 1 dimension");
//        return NULL;
//    }

    auto input_dims = PyArray_NDIM(in_arr);

    if (input_dims != 4) {
        std::cout << input_dims << std::endl;
        PyErr_SetString(DownscaleError, "Input array should have 4 dimensions");
        return nullptr;
    }

    auto factors_dims = PyArray_NDIM(factors);
    auto factors_size = PyArray_SHAPE(factors)[0];

    if (factors_dims != 1 || factors_size != 4) {
        PyErr_SetString(
                DownscaleError,
                "Factors array should have 1 dimension and 4 values in it");
        return nullptr;
    }

    auto input_shape = PyArray_SHAPE(in_arr);
    auto *factor_values =
            reinterpret_cast<unsigned int *>(PyArray_DATA(factors));

    if (factor_values[3] != 1) {
        PyErr_SetString(
                DownscaleError,
                "Factor value for the channels should equal 1");
        return nullptr;
    }

    for (int i = 0; i < 4; ++i) {
        if (input_shape[i] % factor_values[i]) {
            PyErr_SetString(
                    DownscaleError, "Input shape cannot be scaled precisely "
                            "with the factors given (check shapes)");
            return nullptr;
        }
    }

    npy_intp out_shape[4]{
            input_shape[0] / factor_values[0],
            input_shape[1] / factor_values[1],
            input_shape[2] / factor_values[2],
            input_shape[3] / factor_values[3]
    };
    out_arr = reinterpret_cast<PyArrayObject *>(
            PyArray_ZEROS(4, out_shape, NPY_FLOAT32, 0));

    int counter;
    double sum;

    for (int z = 0; z < out_shape[0]; ++z) {
        for (int y = 0; y < out_shape[1]; ++y) {
            for (int x = 0; x < out_shape[2]; ++x) {
                for (int c = 0; c < out_shape[3]; ++c) {
                    counter = 0;
                    sum = 0;
                    for (unsigned int i = 0; i < factor_values[0]; ++i) {
                        for (unsigned int j = 0; j < factor_values[1]; ++j) {
                            for (unsigned int k = 0; k < factor_values[2]; ++k) {
                                auto value = *reinterpret_cast<float *>(
                                        PyArray_GETPTR4(
                                                in_arr,
                                                factor_values[0] * z + i,
                                                factor_values[1] * y + j,
                                                factor_values[2] * x + k, c));
                                sum += value;
                                counter += 1;
                            }
                        }
                    }
                    *reinterpret_cast<float *>(
                            PyArray_GETPTR4(
                                    out_arr, z, y, x, c)) =
                            static_cast<float>(sum / counter);
                }
            }
        }
    }

    /* Return new array without increase reference count:
     * O - increase reference count
     * N - not increase reference count
     */
    return Py_BuildValue("N", out_arr);
}


inline double get_value(PyArrayObject *v, unsigned int z, unsigned int y, unsigned int x, unsigned int c) {
    return *reinterpret_cast<double *>(PyArray_GETPTR4(v, z, y, x, c));
}


inline double get_sum_from_sat(
        PyArrayObject *v,
        unsigned int z1, unsigned int z2,
        unsigned int y1, unsigned int y2,
        unsigned int x1, unsigned int x2, unsigned int c) {
    return (
              get_value(v, z2, y2, x2, c) - get_value(v, z2, y2, x1, c)
            - get_value(v, z2, y1, x2, c) - get_value(v, z1, y2, x2, c)
            + get_value(v, z1, y1, x2, c) + get_value(v, z1, y2, x1, c)
            + get_value(v, z2, y1, x1, c) - get_value(v, z1, y1, x1, c)
    );
}


inline void fill_one_scale(
        PyArrayObject *volume_sat, npy_intp *volume_sat_shape,
        float *out_arr, bool allow_padding,
        int patch_size_z, int patch_size_y, int patch_size_x,
        int scale_kernel_z, int scale_kernel_y, int scale_kernel_x,
        int channels, int coord_x, int coord_y
        ) {
    int patch_start_x = coord_x - patch_size_x * scale_kernel_x / 2;
    int patch_start_y = coord_y - patch_size_y * scale_kernel_y / 2;
    int patch_start_z = 0;

    int patch_area = scale_kernel_z * scale_kernel_y * scale_kernel_x;
    int z1, z2, y1, y2, x1, x2;
    bool border_violation = false;

    for (int z = 0; z < patch_size_z; ++z) {
        z1 = patch_start_z + z * scale_kernel_z;
        z2 = z1 + scale_kernel_z;

        if (z1 < 0) {
            border_violation = !allow_padding;
            z1 = 0;
            if (z2 < 0) {
                z2 = 0;
            }
        }
        if (z2 >= volume_sat_shape[0]) {
            border_violation = !allow_padding;
            z2 = volume_sat_shape[0] - 1;
            if (z1 >= volume_sat_shape[0]) {
                z1 = volume_sat_shape[0] - 1;
            }
        }

        for (int y = 0; y < patch_size_y; ++y) {
            y1 = patch_start_y + y * scale_kernel_y;
            y2 = y1 + scale_kernel_y;

            if (y1 < 0) {
                border_violation = !allow_padding;
                y1 = 0;
                if (y2 < 0) {
                    y2 = 0;
                }
            }
            if (y2 >= volume_sat_shape[1]) {
                border_violation = !allow_padding;
                y2 = volume_sat_shape[1] - 1;
                if (y1 >= volume_sat_shape[1]) {
                    y1 = volume_sat_shape[1] - 1;
                }
            }

            for (int x = 0; x < patch_size_x; ++x) {
                x1 = patch_start_x + x * scale_kernel_x;
                x2 = x1 + scale_kernel_x;

                if (x1 < 0) {
                    border_violation = !allow_padding;
                    x1 = 0;
                    if (x2 < 0) {
                        x2 = 0;
                    }
                }
                if (x2 >= volume_sat_shape[2]) {
                    border_violation = !allow_padding;
                    x2 = volume_sat_shape[2] - 1;
                    if (x1 >= volume_sat_shape[2]) {
                        x1 = volume_sat_shape[2] - 1;
                    }
                }

                for (int c = 0; c < channels; ++c) {
                    auto sum = get_sum_from_sat(
                            volume_sat,
                            static_cast<unsigned int>(z1),
                            static_cast<unsigned int>(z2),
                            static_cast<unsigned int>(y1),
                            static_cast<unsigned int>(y2),
                            static_cast<unsigned int>(x1),
                            static_cast<unsigned int>(x2), c);
                    size_t arr_idx = (
                            z * patch_size_y * patch_size_x * channels +
                            y * patch_size_x * channels +
                            x * channels +
                            c);
                    out_arr[arr_idx] = static_cast<float>(sum / patch_area);
                }
            }
        }
    }

    if (border_violation) {
        int patch_end_x = coord_x + patch_size_x * scale_kernel_x / 2;
        int patch_end_y = coord_y + patch_size_y * scale_kernel_y / 2;
        int patch_end_z = patch_size_z * scale_kernel_z;

        std::cerr << "patch_start_{x,y,z} " << patch_start_x << " " << patch_start_y << " " << patch_start_z << " " << std::endl;
        std::cerr << "patch_end_{x,y,z} " << patch_end_x << " " << patch_end_y << " " << patch_end_z << " " << std::endl;
        std::cerr << "volume_sat_shape_{x,y,z} " << volume_sat_shape[2] << " " << volume_sat_shape[1] << " " << volume_sat_shape[0] << " " << std::endl;
        throw std::invalid_argument("The volume_sat dimensions are not sufficient to compute the patch, padding is not allowed");
    }
}


static PyObject* downscale_local_mean_sat(PyObject* self, PyObject* args) {
    PyArrayObject *volume_sat = NULL, *out_arr = NULL;
    int coord_x, coord_y;
    int patch_size_z, patch_size_y, patch_size_x;
    int scale_kernel_z, scale_kernel_y, scale_kernel_x, scale_kernel_c;
    int allow_padding = 0;

    /* Parse arguments from Python:
       "O!" - check type of object
       &in_arr  - to variable in_arr
     */
    if (!PyArg_ParseTuple(args, "O!ii(iii)(iiii)|i",
                          &PyArray_Type, &volume_sat,
                          &coord_x, &coord_y,
                          &patch_size_z, &patch_size_y, &patch_size_x,
                          &scale_kernel_z, &scale_kernel_y, &scale_kernel_x, &scale_kernel_c,
                          &allow_padding
    )) {
        return nullptr;
    }

    auto volume_sat_dims = PyArray_NDIM(volume_sat);
    auto volume_sat_shape = PyArray_SHAPE(volume_sat);

    if (volume_sat_dims != 4) {
        std::cout << volume_sat_dims << std::endl;
        PyErr_SetString(DownscaleError, "volume_sat array should have 4 dimensions");
        return nullptr;
    }

    npy_intp out_shape[4]{patch_size_z, patch_size_y, patch_size_x, volume_sat_shape[3]};
    out_arr = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(4, out_shape, NPY_FLOAT32, 0));

    auto out_arr_scale = reinterpret_cast<float *>(PyArray_GETPTR1(out_arr, 0));

    try {
        fill_one_scale(volume_sat, volume_sat_shape, out_arr_scale,
                       static_cast<bool>(allow_padding),
                       patch_size_z, patch_size_y, patch_size_x,
                       scale_kernel_z, scale_kernel_y, scale_kernel_x,
                       volume_sat_shape[3], coord_x, coord_y);
    } catch( const std::invalid_argument& e ) {
        PyErr_SetString(DownscaleError, e.what());
        return nullptr;
    }

    /* Return new array without increase reference count:
     * O - increase reference count
     * N - not increase reference count
     */
    return Py_BuildValue("N", out_arr);
}


static PyObject* generate_scales_sat(PyObject* self, PyObject* args) {
    PyArrayObject *volume_sat = NULL, *scale_kernels = NULL, *out_arr = NULL;
    int coord_x, coord_y;
    int patch_size_z, patch_size_y, patch_size_x;
    int allow_padding = 0;

    /* Parse arguments from Python:
       "O!" - check type of object
       &in_arr  - to variable in_arr
     */
    if (!PyArg_ParseTuple(args, "O!(iii)O!ii|i",
                          &PyArray_Type, &scale_kernels,
                          &patch_size_z, &patch_size_y, &patch_size_x,
                          &PyArray_Type, &volume_sat,
                          &coord_y, &coord_x,
                          &allow_padding
    )) {
        return nullptr;
    }

    auto scale_kernels_dims = PyArray_NDIM(scale_kernels);
    auto scale_kernels_shape = PyArray_SHAPE(scale_kernels);

    if (scale_kernels_dims != 2) {
        std::cout << scale_kernels_dims << std::endl;
        PyErr_SetString(DownscaleError, "scale_kernels array should have 2 dimensions");
        return nullptr;
    }

    auto volume_sat_dims = PyArray_NDIM(volume_sat);
    auto volume_sat_shape = PyArray_SHAPE(volume_sat);

    if (volume_sat_dims != 4) {
        std::cout << volume_sat_dims << std::endl;
        PyErr_SetString(DownscaleError, "volume_sat array should have 4 dimensions");
        return nullptr;
    }
    unsigned short scale_kernel_x, scale_kernel_y, scale_kernel_z;

    npy_intp out_shape[5] {scale_kernels_shape[0], patch_size_z, patch_size_y, patch_size_x, volume_sat_shape[3]};
    out_arr = reinterpret_cast<PyArrayObject *>(PyArray_EMPTY(5, out_shape, NPY_FLOAT32, 0));

    Py_BEGIN_ALLOW_THREADS
    for (int i = 0; i < scale_kernels_shape[0]; ++i) {
        scale_kernel_z = *reinterpret_cast<unsigned short *>(PyArray_GETPTR2(scale_kernels, i, 0));
        scale_kernel_y = *reinterpret_cast<unsigned short *>(PyArray_GETPTR2(scale_kernels, i, 1));
        scale_kernel_x = *reinterpret_cast<unsigned short *>(PyArray_GETPTR2(scale_kernels, i, 2));
        auto out_arr_scale = reinterpret_cast<float *>(PyArray_GETPTR1(out_arr, i));
        try {
            fill_one_scale(volume_sat, volume_sat_shape, out_arr_scale,
                           static_cast<bool>(allow_padding),
                           patch_size_z, patch_size_y, patch_size_x,
                           scale_kernel_z, scale_kernel_y, scale_kernel_x,
                           static_cast<int>(volume_sat_shape[3]), coord_x, coord_y);
        } catch( const std::invalid_argument& e ) {
            PyErr_SetString(DownscaleError, e.what());
            return nullptr;
        }
    }
    Py_END_ALLOW_THREADS

    /* Return new array without increase reference count:
     * O - increase reference count
     * N - not increase reference count
     */
    return Py_BuildValue("N", out_arr);
}


/* Array with methods
 */
static PyMethodDef module_methods[] = {
        /* name from python, name in C-file, ..., __doc__ string of method */
        {"downscale_local_mean", downscale_local_mean, METH_VARARGS,
                "Perform array downscaling"},
        {"downscale_local_mean_sat", downscale_local_mean_sat, METH_VARARGS,
                "Perform array downscaling from SAT volume"},
        {"generate_scales_sat", generate_scales_sat, METH_VARARGS,
                "Perform array downscaling from SAT volume with given scales"},
        {NULL, NULL, 0, NULL}
};

/* Array with info about module to create it in Python
 */
static struct PyModuleDef moduledef =
        {
                PyModuleDef_HEAD_INIT,
                "downscale",  // name of module
                "Perform array downscaling",  // module documentation, may be NULL
                -1,  /* size of per-interpreter state of the module,
            or -1 if the module keeps state in global variables. */
                module_methods
        };

/* Init our module in Python
 */
PyMODINIT_FUNC PyInit_downscale(void)
{
    PyObject *m;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    /* Import NUMPY settings
     */
    import_array();

    /* Init errors */
    DownscaleError = PyErr_NewException("downscale.error", NULL, NULL);
    Py_INCREF(DownscaleError); /* Increment reference count for object */
    PyModule_AddObject(m, "error", DownscaleError);

    return m;
}

