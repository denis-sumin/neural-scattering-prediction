import numpy
from skimage.color import lab2lch, rgb2lab, xyz2lab
from skimage.color.colorconv import (
    _convert,
    _prepare_colorarray,
    get_xyz_coords,
    rgb_from_xyz,
    xyz_from_rgb,
)


def _gaussian(wavelegth, alpha, mu, sigma_1, sigma_2):
    sigma = numpy.empty(wavelegth.shape)
    mask = wavelegth < mu
    sigma[mask] = sigma_1
    sigma[numpy.logical_not(mask)] = sigma_2
    return alpha * numpy.exp((wavelegth - mu) ** 2 / (-2 * sigma ** 2))


def CIEXYZ_primaries(wavelengths: numpy.ndarray):
    assert (
        1000 < numpy.min(wavelengths) and numpy.max(wavelengths) < 10000
    ), "wavelengths must be in angstrom"

    X = (
        _gaussian(wavelengths, 1.056, 5998, 379, 310)
        + _gaussian(wavelengths, 0.362, 4420, 160, 267)
        + _gaussian(wavelengths, -0.065, 5011, 204, 262)
    )
    Y = _gaussian(wavelengths, 0.821, 5688, 469, 405) + _gaussian(
        wavelengths, 0.286, 5309, 163, 311
    )
    Z = _gaussian(wavelengths, 1.217, 4370, 118, 360) + _gaussian(
        wavelengths, 0.681, 4590, 260, 138
    )

    normalization = numpy.sum(Y)

    return X / normalization, Y / normalization, Z / normalization


def xyz2linear_rgb(rgb):
    """
    Copy of skimage.color.colorconv.xyz2rgb without sRGB to linear RGB conversion
    """
    arr = _prepare_colorarray(rgb).copy()
    return _convert(rgb_from_xyz, arr)


def linear_rgb2xyz(rgb):
    """
    Copy of skimage.color.colorconv.rgb2xyz without sRGB to linear RGB conversion
    """
    arr = _prepare_colorarray(rgb).copy()
    return _convert(xyz_from_rgb, arr)


def linear_rgb2lab(rgb, illuminant="D65", observer="2"):
    """
    Copy of skimage.color.colorconv.rgb2lab for linear RGB values
    """
    return xyz2lab(linear_rgb2xyz(rgb), illuminant, observer)


def linear_rgb2xyz_grad(rgb):
    return numpy.matmul(xyz_from_rgb, rgb)


def linear_rgb2lab_grad(rgb, illuminant="D65", observer="2"):
    rgb = numpy.asanyarray(rgb)
    ref_white = numpy.asanyarray(get_xyz_coords(illuminant, observer))

    def elevate_replicate(array, shape):
        # replicate same shape as xyz with second last dimension aligned
        array = numpy.reshape(array, (1,) * len(shape[:-1]) + (-1, 1))
        return numpy.tile(array, shape[:-1] + (1, 3))

    def replicate(array):
        return numpy.tile(array[..., numpy.newaxis, :], (1,) * (len(array.shape) - 1) + (3, 1))

    white_point = elevate_replicate(ref_white, rgb.shape)

    lab = replicate(rgb)  # duplicate last axis
    xyz = replicate(numpy.squeeze(numpy.matmul(xyz_from_rgb, rgb[..., numpy.newaxis])))

    mask = xyz > (6 / 29) ** 3

    weights = xyz_from_rgb / (3 * white_point)
    lab[mask] = (
        numpy.power(xyz[mask], -2.0 / 3.0) * (xyz_from_rgb / (3 * white_point ** (1 / 3)))[mask]
    )
    lab[~mask] = (29 / 6) ** 2 * weights[~mask]  # verified

    J76 = numpy.array(
        [
            # X   Y   Z
            [0, 116, 0],  # L
            [500, -500, 0],  # a
            [0, 200, -200],  # b
        ],
        dtype=numpy.float,
    )

    return numpy.matmul(J76, lab)
