from functools import partial

import numpy
from skimage.color import deltaE_cie76, deltaE_ciede94, deltaE_ciede2000, rgb2lab
from skimage.measure import compare_ssim

# weighting factor used for assymetrical weightings of luminance, chroma & hue
# range: [0,\infinity),  higher -> less important
CIE_DE_LCH_WEIGHT_DIFFERENCE = 100


def rms(reference: numpy.ndarray, distorted: numpy.ndarray) -> float:
    return numpy.sqrt(numpy.mean(numpy.square(reference - distorted)))


def ssim(reference: numpy.ndarray, distorted: numpy.ndarray) -> float:
    assert reference.shape == distorted.shape, "Shapes do not match"

    if len(reference.shape) == 3:
        channels = reference.shape[2]
    elif len(reference.shape) == 2:
        channels = 1
    else:
        raise ValueError("Wrong shape dimensions")

    if channels == 3:
        return compare_ssim(reference, distorted, multichannel=True)
    elif channels == 1:
        return compare_ssim(reference, distorted)
    else:
        raise ValueError("Support 1- or 3-channel images only")


def cie_de(
    reference: numpy.ndarray,
    distorted: numpy.ndarray,
    dE_function="2000",
    lightness_weight=1.0,
    chroma_weight=1.0,
    hue_weight=1.0,
) -> numpy.ndarray:
    assert reference.shape == distorted.shape, "Shapes do not match"

    if len(reference.shape) == 2:
        reference = reference[numpy.newaxis, ...]
        distorted = distorted[numpy.newaxis, ...]

    reference_lab = rgb2lab(reference)
    distorted_lab = rgb2lab(distorted)

    if dE_function == "2000":
        deltaE = deltaE_ciede2000(
            reference_lab, distorted_lab, lightness_weight, chroma_weight, hue_weight
        )
    elif dE_function == "1994":
        deltaE = deltaE_ciede94(
            reference_lab, distorted_lab, hue_weight, chroma_weight, lightness_weight
        )
    elif dE_function == "1976":
        deltaE = deltaE_cie76(reference_lab, distorted_lab)
    else:
        raise NameError("CIE dE function with name {} not found".format(dE_function))
    return deltaE


def get_difference_metric(metric_key="rms"):
    """
    Returns a function with signature:
    (reference: numpy.ndarray, distored: numpy.ndarray) -> float/numpy.ndarray

    For compatiblity call numpy.mean() on the result to get a single float
    """
    w = CIE_DE_LCH_WEIGHT_DIFFERENCE
    return {
        "rms": rms,
        "ciede2000": partial(cie_de, dE_function="2000"),
        "ciede1994": partial(cie_de, dE_function="1994"),
        "ciede1976": partial(cie_de, dE_function="1976"),
        "lightness": partial(
            cie_de,
            dE_function="2000",
            lightness_weight=1.0,
            chroma_weight=w,
            hue_weight=w,
        ),
        "chroma": partial(
            cie_de,
            dE_function="2000",
            lightness_weight=w,
            chroma_weight=1.0,
            hue_weight=w,
        ),
        "hue": partial(
            cie_de,
            dE_function="2000",
            lightness_weight=w,
            chroma_weight=w,
            hue_weight=1.0,
        ),
        "ssim": ssim,
    }.get(metric_key.lower(), rms)


def get_colormap(metric_key="rms"):
    """
    Returns a meaningful colormap for a difference metric
    """
    return {"error": "RdBu"}.get(metric_key.lower(), "inferno")
