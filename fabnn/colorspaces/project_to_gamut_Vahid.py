import sys

import numpy
from utils import dump_array, dump_image, read_image, resolve_project_path

MAPPING_FILE = resolve_project_path("data/gamut_mapping_Vahid_table.npz")


def build_map():
    # these PNGs are obtained from tif via ImageMagick's convert command (convert allRgb.tif allRgb.png)
    all_rgb_png = read_image("/HPS/FabricationEnhancement_DISTRO/work/objects/allRgb.png")
    all_rgb_gm_png = read_image(
        "/HPS/FabricationEnhancement_DISTRO/work/objects/allRgb_gm_vb_lin3050.png"
    )

    all_rgb_png_uint8 = (all_rgb_png * 255 + 0.5).astype("uint8")

    map_ = numpy.empty(shape=(256, 256, 256, 3), dtype="float32")
    for i in range(all_rgb_png_uint8.shape[0]):
        for j in range(all_rgb_png_uint8.shape[1]):
            r, g, b = all_rgb_png_uint8[i, j]
            map_[r, g, b] = all_rgb_gm_png[i, j]
        print("\r{} / {}".format(i + 1, all_rgb_png_uint8.shape[0]), end="")
    print()

    dump_array(map_, MAPPING_FILE)


def project(image_src_path: str, image_dst_path: str):
    map_ = numpy.load(MAPPING_FILE)["data"]

    image = read_image(image_src_path)
    image_uint8 = (image * 255 + 0.5).astype("uint8")
    image_projected = numpy.zeros(shape=image_uint8.shape, dtype="float32")
    for i in range(image_uint8.shape[0]):
        for j in range(image_uint8.shape[1]):
            r, g, b = image_uint8[i, j]
            image_projected[i, j] = map_[r, g, b]
        print("\r{} / {}".format(i + 1, image_uint8.shape[0]), end="")
    print()

    dump_image(image_projected, image_dst_path)


def main():
    input_file = sys.argv[1]
    output_file = sys.argv[2]

    project(input_file, output_file)


if __name__ == "__main__":
    main()
