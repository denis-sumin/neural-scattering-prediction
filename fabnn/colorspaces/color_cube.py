import math

import numpy
from utils import dump_image


def build_color_cube(steps, cell_size):
    color_cube = numpy.empty(
        shape=(steps, steps * cell_size, steps * cell_size, 3), dtype=numpy.float32
    )

    for l, layer in enumerate(color_cube):
        r = l / (steps - 1)
        for i in range(steps):
            g = i / (steps - 1)
            for j in range(steps):
                b = j / (steps - 1)
                # print(r, g, b)
                rgb = numpy.array([r, g, b], dtype=numpy.float32)
                layer[
                    cell_size * i : cell_size * (i + 1),
                    cell_size * j : cell_size * (j + 1),
                ] = rgb

    return color_cube


def color_cube_to_image(color_cube, steps, cell_size):
    assert pow(int(math.sqrt(steps)), 2) == steps

    image = numpy.empty(
        shape=(
            int(math.sqrt(steps)) * steps * cell_size,
            int(math.sqrt(steps)) * steps * cell_size,
            3,
        ),
        dtype=numpy.float32,
    )

    stride = steps * cell_size
    for i in range(int(math.sqrt(steps))):
        for j in range(int(math.sqrt(steps))):
            tile = color_cube[i * int(math.sqrt(steps)) + j]
            image[stride * i : stride * (i + 1), stride * j : stride * (j + 1)] = tile

    return image


def build_color_slices_image(steps, cell_size):
    color_cube = build_color_cube(steps, cell_size)
    image = color_cube_to_image(color_cube, steps, cell_size)
    return image


def main():
    steps = 4
    cell_size = 118

    color_cube = build_color_cube(steps, cell_size)
    image = color_cube_to_image(color_cube, steps, cell_size)

    dump_image(image, "color_cube_4.png")


if __name__ == "__main__":
    main()
