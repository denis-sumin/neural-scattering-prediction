import pickle
import struct
import sys

import numpy

CPP_DOUBLE_DTYPE = numpy.float64

lookup_table_file = sys.argv[1]
sampling = int(sys.argv[2])
output_filename = sys.argv[3]

with open(lookup_table_file, "rb") as f:
    lookup_table = pickle.load(f)
print("Lookup table loaded")


grid = numpy.empty(shape=(sampling, sampling, sampling, 5), dtype=CPP_DOUBLE_DTYPE)

for r_idx in range(sampling):
    print(r_idx, "/", sampling - 1, end="\r")
    for g_idx in range(sampling):
        for b_idx in range(sampling):
            grid[r_idx][g_idx][b_idx] = numpy.array(
                lookup_table[
                    (
                        r_idx / (sampling - 1),
                        g_idx / (sampling - 1),
                        b_idx / (sampling - 1),
                    )
                ],
                dtype=CPP_DOUBLE_DTYPE,
            )


with open(output_filename, "wb") as f:
    f.write(struct.pack("i", sampling))
    f.write(grid.tobytes(order="C"))
