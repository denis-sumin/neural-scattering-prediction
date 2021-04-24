import random
import unittest
from timeit import default_timer as timer

import numpy
from average_octree import AverageKDTree2D as AverageKDTree


def sizeof_fmt(num, suffix="B"):
    for unit in ["", "Ki", "Mi", "Gi", "Ti", "Pi", "Ei", "Zi"]:
        if abs(num) < 1024.0:
            return "%3.1f%s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f%s%s" % (num, "Yi", suffix)


class TestAverageKDTree(unittest.TestCase):
    def setUp(self):
        raise unittest.SkipTest("parent class")

    def tearDown(self):
        self.testend = timer()
        numpy_size = self.values.size * self.values.itemsize
        # print("{:.4E}\t{:.4E}\t{}".format(self.buildend - self.buildstart, self.testend - self.buildend, self.id()))

        tree_size = self.tree.size()
        tree_node_count = int(tree_size / 24)
        print(
            "{:.4E}\t{:.4E}\t{}/{}\t{}\t{}".format(
                self.buildend - self.buildstart,
                self.testend - self.buildend,
                sizeof_fmt(tree_size),
                sizeof_fmt(numpy_size),
                tree_node_count,
                self.id(),
            )
        )
        # print(self.tree.getStats())

    def query_helper(self, indices):
        tree_result = self.tree.query_average(*tuple(indices)[::-1])
        numpy_result = numpy.average(self.values[indices].astype(numpy.float), axis=(0, 1, 2))
        result = numpy.isclose(tree_result, numpy_result)
        if not result.all():
            print("Query indices: {}".format(indices))
            print(tree_result, numpy_result)
        return result.all()

    def test_reconstruct(self):
        tree_result = self.tree.toNumpy()
        self.assertTrue(numpy.isclose(tree_result, self.values).all())

    def test_query_empty(self):
        indices = numpy.s_[:0, :0, :0]
        self.assertTrue(numpy.isclose(self.tree.query_average(*tuple(indices)), 0.0).all())

    def test_query_full(self):
        s = self.values.shape
        indices = numpy.s_[: s[0], : s[1], : s[2]]
        self.assertTrue(self.query_helper(indices))

    def test_query_outside(self):
        s = self.values.shape
        indices = numpy.s_[: s[0] + 4, : s[1] + 6, : s[2] + 12]
        self.assertTrue(self.query_helper(indices))

    def test_query_partially_outside(self):
        s = self.values.shape
        indices = numpy.s_[: s[0] + 3, :-1, :-2]
        self.assertTrue(self.query_helper(indices))

    def test_query_randomly_inside(self):
        s = self.values.shape
        starts = [
            random.randrange(s[0] - 1),
            random.randrange(s[1] - 1),
            random.randrange(s[2] - 1),
        ]
        indices = numpy.s_[
            starts[0] : starts[0] + random.randrange(1, s[0] - starts[0]),
            starts[1] : starts[1] + random.randrange(1, s[1] - starts[1]),
            starts[2] : starts[2] + random.randrange(1, s[2] - starts[2]),
        ]
        self.assertTrue(self.query_helper(indices))

    def test_generate_scales(self):
        s = self.values.shape
        SCALE_LEVELS = [
            [1, 1],
            [2, 2],
            [4, 4],
            [6, 6],
            [8, 8],
            [12, 12],
            [16, 16],
            [32, 32],
            [64, 64],
            [128, 128],
            # [256, 256],
            # [512, 512],
            # [1024, 1024],
            # [2048, 2048],
            # [128, 128],
            # [256, 256],
            # [512, 512],
            # [1024, 1024],
            # [2048, 2048],
        ]
        scale_levels_list = list(SCALE_LEVELS)
        patch_size = (3, 5, 5)
        for y in range(0, s[1], 10):
            for x in range(0, s[2], 10):
                self.tree.generate_scales(scale_levels_list, patch_size, 0, y, x)
        self.assertTrue(True)

    # def test_query_debug(self):
    #     s = self.values.shape
    #     indices = numpy.s_[0:2,2:3,:3]
    #     self.tree.print()
    #     self.assertTrue(self.query_helper(indices))


class DenseTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(4, 4, 4, 2)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class SparseSubTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(4, 4, 4, 2)
        self.values[:2, :2, :2] = numpy.random.rand(1)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class ConstantInnerTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(4, 4, 4, 2)
        s = self.values.shape
        self.values[1:-1, 1:-1, 1:-1] = 0.0
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class ThinDenseTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(3, 3, 33, 2)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class ThinSparseSubTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(33, 3, 3, 2)
        self.values[:2, :2, :2] = numpy.random.rand(1)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class ThinConstantInnerTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(5, 33, 3, 2)
        s = self.values.shape
        self.values[1:-1, 1:-1, 1:-1] = 0.0
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class LargeDenseTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(300, 128, 128, 2)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class LargeSparseSubTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(300, 128, 256, 2)
        self.values[:2, :2, :2] = numpy.random.rand(1)
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class LargeConstantInnerTree(TestAverageKDTree):
    def setUp(self):
        self.values = numpy.random.rand(256, 128, 300, 2)
        self.values[30:-30, :, :] = numpy.random.rand(1)
        self.buildstart = timer()
        for i in range(10):
            self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


class RepetitiveTree(TestAverageKDTree):
    def setUp(self):
        self.values = 55555555 + numpy.random.rand(5, 4, 7, 2) * 1e-4
        self.values = numpy.tile(self.values, (23, 16, 33, 1))
        self.buildstart = timer()
        self.tree = AverageKDTree(self.values, False)
        self.buildend = timer()


if __name__ == "__main__":
    unittest.main()
