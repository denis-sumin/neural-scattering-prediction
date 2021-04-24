from functools import partial
from itertools import repeat
from multiprocessing.dummy import Pool as ThreadPool
from time import time

import numpy

from fabnn.utils import log_timing, read_image, setup_console_logger

from .data import Data

logger = setup_console_logger(__name__)


def build_sat_object_with_render(
    sat_object_class,
    dataset_item,
    discrete_volume,
    render,
    labels,
    materials,
    stencil_channels,
    patch_size,
    scale_levels,
    alignment_z_centered,
    stencils_only,
    verbose_logging,
    timings,
    channel,
):
    s = time()
    volume = numpy.empty(
        shape=discrete_volume.shape + (len(stencil_channels),), dtype=numpy.float32
    )
    log_timing(timings, "numpy.empty", time() - s)
    s = time()
    with ThreadPool() as p:
        p.starmap(
            Data.map_discrete_voxels,
            zip(
                discrete_volume,
                volume,
                repeat(labels),
                repeat(materials),
                repeat(channel),
                repeat(len(stencil_channels)),
            ),
        )
    log_timing(timings, "map_discrete_voxels", time() - s)

    if sat_object_class.__name__ not in timings:
        timings[sat_object_class.__name__] = {}

    s = time()
    sat_object = sat_object_class(
        volume=volume,
        patch_size=patch_size,
        scale_levels=scale_levels,
        stencil_channels=stencil_channels,
        verbose_logging=verbose_logging,
        tile_size=dataset_item["metadata"]["tile_size"],
        alignment_z_centered=alignment_z_centered,
        timings=timings[sat_object_class.__name__],
    )
    log_timing(timings, "Create {}".format(sat_object_class.__name__), time() - s)
    s = time()
    del volume
    log_timing(timings, "numpy.deallocate", time() - s)
    render_channel = render[:, :, channel] if not stencils_only else None
    return sat_object, render_channel


class DataPlanar(Data):
    """
    Data implementation for planar 2.5D slabs
    """

    def __init__(self, *args, **kwargs):
        self.voxel_size = (25.4 / 300.0, 25.4 / 300.0, 0.027)
        super().__init__(*args, **kwargs)

    def load_volume_s_b_with_render(self, idx=None, dataset_item_key=None):
        dataset_item = self.data_items[dataset_item_key]
        filename = dataset_item["filename"]

        volume_path = dataset_item["volume_path"]

        if self.verbose_logging:
            logger.info("Loading volume {}".format(filename))
        d = numpy.load(volume_path, allow_pickle=True)

        thickness = self.thickness
        discrete_volume = d["halftoned_voxels"][:thickness]

        if not self.stencils_only:
            render = self.get_render(dataset_item_key=dataset_item_key, data_object=d)
            assert (discrete_volume.shape[1] - render.shape[0]) % 2 == 0
            assert (discrete_volume.shape[2] - render.shape[1]) % 2 == 0
        else:
            render = None

        if "build_sat_object_with_render" not in self.timings:
            self.timings["build_sat_object_with_render"] = {}
        build_sat_object_with_render_func = partial(
            build_sat_object_with_render,
            self.sat_object_class,
            dataset_item,
            discrete_volume,
            render,
            self.labels,
            self.materials,
            self.stencil_channels,
            self.patch_size,
            self.scale_levels,
            self.alignment_z_centered,
            self.stencils_only,
            self.verbose_logging,
            self.timings["build_sat_object_with_render"],
        )

        # try:
        #     with Pool(3) as pool:
        #         volumes_s_b_renders = pool.map(build_sat_object_with_render_func, range(3))
        # except AssertionError:  # for the case multiprocessing is already used
        volumes_s_b_renders = list(map(build_sat_object_with_render_func, range(3)))

        return volumes_s_b_renders

    def get_render(self, dataset_item_key=None, data_object=None, **kwargs):
        dataset_item = self.data_items[dataset_item_key]

        volume_path = dataset_item["volume_path"]
        render_path = dataset_item["render_path"]

        if render_path is None or render_path == volume_path:
            if data_object is None:
                data_object = numpy.load(volume_path, allow_pickle=True)
            try:
                render = data_object["rendered_image"]
                h, w = render.shape[:2]
            except (IndexError, KeyError):
                logger.error("No rendering in {}".format(volume_path))
                raise
        else:
            render = read_image(render_path)
        return render

    @staticmethod
    def estimate_patch_count(metadata):
        return metadata["height"] * metadata["width"]

    def convert_patch_index(self, datafile_idx, channel, pidx):
        _, metadata = self.volume_files[datafile_idx]

        assert pidx < DataPlanar.estimate_patch_count(metadata), "patch index within range"
        x = pidx % metadata["width"]
        y = pidx // metadata["width"]
        z = None
        return x, y, z

    def get_stencil_prediction(self, datafile_idx, channel, pidx):
        volumes_s_b_renders = self.s_b_sat_volumes[datafile_idx]
        sat_object, render = volumes_s_b_renders[channel]

        x, y, z = self.convert_patch_index(datafile_idx, channel, pidx)
        _, metadata = self.volume_files[datafile_idx]

        prediction = render[y, x] if not self.stencils_only else None
        stencils = sat_object.get_stencils(x=x + metadata["pad_x"], y=y + metadata["pad_y"])
        return stencils, prediction

    def get_stencil_prediction_list(self, datafile_idx, channel, pidxs):
        volumes_s_b_renders = self.s_b_sat_volumes[datafile_idx]
        sat_object, render = volumes_s_b_renders[channel]

        xyz_list = [self.convert_patch_index(datafile_idx, channel, pidx) for pidx in pidxs]
        _, metadata = self.volume_files[datafile_idx]
        zyx_pad_list = [(0, y + metadata["pad_y"], x + metadata["pad_x"]) for x, y, z in xyz_list]

        predictions = [render[y, x] for x, y, z in xyz_list] if not self.stencils_only else None
        stencils = sat_object.get_stencils_list(zyx_pad_list)
        return stencils, predictions
