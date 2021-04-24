#!/usr/bin/env python3
import argparse
import json

import mitsuba
import numpy as np
from materials import populate_materials
from render import render_scene
from utils import resolve_project_path

FILM_WIDTH = 256
FILM_HEIGHT = 256
OUTFILE = "thickness_gammut.json"


def get_args():
    parser = argparse.ArgumentParser(
        description="Run a rendering of homogeneous slabs with different thicknesses"
    )

    parser.add_argument(
        "--range-start", type=float, default=0.1, help="Thickness (in mm) to start"
    )
    parser.add_argument("--range-end", type=float, default=2, help="Thickness (in mm) to end")
    parser.add_argument(
        "--range-stepsize",
        type=float,
        default=0.1,
        help="Thickness (in mm) for the intermediate steps",
    )
    parser.add_argument(
        "--samples", type=int, default=256, help="Number of samples per simulation"
    )
    return parser.parse_args()


def main():
    args = get_args()

    scene_file = resolve_project_path("data/scenes/template/scene_thickness_gamut.xml")

    materials_file_path = resolve_project_path(
        "data/materials/g_constrained_fit_0.4_canonical_optimized.json"
    )
    materials = populate_materials(materials_file_path)

    material_points = dict()

    for material_label in (3, 4, 0, 1, 2):  # K,W,C,M,Y
        material = materials[material_label]
        material_points[material_label] = []
        for thickness in np.arange(args.range_start, args.range_end, args.range_stepsize):

            material_point = [0.0, 0.0, 0.0]
            for i, channel in enumerate(("r", "g", "b")):

                # TODO scaling necessary?
                max_density = 1.0  # max((float(m.density[i]) for m in materials.values()))

                albedo = material.albedo[i]
                density = material.density[i]

                param_map = mitsuba.core.StringMap()
                param_map["albedo"] = str(albedo)
                param_map["density"] = str(density)
                param_map["g"] = str(0.4)
                param_map["sample_count"] = str(args.samples)
                param_map["film_width"] = str(FILM_WIDTH)
                param_map["film_height"] = str(FILM_HEIGHT)
                param_map["density_scale"] = str(max_density)
                param_map["thickness"] = str(thickness)

                scene = mitsuba.render.SceneHandler.loadScene(scene_file, param_map)

                prediction = render_scene(scene, num_channels=3)[:, :, i]
                prediciton_mean = np.mean(prediction, dtype=np.float64)
                material_point[i] = prediciton_mean

                print("{} ({}): {}".format(material.name, thickness, material_point))

            material_points[material_label].append((thickness, material_point))
            with open(OUTFILE, "w") as outfile:
                json.dump(material_points, outfile)

        print("Finished {}".format(material.name))
        print(material_points[material_label])


if __name__ == "__main__":
    main()
