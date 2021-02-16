from . import (
    first_baseline,
    level_shared,
    planar_radial_symmetry,
    planar_radial_symmetry_2,
    planar_radial_symmetry_3,
    x3d_radial_symmetry_2,
    x3d_radial_symmetry_3_level_shared,
    x3d_radial_symmetry_4_level_shared,
)

models_collection = {
    "3d_radial_symmetry_2": x3d_radial_symmetry_2.make_model,
    "3d_radial_symmetry_3_level_shared": x3d_radial_symmetry_3_level_shared.make_model,
    "3d_radial_symmetry_4_level_shared": x3d_radial_symmetry_4_level_shared.make_model,
    "first_baseline": first_baseline.make_model,
    "level_shared": level_shared.make_model,
    "planar_first": first_baseline.make_model,
    "planar_radial_symmetry": planar_radial_symmetry.make_model,
    "planar_radial_symmetry_2": planar_radial_symmetry_2.make_model,
    "planar_radial_symmetry_3": planar_radial_symmetry_3.make_model,
}
