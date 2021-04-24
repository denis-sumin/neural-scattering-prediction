//
// Created by denis on 1/22/18.
//

#include <fstream>
#include <iostream>
#include <unordered_map>

#include "../json.hpp"
#include "../materials.hpp"
#include "nanoflann.hpp"


using json = nlohmann::json;


double round_n_decimals(double x, int n) {
    auto p = pow(10.0, n);
    return std::round(x * p) / p;
}


inline double index_transform_linear(size_t idx, size_t steps) {
    return static_cast<double>(idx) / (steps - 1);
}


inline double index_transform_pow2(size_t idx, size_t steps) {
    return std::pow((static_cast<double>(idx) / (steps - 1)), 2);
}


inline Vec5d cmykw_idx_to_values(
        size_t c_idx, size_t m_idx, size_t y_idx, size_t k_idx, size_t steps) {
    auto round_cmykw = static_cast<int>(
            std::round(std::log10(static_cast<double>(steps))));

    double eps = 0.00001;
    auto cyan = index_transform_pow2(c_idx, steps);
    auto magenta = index_transform_pow2(m_idx, steps);
    auto yellow = index_transform_pow2(y_idx, steps);
    auto black = index_transform_pow2(k_idx, steps);
    auto cmyk_sum = (cyan + magenta + yellow + black);
    double white;
    if (std::abs(cmyk_sum - 1.0) < eps) {
        white = 0.0;
    } else {
        white = round_n_decimals(1.0 - cmyk_sum, round_cmykw);
    }
    Vec5d result;
    result << cyan, magenta, yellow, black, white;
    return result;
}


void gen_cmykw_to_rgb_table(
        Vec3d* rgb_values,
        size_t steps,
        const std::unordered_map<unsigned char, Material>& materials) {
    Vec3d inactive_rgb = {-100.0, -100.0, -100.0};

#pragma omp parallel for
    for (size_t c_idx = 0; c_idx < steps; ++c_idx) {
        std::cout << c_idx << " / " << steps << "\r" << std::flush;
        for (size_t m_idx = 0; m_idx < steps; ++m_idx) {
            for (size_t y_idx = 0; y_idx < steps; ++y_idx) {
                for (size_t k_idx = 0; k_idx < steps; ++k_idx) {

                    Vec5d cmykw = cmykw_idx_to_values(
                            c_idx, m_idx, y_idx, k_idx, steps);
                    size_t idx =
                            steps * steps * steps * c_idx +
                            steps * steps * m_idx +
                            steps * y_idx +
                            k_idx;
                    if (cmykw[4] < 0.0) {
                        rgb_values[idx] = inactive_rgb;
                        continue;
                    }
                    rgb_values[idx] = fm_mixture(cmykw, materials);
                }
            }
        }
    }
}


struct NanoFlannPointCloud {
    const Vec3d* pts;
    size_t pts_count;

    explicit NanoFlannPointCloud(
            const Vec3d* points, size_t points_number):
            pts(points), pts_count(points_number) {}

    inline size_t kdtree_get_point_count() const {
        return pts_count;
    }

    inline double kdtree_get_pt(const size_t idx, int dim) const {
        return pts[idx][dim];
    }

    template <class BBOX>
    bool kdtree_get_bbox(BBOX& /* bb */) const {
        return false;
    }

};


typedef nanoflann::KDTreeSingleIndexAdaptor<nanoflann::L2_Simple_Adaptor<
        double, NanoFlannPointCloud>, NanoFlannPointCloud, 3> my_kd_tree_t;


size_t find_nn(const my_kd_tree_t& index, const Vec3d& query) {
    const size_t num_results = 1;
    size_t res_index;
    double res_sqr_distance;

    nanoflann::KNNResultSet<double> resultSet(num_results);
    resultSet.init(&res_index, &res_sqr_distance);
    index.findNeighbors(resultSet, &query[0], nanoflann::SearchParams());

    return res_index;
}


void test_index(const Vec3d* points, size_t points_count,
                const my_kd_tree_t& rgb_values_index) {
    size_t steps = 11;
    for (size_t r_idx = 0; r_idx < steps; ++r_idx) {
        for (size_t g_idx = 0; g_idx < steps; ++g_idx) {
            for (size_t b_idx = 0; b_idx < steps; ++b_idx) {
                auto r = index_transform_linear(r_idx, steps);
                auto g = index_transform_linear(g_idx, steps);
                auto b = index_transform_linear(b_idx, steps);
                auto idx = find_nn(rgb_values_index, {r, g, b});
                std::cout << Vec3d({r, g, b}) << " " << points[idx] << std::endl;
            }
        }
    }
}


int main(int argc, char** argv) {
    if (argc != 6) {
        std::cout << "Usage: " << argv[0]
                  << " <input json with materials data: string>"
                  << " <output file for the cmykw-to-rgb table: string>"
                  << " <number of bins in the cmykw-to-rgb table: int>"
                  << " <output file for the rgb-to-cmykw table: string>"
                  << " <number of bins in the rgb-to-cmykw table: int>"
                  << std::endl;
        exit(1);
    }

    std::string materials_file = argv[1];
    std::string outfile_cmykw_to_rgb_table = argv[2];
    auto cmykw_to_rgb_table_bins = static_cast<size_t>(std::stoi(argv[3]));
    std::string outfile_rgb_to_cmykw_table = argv[4];
    auto rgb_to_cmykw_bins = static_cast<size_t>(std::stoi(argv[5]));

    auto materials = build_materials_map(materials_file);

    size_t rgb_values_count =
            cmykw_to_rgb_table_bins * cmykw_to_rgb_table_bins *
                    cmykw_to_rgb_table_bins * cmykw_to_rgb_table_bins;
    auto rgb_values = new Vec3d[rgb_values_count];

    gen_cmykw_to_rgb_table(rgb_values, cmykw_to_rgb_table_bins, materials);

    NanoFlannPointCloud point_cloud(rgb_values, rgb_values_count);

    my_kd_tree_t rgb_values_index(
            3, point_cloud, nanoflann::KDTreeSingleIndexAdaptorParams(10));
    rgb_values_index.buildIndex();

    test_index(rgb_values, rgb_values_count, rgb_values_index);

    delete[] rgb_values;
}
