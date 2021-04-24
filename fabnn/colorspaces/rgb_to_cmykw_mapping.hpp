//
// Created by Denis Sumin on 24/01/2018.
//

#include <Eigen/Core>

#include <fstream>
#include <iostream>
#include <stdexcept>

typedef Eigen::Array< double, 5, 1 > Vec5d;


class MappingRGBtoCMYKW {
    Vec5d* m_data;
    size_t m_samples_per_axis;

    inline size_t grid_elements_count() const {
        return m_samples_per_axis * m_samples_per_axis * m_samples_per_axis;
    }

    inline size_t grid_bytes_count() const {
        return grid_elements_count() * 5 * sizeof(double);
    }

 public:
    MappingRGBtoCMYKW(): m_data(nullptr), m_samples_per_axis(0) {}

    ~MappingRGBtoCMYKW() {
        delete[] m_data;
    }

    inline Vec5d at(size_t r_idx, size_t g_idx, size_t b_idx) const {
        size_t idx = r_idx * m_samples_per_axis * m_samples_per_axis +
                     g_idx * m_samples_per_axis + b_idx;
        if (idx < grid_elements_count()) {
            return m_data[idx];
        } else {
// TODO(denis): fix the exception throwing
            throw std::out_of_range("Index is out of the range");
        }
    }

    inline Vec5d get_cmykw_mixture(double r, double g, double b) const {
        auto r_idx = static_cast<size_t>(
                std::round(r * (m_samples_per_axis - 1)));
        auto g_idx = static_cast<size_t>(
                std::round(g * (m_samples_per_axis - 1)));
        auto b_idx = static_cast<size_t>(
                std::round(b * (m_samples_per_axis - 1)));
        return at(r_idx, g_idx, b_idx);
    }

    void read_from_file(const std::string& filename) {
        std::ifstream file(filename,
                           std::ios::in|std::ios::binary|std::ios::ate);
        if (file.is_open()) {
            file.seekg(0, std::ios::beg);

            int samples_per_axis;
            file.read(reinterpret_cast<char*>(&samples_per_axis), sizeof(int));

            if (samples_per_axis <= 0) {
// TODO(denis): report the error properly
                std::cout << "Sample integer in the grid file is negative"
                          << std::endl;
                return;
            }
            m_samples_per_axis = static_cast<size_t>(samples_per_axis);

            m_data = new Vec5d[grid_elements_count()];
            file.read(reinterpret_cast<char*>(m_data), grid_bytes_count());
            file.close();
        } else {
            std::ostringstream err;
            err << "Unable to open file with RGB to CMYKW mapping " << filename;
            throw std::runtime_error(err.str());
        }
    }
};
