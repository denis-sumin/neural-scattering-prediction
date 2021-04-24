//
// Created by Denis Sumin on 29.11.2019.
//

#include <algorithm>
#include <iostream>
#include <variant>
#include <vector>

#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include "bounding_box.h"

namespace py = pybind11;

// #define SAT_TILE_TREE_STATS
#define SAT_TILE_TREE_USE_CACHING

template <class _DataType>
class SATTileTree {
    using DataType = _DataType;
    using ArrayType = py::array_t<typename SATTileTree<_DataType>::DataType::Scalar, py::array::c_style | py::array::forcecast>;
    using TensorType = Eigen::TensorMap<Eigen::Tensor<DataType, 3, Eigen::RowMajor>>;

    enum {
        Dimensionality = BoundingBox::Scalar(DataType::RowsAtCompileTime),
    };

    using UIntDataType = Eigen::Matrix<unsigned int, Dimensionality, 1>;
    using DoubleDataType = Eigen::Matrix<double, Dimensionality, 1>;
    struct QueryCacheType{
        QueryCacheType(vec3u extends) : m_extends(extends), 
        m_emptyCacheFlag(DoubleDataType::Constant(std::numeric_limits< double >::lowest())) {
            // allocate and clear at once
            m_data = std::vector<DoubleDataType>(m_extends[0] * m_extends[1] * m_extends[2], m_emptyCacheFlag);
        }

        void clear(){
            std::fill(m_data.begin(), m_data.end(), m_emptyCacheFlag);
        }

        inline size_t convertIndex(unsigned int rel_x, unsigned int rel_y, unsigned int rel_z){
            return 
                rel_z * m_extends[1] * m_extends[0]+
                rel_y * m_extends[0]+
                rel_x;
        }

        /**
         * check for existence and return the value immediately
        */
        inline bool contains(size_t index, DoubleDataType & value){
            value = m_data[index];
            return value != m_emptyCacheFlag;
        }

        inline void insert(size_t index, DoubleDataType value){
            m_data[index] = value;
        }

        std::vector<DoubleDataType> m_data;
        DoubleDataType m_emptyCacheFlag;
        vec3u m_extends;
    };
public:
    SATTileTree(const ArrayType& volume, unsigned short tile_size, bool alignment_z_centered) {
        py::buffer_info buffer = volume.request();

        if (buffer.ndim != 4)
            throw std::runtime_error("Number of dimensions must be 4");

        if (buffer.shape[3] != Dimensionality){
            std::ostringstream err;
            err << "Input array of shape [..., "<< buffer.shape[3]
                << "] does not match Dimensionality "<< Dimensionality;
            throw std::runtime_error(err.str());
        }

        m_tile_size = tile_size;

        auto volume_tensor = TensorType(
                reinterpret_cast<DataType*>(buffer.ptr),
                buffer.shape[0], buffer.shape[1], buffer.shape[2]);

        m_dataBBox = BoundingBox(
                {0,0,0},
                {buffer.shape[2], buffer.shape[1], buffer.shape[0]});

        m_alignment_z_centered = alignment_z_centered;
        #ifdef SAT_TILE_TREE_STATS
        __STATS_requests = 0;
        __STATS_cache_hits = 0;
        __STATS_cache_collisions = 0;
        #endif // SAT_TILE_TREE_STATS

        constructTree(m_dataBBox, volume_tensor);
    }
    ~SATTileTree() {
        #ifdef SAT_TILE_TREE_STATS
        std::cout << "hits " << __STATS_cache_hits << " requests " << __STATS_requests << std::endl;
        std::cout << "hits " << double(__STATS_cache_hits) / double(__STATS_requests) << " collisions " << double(__STATS_cache_collisions) / double(__STATS_requests) << std::endl;
        #endif // SAT_TILE_TREE_STATS
    }

    DataType queryAverage(const BoundingBox& queryBox, QueryCacheType* localCache, vec3u rel_offset){
        DataType sum;
        if (m_dataBBox.contains(queryBox)) {
            sum = get_sum_from_sat(queryBox, localCache, rel_offset);
        } else {
            //handle virtual zeros around
            auto overlap_box = m_dataBBox.overlap(queryBox);
            // std::cout << "m_dataBBox  " << m_dataBBox.lower << " " << m_dataBBox.upper << std::endl;
            // std::cout << "queryBox  " << queryBox.lower << " " << queryBox.upper << std::endl;
            // std::cout << "overlap_box  " << overlap_box.lower << " " << overlap_box.upper << std::endl;
            if (overlap_box.volume() > 0) {
                sum = get_sum_from_sat(overlap_box, localCache, rel_offset);
            } else {
                return DataType::Zero();
            }
        }
        return sum / queryBox.volume();
    }

    DataType queryAverageSlice(py::slice x_slice, py::slice y_slice, py::slice z_slice){
        size_t x1, x2, y1, y2, z1, z2, step, slicelength;
        x_slice.compute(m_dataBBox.upper[0], &x1, &x2, &step, &slicelength);
        y_slice.compute(m_dataBBox.upper[1], &y1, &y2, &step, &slicelength);
        z_slice.compute(m_dataBBox.upper[2], &z1, &z2, &step, &slicelength);

        QueryCacheType localCache({2, 2, 2});
        BoundingBox queryBox(
                {
                        static_cast<int>(x1),
                        static_cast<int>(y1),
                        static_cast<int>(z1)
                },
                {
                        static_cast<int>(x2),
                        static_cast<int>(y2),
                        static_cast<int>(z2)
                });

        return queryAverage(queryBox, &localCache, {0, 0, 0});
    }

    ArrayType generate_scales(
            const std::vector<std::tuple<int, int>>& scale_levels,
            const std::tuple<int, int, int>& patch_size,
            int coord_z, int coord_y, int coord_x) {
        if (scale_levels.empty()) {
            throw std::invalid_argument("scale_levels list cannot be empty");
        }

        auto patch_size_x = std::get<2>(patch_size);
        auto patch_size_y = std::get<1>(patch_size);
        auto patch_size_z = std::get<0>(patch_size);

        auto result = ArrayType(
                {
                        static_cast<int>(scale_levels.size()),
                        static_cast<int>(patch_size_z),
                        static_cast<int>(patch_size_y),
                        static_cast<int>(patch_size_x),
                        static_cast<int>(Dimensionality)
                });

        py::gil_scoped_release release;
        fill_one_stencil(result.mutable_data(), scale_levels, patch_size_x, patch_size_y, patch_size_z, coord_z, coord_y, coord_x);
        py::gil_scoped_acquire acquire;

        return result;
    }

    ArrayType generate_scales_list(
            const std::vector<std::tuple<int, int>>& scale_levels,
            const std::tuple<int, int, int>& patch_size,
            const std::vector<std::tuple<int, int, int>>& coords) {
        if (scale_levels.empty()) {
            throw std::invalid_argument("scale_levels list cannot be empty");
        }
        if (coords.empty()) {
            throw std::invalid_argument("coords list cannot be empty");
        }

        auto patch_size_x = std::get<2>(patch_size);
        auto patch_size_y = std::get<1>(patch_size);
        auto patch_size_z = std::get<0>(patch_size);

        auto result = ArrayType(
                {
                        static_cast<int>(coords.size()),
                        static_cast<int>(scale_levels.size()),
                        static_cast<int>(patch_size_z),
                        static_cast<int>(patch_size_y),
                        static_cast<int>(patch_size_x),
                        static_cast<int>(Dimensionality)
                });

        size_t stride_len =
                scale_levels.size() * patch_size_z * patch_size_y * patch_size_x * Dimensionality;

        py::gil_scoped_release release;

        {
#pragma omp parallel for default(none) shared(coords, scale_levels, patch_size_x, patch_size_y, patch_size_z, result, stride_len)
            for (size_t coord_idx = 0; coord_idx < coords.size(); ++coord_idx) {
                auto coord = coords[coord_idx];
                auto coord_z = std::get<0>(coord);
                auto coord_y = std::get<1>(coord);
                auto coord_x = std::get<2>(coord);
                auto ptr = result.mutable_data() + coord_idx * stride_len;
                fill_one_stencil(ptr, scale_levels,
                                 patch_size_x, patch_size_y, patch_size_z,
                                 coord_z, coord_y, coord_x);
            }
        }

        py::gil_scoped_acquire acquire;

        return result;
    }

    ArrayType get_sat_value_py(size_t x, size_t y, size_t z) {
        auto result = ArrayType({BoundingBox::Scalar(Dimensionality)});
        auto r = result.mutable_unchecked();
        auto value = get_sat_value(x, y, z);
        for (unsigned char c = 0; c < Dimensionality; ++c) {
            r(c) = value[c];
        }
        return result;
    }

    ArrayType get_sat() {
        vec3 dimensions = m_dataBBox.upper - m_dataBBox.lower;
        auto result = ArrayType({dimensions[2], dimensions[1], dimensions[0], BoundingBox::Scalar(Dimensionality)});
        auto r = result.mutable_unchecked();

        for (int z = 0; z < dimensions[2]; ++z) {
#pragma omp parallel for
            for (int y = 0; y < dimensions[1]; ++y) {
                for (int x = 0; x < dimensions[0]; ++x) {
                    auto& tile = m_tiles_tensor(
                            static_cast<long>(z / m_tile_size),
                            static_cast<long>(y / m_tile_size),
                            static_cast<long>(x / m_tile_size));
                    auto value = tile.get_value(
                            x - x / m_tile_size * m_tile_size,
                            y - y / m_tile_size * m_tile_size,
                            z - z / m_tile_size * m_tile_size);
                    for (unsigned char c = 0; c < Dimensionality; ++c) {
                        r(z, y, x, c) = value[c];
                    }
                }
            }
        }
        return result;
    }

    ArrayType toNumpy() {
        /* No pointer is passed, so NumPy will allocate the buffer */
        auto result = ArrayType({m_dataBBox.upper[2], m_dataBBox.upper[1], m_dataBBox.upper[0], BoundingBox::Scalar(Dimensionality)});
//        writeToArray(m_dataBBox, m_root, result);
        return result;
    }

    void print() const {
//        std::cout << m_root->toString(0);
    }

    long unsigned int size() const {
        long unsigned int size = 0;

        vec3 volume_dimensions = m_dataBBox.extend();
        vec3 tile_tree_dimensions = (volume_dimensions + vec3(m_tile_size - 1, m_tile_size - 1, m_tile_size - 1)) / m_tile_size;
        for (int tile_z = 0; tile_z < tile_tree_dimensions[2]; ++tile_z) {
            for (int tile_y = 0; tile_y < tile_tree_dimensions[1]; ++tile_y) {
                for (int tile_x = 0; tile_x < tile_tree_dimensions[0]; ++tile_x) {
                    auto& tile_tensor = m_tiles_tensor(
                            static_cast<long>(tile_z),
                            static_cast<long>(tile_y),
                            static_cast<long>(tile_x));
//                    std::cout << tile_tensor.m_type << " " << tile_tensor.getSize() << std::endl;
                    size += tile_tensor.getSize();
                }
            }
        }
        return size;
    }

private:
    struct Tile {
        Tile() :
                m_data(static_cast<DataType*>(nullptr)), m_tile_offset(UIntDataType::Zero()),
                m_prev_slice_xy(nullptr), m_prev_slice_yz(nullptr), m_prev_slice_xz(nullptr), m_tile_size(0) {}

        void tile_set_data(
                unsigned short tile_size, UIntDataType tile_offset,
                DoubleDataType* prev_slice_xy, DoubleDataType* prev_slice_yz, DoubleDataType* prev_slice_xz,
                DataType tile_value) {
            m_tile_offset = tile_offset;
            m_data = tile_value;
            m_prev_slice_xy = prev_slice_xy;
            m_prev_slice_yz = prev_slice_yz;
            m_prev_slice_xz = prev_slice_xz;
            m_tile_size = tile_size;
        }

        void tile_set_data(
                unsigned short tile_size, UIntDataType tile_offset,
                DoubleDataType* prev_slice_xy, DoubleDataType* prev_slice_yz, DoubleDataType* prev_slice_xz,
                DataType* tile_data) {
            m_tile_offset = tile_offset;
            m_data = tile_data;
            m_prev_slice_xy = prev_slice_xy;
            m_prev_slice_yz = prev_slice_yz;
            m_prev_slice_xz = prev_slice_xz;
            m_tile_size = tile_size;

            for (unsigned int z = 1; z < m_tile_size; ++z) {
#pragma omp parallel for
                for (unsigned int y = 0; y < m_tile_size; ++y) {
                    for (unsigned int x = 0; x < m_tile_size; ++x) {
                        set_tile_data_value(
                                x, y, z,
                                get_tile_data_value(x, y, z) +
                                get_tile_data_value(x, y, z - 1));
                    }
                }
            }
            for (unsigned int y = 1; y < m_tile_size; ++y) {
#pragma omp parallel for
                for (unsigned int z = 0; z < m_tile_size; ++z) {
                    for (unsigned int x = 0; x < m_tile_size; ++x) {
                        set_tile_data_value(
                                x, y, z,
                                get_tile_data_value(x, y, z) +
                                get_tile_data_value(x, y - 1, z));
                    }
                }
            }
            for (unsigned int x = 1; x < m_tile_size; ++x) {
#pragma omp parallel for
                for (unsigned int z = 0; z < m_tile_size; ++z) {
                    for (unsigned int y = 0; y < m_tile_size; ++y) {
                        set_tile_data_value(
                                x, y, z,
                                get_tile_data_value(x, y, z) +
                                get_tile_data_value(x - 1, y, z));
                    }
                }
            }
        }

        inline DataType get_tile_data_value(size_t x, size_t y, size_t z) const {
            if (m_data.index() == 0) {  // 0 == DataType <-> sparse case
                return std::get<DataType>(m_data) * (x + 1) * (y + 1) * (z + 1);
            } else {  // 1 == DataType* <-> dense case
                size_t ptr_offset = z * m_tile_size * m_tile_size + y * m_tile_size + x;
                return std::get<DataType*>(m_data)[ptr_offset];
            }
        }

        inline void set_tile_data_value(size_t x, size_t y, size_t z, DataType value) {
            size_t ptr_offset = z * m_tile_size * m_tile_size + y * m_tile_size + x;
            std::get<DataType*>(m_data)[ptr_offset] = value;
        }

        DoubleDataType get_value(size_t x, size_t y, size_t z) const {
//            std::cout << "tile type " << m_type << " value " << x << " " << y << " " << z << " " << get_tile_data_value(x, y, z) << std::endl;
            return
                get_tile_data_value(x, y, z).template cast<double>()
                + m_prev_slice_xy[0]  // - Sxy(x1, y1, z1)
                - m_prev_slice_yz[z * (m_tile_size + 1)]  // + Syz( x1, y1, z2 )
                - m_prev_slice_xy[(y + 1) * (m_tile_size + 1)]  // + Sxy( x1, y2, z1 )
                - m_prev_slice_xy[x + 1]  // + Sxy( x2, y1, z1 )
                + m_prev_slice_xy[(y + 1) * (m_tile_size + 1) + x + 1]  // - Sxy( x2, y2, z1 )
                + m_prev_slice_xz[z * m_tile_size + x]  // - Sxz( x2, y1, z2 )
                + m_prev_slice_yz[z * (m_tile_size + 1) + y + 1]  // - Syz( x1, y2, z2 )
                + m_tile_offset.template cast<double>()
                ;
        }

        ~Tile() {
            delete[] m_prev_slice_xy;
            delete[] m_prev_slice_yz;
            delete[] m_prev_slice_xz;

            if (m_data.index() == 0) {  // 0 == DataType <-> sparse case
                // do nothing
            } else {  // 1 == DataType* <-> dense case
                auto data_ptr = std::get<DataType*>(m_data);
                delete[] data_ptr;
            }
        }

//        std::string toString(unsigned int level) {
//            std::ostringstream oss;
//            for(unsigned int i = 0; i<level; ++i) oss << "    ";
//            oss << value << " ["<<(int)splitplane << "]: "<<split;
//            if(!isLeaf()){
//                oss << std::endl;
//                oss << children[0].toString(level+1) << std::endl;
//                oss << children[1].toString(level+1) << std::endl;
//            }
//            return oss.str();
//        }
//
        size_t getSize() const {
            size_t size = sizeof(m_tile_offset) +
                    3 * sizeof(DoubleDataType*) +
                    sizeof(DoubleDataType) * ((m_tile_size + 1) * (m_tile_size + 1) + (m_tile_size + 1) * m_tile_size + m_tile_size * m_tile_size);

            if (m_data.index() == 0) {  // 0 == DataType <-> sparse case
                return size + sizeof(DataType);
            } else {  // 1 == DataType* <-> dense case
                auto ptr = std::get<DataType*>(m_data);
                return size + sizeof(ptr) + sizeof(DataType) * m_tile_size * m_tile_size * m_tile_size;
            }
        }

        std::variant<DataType, DataType*> m_data;
        UIntDataType m_tile_offset;
        DoubleDataType* m_prev_slice_xy;  // (m_tile_size + 1) * (m_tile_size + 1)
        DoubleDataType* m_prev_slice_yz;  // (m_tile_size + 1) * m_tile_size
        DoubleDataType* m_prev_slice_xz;  // m_tile_size * m_tile_size
        unsigned short m_tile_size;
    };

    Eigen::Tensor<Tile, 3, Eigen::RowMajor> m_tiles_tensor;
    BoundingBox m_dataBBox;
    unsigned short m_tile_size;
    bool m_alignment_z_centered;
    #ifdef SAT_TILE_TREE_STATS
    long int __STATS_requests = 0;
    long int __STATS_cache_hits = 0;
    long int __STATS_cache_collisions = 0;
    #endif // SAT_TILE_TREE_STATS

    void constructTree(const BoundingBox& dimensions, const TensorType& data) {
        vec3 volume_dimensions = dimensions.upper - dimensions.lower;
        vec3 tile_tree_dimensions = (volume_dimensions + vec3(m_tile_size - 1, m_tile_size - 1, m_tile_size - 1)) / m_tile_size;
        m_tiles_tensor = Eigen::Tensor<Tile, 3, Eigen::RowMajor>(tile_tree_dimensions[2], tile_tree_dimensions[1], tile_tree_dimensions[0]);

        DoubleDataType sat_tile_offset_double;
        UIntDataType sat_tile_offset;
        bool all_values_equal_flag;
        DataType first_tile_value;
        int volume_index_offset_x, volume_index_offset_y, volume_index_offset_z;

        for (int tile_z = 0; tile_z < tile_tree_dimensions[2]; ++tile_z) {
            for (int tile_y = 0; tile_y < tile_tree_dimensions[1]; ++tile_y) {
                for (int tile_x = 0; tile_x < tile_tree_dimensions[0]; ++tile_x) {
                    all_values_equal_flag = true;

                    volume_index_offset_x = tile_x * m_tile_size;
                    volume_index_offset_y = tile_y * m_tile_size;
                    volume_index_offset_z = tile_z * m_tile_size;

                    first_tile_value = data(
                            static_cast<long>(volume_index_offset_z),
                            static_cast<long>(volume_index_offset_y),
                            static_cast<long>(volume_index_offset_x));

                    auto tile_data = new DataType[m_tile_size * m_tile_size * m_tile_size];

                    for (int z = 0; z < m_tile_size; ++z) {
                        for (int y = 0; y < m_tile_size; ++y) {
                            for (int x = 0; x < m_tile_size; ++x) {
                                DataType value;
                                if (
                                        z + volume_index_offset_z >= volume_dimensions[2] ||
                                        y + volume_index_offset_y >= volume_dimensions[1] ||
                                        x + volume_index_offset_x >= volume_dimensions[0]
                                ) {
                                    value = DataType::Zero();
                                } else {
                                    value = data(
                                            static_cast<long>(volume_index_offset_z + z),
                                            static_cast<long>(volume_index_offset_y + y),
                                            static_cast<long>(volume_index_offset_x + x));
                                    if (value != first_tile_value) {
                                        all_values_equal_flag = false;
                                    }
                                }
                                size_t ptr_offset = z * m_tile_size * m_tile_size + y * m_tile_size + x;
                                tile_data[ptr_offset] = value;
                                // std::cout << z << " " << y << " " << x << " " << value << std::endl;
                            }
                        }
                    }

                    if (tile_x > 0 && tile_y > 0 && tile_z > 0) {
                        sat_tile_offset_double = m_tiles_tensor(
                                static_cast<long>(tile_z - 1),
                                static_cast<long>(tile_y - 1),
                                static_cast<long>(tile_x - 1))
                                        .get_value(m_tile_size - 1, m_tile_size - 1, m_tile_size - 1);
                    } else {
                        sat_tile_offset_double = DoubleDataType::Zero();
                    }
                    sat_tile_offset = sat_tile_offset_double.template cast<unsigned int>();
                    // std::cout << "sat tile offset " << sat_tile_offset << std::endl;

                    auto prev_slice_xy = new DoubleDataType[(m_tile_size + 1) * (m_tile_size + 1)];
                    auto prev_slice_yz = new DoubleDataType[(m_tile_size + 1) * m_tile_size];
                    auto prev_slice_xz = new DoubleDataType[m_tile_size * m_tile_size];

                    DoubleDataType offset_to_substract = sat_tile_offset.template cast<double>();
                    prev_slice_xy[0] = sat_tile_offset_double - offset_to_substract;
                    if (tile_z > 0) {
                        auto& tile_tensor = m_tiles_tensor(
                                static_cast<long>(tile_z - 1),
                                static_cast<long>(tile_y),
                                static_cast<long>(tile_x));
                        for (unsigned short y = 1; y < m_tile_size + 1; ++y) {
                            for (unsigned short x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[y * (m_tile_size + 1) + x] =
                                        tile_tensor.get_value(x - 1, y - 1, m_tile_size - 1) - offset_to_substract;
                            }
                        }
                        if (tile_y > 0) {
                            auto& tile_tensor2 = m_tiles_tensor(
                                    static_cast<long>(tile_z - 1),
                                    static_cast<long>(tile_y - 1),
                                    static_cast<long>(tile_x));
                            for (unsigned short x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[x] = tile_tensor2.get_value(x - 1, m_tile_size - 1, m_tile_size - 1) - offset_to_substract;
                            }
                        } else {
                            for (unsigned short x = 1; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[x] = DoubleDataType::Zero();
                            }
                        }
                        if (tile_x > 0) {
                            auto& tile_tensor3 = m_tiles_tensor(
                                    static_cast<long>(tile_z - 1),
                                    static_cast<long>(tile_y),
                                    static_cast<long>(tile_x - 1));
                            for (unsigned short y = 1; y < m_tile_size + 1; ++y) {
                                prev_slice_xy[y * (m_tile_size + 1)] = tile_tensor3.get_value(m_tile_size - 1, y - 1, m_tile_size - 1) - offset_to_substract;
                            }
                        } else {
                            for (unsigned short y = 1; y < m_tile_size + 1; ++y) {
                                prev_slice_xy[y * (m_tile_size + 1)] = DoubleDataType::Zero();
                            }
                        }
                    } else {
                        for (unsigned short y = 0; y < m_tile_size + 1; ++y) {
                            for (unsigned short x = 0; x < m_tile_size + 1; ++x) {
                                prev_slice_xy[y * (m_tile_size + 1) + x] = DoubleDataType::Zero();
                            }
                        }
                    }

                    if (tile_x > 0) {
                        auto& tile_tensor = m_tiles_tensor(
                                static_cast<long>(tile_z),
                                static_cast<long>(tile_y),
                                static_cast<long>(tile_x - 1));
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short y = 1; y < m_tile_size + 1; ++y) {
                                prev_slice_yz[z * (m_tile_size + 1) + y] = tile_tensor.get_value(m_tile_size - 1, y - 1, z) - offset_to_substract;
                            }
                        }
                        if (tile_y > 0) {
                            auto& tile_tensor2 = m_tiles_tensor(
                                    static_cast<long>(tile_z),
                                    static_cast<long>(tile_y - 1),
                                    static_cast<long>(tile_x - 1));
                            for (unsigned short z = 0; z < m_tile_size; ++z) {
                                prev_slice_yz[z * (m_tile_size + 1)] = tile_tensor2.get_value(m_tile_size - 1, m_tile_size - 1, z) - offset_to_substract;
                            }
                        } else {
                            for (unsigned short z = 0; z < m_tile_size; ++z) {
                                prev_slice_yz[z * (m_tile_size + 1)] = DoubleDataType::Zero();
                            }
                        }
                    } else {
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short y = 0; y < m_tile_size + 1; ++y) {
                                prev_slice_yz[z * (m_tile_size + 1) + y] = DoubleDataType::Zero();
                            }
                        }
                    }

                    if (tile_y > 0) {
                        auto& tile_tensor = m_tiles_tensor(
                                static_cast<long>(tile_z),
                                static_cast<long>(tile_y - 1),
                                static_cast<long>(tile_x));
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short x = 0; x < m_tile_size; ++x) {
                                prev_slice_xz[z * m_tile_size + x] = tile_tensor.get_value(x, m_tile_size - 1, z) - offset_to_substract;
                            }
                        }
                    } else {
                        for (unsigned short z = 0; z < m_tile_size; ++z) {
                            for (unsigned short x = 0; x < m_tile_size; ++x) {
                                prev_slice_xz[z * m_tile_size + x] = DoubleDataType::Zero();
                            }
                        }
                    }

                    if (all_values_equal_flag) {
                        delete[] tile_data;
                        // std::cout << "first_tile_value " << first_tile_value << std::endl;
                        m_tiles_tensor(static_cast<long>(tile_z), static_cast<long>(tile_y), static_cast<long>(tile_x))
                        .tile_set_data(m_tile_size, sat_tile_offset, prev_slice_xy, prev_slice_yz, prev_slice_xz, first_tile_value);
                    } else {
                        m_tiles_tensor(static_cast<long>(tile_z), static_cast<long>(tile_y), static_cast<long>(tile_x))
                        .tile_set_data(m_tile_size, sat_tile_offset, prev_slice_xy, prev_slice_yz, prev_slice_xz, tile_data);
                    }
                }
            }
        }
    }

    inline DoubleDataType get_sat_value(int x, int y, int z) {
        if (x < 0 || y < 0 || z < 0) {
            return DoubleDataType::Zero();
        }
        return m_tiles_tensor(
                static_cast<long>(z / m_tile_size),
                static_cast<long>(y / m_tile_size),
                static_cast<long>(x / m_tile_size)
                ).get_value(
                        x - x / m_tile_size * m_tile_size,
                        y - y / m_tile_size * m_tile_size,
                        z - z / m_tile_size * m_tile_size);
    }

    inline DoubleDataType get_sat_value(int x, int y, int z, QueryCacheType* localCache, unsigned int rel_x, unsigned int rel_y, unsigned int rel_z) {
        if (x < 0 || y < 0 || z < 0) {
            return DoubleDataType::Zero();
        }
        DoubleDataType value;
        #ifdef SAT_TILE_TREE_STATS
        ++__STATS_requests;
        #endif
        size_t index = localCache->convertIndex(rel_x, rel_y, rel_z);
        if(!localCache->contains(index, value)){
            value = m_tiles_tensor(
                    static_cast<long>(z / m_tile_size),
                    static_cast<long>(y / m_tile_size),
                    static_cast<long>(x / m_tile_size)
                    ).get_value(
                            x - x / m_tile_size * m_tile_size,
                            y - y / m_tile_size * m_tile_size,
                            z - z / m_tile_size * m_tile_size);
            localCache->insert(index, value);
            // #ifdef SAT_TILE_TREE_STATS
            // if(!insert.second) 
            // { ++__STATS_cache_collisions; }
            // #endif
        }
        #ifdef SAT_TILE_TREE_STATS
        else { ++__STATS_cache_hits; }
        #endif
        return value;
    }

    inline DataType get_sum_from_sat(const BoundingBox& queryBox, QueryCacheType* localCache, vec3u rel_offset) {
        int x1, x2, y1, y2, z1, z2;
        x1 = queryBox.lower[0] - 1;
        x2 = queryBox.upper[0] - 1;
        y1 = queryBox.lower[1] - 1;
        y2 = queryBox.upper[1] - 1;
        z1 = queryBox.lower[2] - 1;
        z2 = queryBox.upper[2] - 1;

        return (
                #ifdef SAT_TILE_TREE_USE_CACHING
                  get_sat_value(x2, y2, z2, localCache, rel_offset[0] + 1, rel_offset[1] + 1, rel_offset[2] + 1)
                - get_sat_value(x1, y2, z2, localCache, rel_offset[0]    , rel_offset[1] + 1, rel_offset[2] + 1)
                + get_sat_value(x2, y1, z1, localCache, rel_offset[0] + 1, rel_offset[1]    , rel_offset[2]    )
                + get_sat_value(x1, y2, z1, localCache, rel_offset[0]    , rel_offset[1] + 1, rel_offset[2]    )
                - get_sat_value(x2, y1, z2, localCache, rel_offset[0] + 1, rel_offset[1]    , rel_offset[2] + 1)
                - get_sat_value(x2, y2, z1, localCache, rel_offset[0] + 1, rel_offset[1] + 1, rel_offset[2]    )
                + get_sat_value(x1, y1, z2, localCache, rel_offset[0]    , rel_offset[1]    , rel_offset[2] + 1)
                - get_sat_value(x1, y1, z1, localCache, rel_offset[0]    , rel_offset[1]    , rel_offset[2]    )
                #else
                  get_sat_value(x2, y2, z2) - get_sat_value(x1, y2, z2)
                + get_sat_value(x2, y1, z1) + get_sat_value(x1, y2, z1)
                - get_sat_value(x2, y1, z2) - get_sat_value(x2, y2, z1)
                + get_sat_value(x1, y1, z2) - get_sat_value(x1, y1, z1)
                #endif
        ).template cast<typename DataType::Scalar>();
    }

    void fill_one_stencil(
            typename DataType::Scalar* ptr,
            const std::vector<std::tuple<int, int>>& scale_levels,
            unsigned int patch_size_x, unsigned int patch_size_y, unsigned int patch_size_z,
            int coord_z, int coord_y, int coord_x
    ) {
        QueryCacheType localCache({patch_size_x+1, patch_size_y+1, patch_size_z+1});
        for (unsigned int scale_level_idx = 0; scale_level_idx < scale_levels.size(); ++scale_level_idx) {
            auto scale_level = scale_levels[scale_level_idx];

            auto scale_kernel_x = std::get<0>(scale_level);
            auto scale_kernel_y = std::get<0>(scale_level);
            auto scale_kernel_z = std::get<1>(scale_level);

            int patch_start_x = coord_x - patch_size_x * scale_kernel_x / 2;
            int patch_start_y = coord_y - patch_size_y * scale_kernel_y / 2;
            int patch_start_z = coord_z;
            if (m_alignment_z_centered){
                patch_start_z -= patch_size_z * scale_kernel_z / 2;
            }
            if(scale_level_idx != 0)
                localCache.clear();

            int z1, z2, y1, y2, x1, x2;

            for (unsigned int z = 0; z < patch_size_z; ++z) {
                z1 = patch_start_z + z * scale_kernel_z;
                z2 = z1 + scale_kernel_z;
                for (unsigned int y = 0; y < patch_size_y; ++y) {
                    y1 = patch_start_y + y * scale_kernel_y;
                    y2 = y1 + scale_kernel_y;
                    for (unsigned int x = 0; x < patch_size_x; ++x) {
                        x1 = patch_start_x + x * scale_kernel_x;
                        x2 = x1 + scale_kernel_x;

                        auto value = queryAverage(BoundingBox({x1, y1, z1}, {x2, y2, z2}), &localCache, {x,y,z});

                        size_t ptr_idx =
                                scale_level_idx * patch_size_z * patch_size_y * patch_size_x * Dimensionality +
                                z * patch_size_y * patch_size_x * Dimensionality +
                                y * patch_size_x * Dimensionality +
                                x * Dimensionality;
                        for (size_t c = 0; c < Dimensionality; ++c) {
                            ptr[ptr_idx + c] = value[c];
                        }
                    }
                }
            }
        }
    }
};


PYBIND11_MODULE(sat_tile_tree, m) {
    py::class_<SATTileTree<Eigen::Matrix<float, 1, 1>>>(m, "SATTileTree")
    .def(py::init<const py::array_t<float>&, unsigned short, bool>(), py::arg("volume"), py::arg("tile_size") = 32, py::arg("alignment_z_centered") = true)
    .def("query_average", &SATTileTree<Eigen::Matrix<float, 1, 1>>::queryAverageSlice)
    .def("get_sat_value", &SATTileTree<Eigen::Matrix<float, 1, 1>>::get_sat_value_py)
    .def("get_sat", &SATTileTree<Eigen::Matrix<float, 1, 1>>::get_sat)
    .def("generate_scales", &SATTileTree<Eigen::Matrix<float, 1, 1>>::generate_scales)
    .def("generate_scales_list", &SATTileTree<Eigen::Matrix<float, 1, 1>>::generate_scales_list)
    .def("toNumpy", &SATTileTree<Eigen::Matrix<float, 1, 1>>::toNumpy)
    .def("print", &SATTileTree<Eigen::Matrix<float, 1, 1>>::print)
    .def("size", &SATTileTree<Eigen::Matrix<float, 1, 1>>::size);

    py::class_<SATTileTree<Eigen::Vector2d>>(m, "SATTileTree2D")
    .def(py::init<const py::array_t<float>&, unsigned short, bool>(), py::arg("volume"), py::arg("tile_size") = 32, py::arg("alignment_z_centered") = true)
    .def("query_average", &SATTileTree<Eigen::Vector2d>::queryAverageSlice)
    .def("get_sat_value", &SATTileTree<Eigen::Vector2d>::get_sat_value_py)
    .def("get_sat", &SATTileTree<Eigen::Vector2d>::get_sat)
    .def("generate_scales", &SATTileTree<Eigen::Vector2d>::generate_scales)
    .def("generate_scales_list", &SATTileTree<Eigen::Vector2d>::generate_scales_list)
    .def("toNumpy", &SATTileTree<Eigen::Vector2d>::toNumpy)
    .def("print", &SATTileTree<Eigen::Vector2d>::print)
    .def("size", &SATTileTree<Eigen::Vector2d>::size);
}
