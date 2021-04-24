#define NDEBUG

#include <cassert>
#include <Eigen/Dense>
#include <unsupported/Eigen/CXX11/Tensor>

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <iostream>

#include "bounding_box.h"
#include "gtest_float.h"
#include "minipool.hpp"

namespace py = pybind11;

using vec3 = Eigen::Vector3i;

#if defined(__GNUC__)
#define FINLINE                inline __attribute__((always_inline))
#define NOINLINE               __attribute__((noinline))
#define EXPECT_TAKEN(a)        __builtin_expect(!!(a), true)
#define EXPECT_NOT_TAKEN(a)    __builtin_expect(!!(a), false)
#elif defined(__MSVC__)
#define FINLINE                __forceinline
#define NOINLINE               __declspec(noinline)
#define MM_ALIGN16             __declspec(align(16))
#define EXPECT_TAKEN(a)        (a)
#define EXPECT_NOT_TAKEN(a)    (a)
#else
#error Unsupported compiler!
#endif

//#define DEBUG_TREE_COUNTERS
#ifdef DEBUG_TREE_COUNTERS
using CounterType = uint64_t;
static CounterType inner_node_response = 0;
static CounterType leaf_node_response = 0;
static CounterType outside_response = 0;
static CounterType split_response = 0;
static CounterType recursive_response = 0;
static CounterType traversals_per_inner_node_response = 0;
static CounterType traversals_per_leaf_node_response = 0;
static CounterType traversals_per_outside_response = 0;
static CounterType traversals_per_split_response = 0;
static CounterType current_depth = 0;
static CounterType max_depth = 0;

static std::string getStats(){
    std::ostringstream oss;
    oss << "inner_node_response " << inner_node_response << std::endl;
    oss << "leaf_node_response " << leaf_node_response << std::endl;
    oss << "outside_response " << outside_response << std::endl;
    oss << "split_response " << split_response << std::endl;
    oss << "recursive_response " << recursive_response << std::endl;
    oss << "traversals_per_inner_node_response " << traversals_per_inner_node_response << std::endl;
    oss << "traversals_per_leaf_node_response " << traversals_per_leaf_node_response << std::endl;
    oss << "traversals_per_outside_response " << traversals_per_outside_response << std::endl;
    oss << "traversals_per_split_response " << traversals_per_split_response << std::endl;
    oss << "Average Depth inner_node_response " << ((inner_node_response > 0) ? double(traversals_per_inner_node_response) / inner_node_response : 0 ) << std::endl;
    oss << "Average Depth leaf_node_response " << ((leaf_node_response > 0) ? double(traversals_per_leaf_node_response) / leaf_node_response : 0 ) << std::endl;
    oss << "Average Depth outside_response " << ((outside_response > 0) ? double(traversals_per_outside_response) / outside_response : 0 ) << std::endl;
    oss << "Average Depth split_response " << ((split_response > 0) ? double(traversals_per_split_response) / split_response : 0 ) << std::endl;
    oss << "current_depth " << current_depth << std::endl;
    oss << "max_depth " << max_depth << std::endl;
    return oss.str();
}
#endif


template <class T>
inline T upper_power_of_two(T v)
{
    v--;
    v |= v >> 1;
    v |= v >> 2;
    v |= v >> 4;
    v |= v >> 8;
    v |= v >> 16;
    v++;
    return v;
}


template <class _DataType>
class AverageKDTree {
    using DataType = _DataType;
    using ComparisonType = FloatingPoint< typename DataType::Scalar >;
    using ArrayType = py::array_t<typename AverageKDTree<_DataType>::DataType::Scalar, py::array::c_style | py::array::forcecast>;
    using TensorType = Eigen::TensorMap<Eigen::Tensor<DataType, 3, Eigen::RowMajor>>;

    enum {
        Dimensionality = BoundingBox::Scalar(DataType::RowsAtCompileTime),
    };
    //static_assert(typename AverageKDTree<_DataType>::DataType::IsVectorAtCompileTime);
public:
    AverageKDTree(const ArrayType& volume, bool alignment_z_centered, unsigned int subtree_check_threshold) : 
        m_memory_pool(minipool<NodeArray>(256)), m_alignment_z_centered(alignment_z_centered), m_subtree_check_threshold(subtree_check_threshold) {
        
        py::buffer_info buffer = volume.request();

        if (buffer.ndim != 4)
            throw std::runtime_error("Number of dimensions must be 4");

        if (buffer.shape[3] != Dimensionality){
            std::ostringstream err;
            err << "Input array of shape [..., "<< buffer.shape[3]
                << "] does not match Dimensionality "<< Dimensionality;
            throw std::runtime_error(err.str());
        }
        
        auto volume_tensor = TensorType( (DataType*) buffer.ptr, buffer.shape[0], buffer.shape[1], buffer.shape[2]);

        m_dataBBox = BoundingBox({0,0,0}, {buffer.shape[2], buffer.shape[1], buffer.shape[0]});
        //bottom up construction of the tree
        //use recursion + stack
        m_root = new Node();
        constructTree(m_root, m_dataBBox, volume_tensor, true);
        m_memory_pool.garbage_collect();
    }
    ~AverageKDTree(){
        if(m_root != nullptr){
            if(!m_root->isLeaf()){
                m_memory_pool.free(reinterpret_cast<NodeArray*>(m_root->children));
            }
            delete m_root;
        }
    }

    DataType queryAverage(const BoundingBox& queryBox){
        #ifdef DEBUG_TREE_COUNTERS
        current_depth = 0;
        #endif
        if (m_dataBBox.contains(queryBox)){
            return queryNode(m_dataBBox, m_root, queryBox).value;
        } else {
            //handle virtual zeros around
            #ifdef DEBUG_TREE_COUNTERS
            ++outside_response;
            #endif // DEBUG_TREE_COUNTERS
            auto query = queryNode(m_dataBBox, m_root, m_dataBBox.overlap(queryBox));
            return query.value * query.volume / queryBox.volume();
        }
    }

    DataType queryAverageSlice(py::slice x_slice, py::slice y_slice, py::slice z_slice){
        size_t x1, x2, y1, y2, z1, z2, step, slicelength;
        x_slice.compute(m_dataBBox.upper[0], &x1, &x2, &step, &slicelength);
        y_slice.compute(m_dataBBox.upper[1], &y1, &y2, &step, &slicelength);
        z_slice.compute(m_dataBBox.upper[2], &z1, &z2, &step, &slicelength);

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

        return queryAverage(queryBox);
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

    ArrayType toNumpy() {
        /* No pointer is passed, so NumPy will allocate the buffer */
        auto result = ArrayType({m_dataBBox.upper[2], m_dataBBox.upper[1], m_dataBBox.upper[0], BoundingBox::Scalar(Dimensionality)});
        writeToArray(m_dataBBox, m_root, result);
        return result;
    }

    void print() const {
        std::cout << m_root->toString(0);
    }

    long unsigned int size() const {
        return m_root->getSize();
    }
private:
    struct Node {
        Node() : children(nullptr){}

        Node(DataType _value, BoundingBox::Scalar _split, char _splitplane)
        : value(_value), split(_split), splitplane(_splitplane), children(nullptr) { }

        // leaf constructor
        Node(DataType _value)
        : value(_value), children(nullptr) { }

        ~Node(){
            if(!isLeaf()){
                // children will be implicitly deallocated when m_memory_pool will be destroyed
                // m_memory_pool.free(children);
            }
        }

        inline bool isLeaf() const {
            return children == nullptr;
        }

        std::string toString(unsigned int level) {
            std::ostringstream oss;
            for(unsigned int i = 0; i<level; ++i) oss << "    ";
            oss << value << " ["<<(int)splitplane << "]: "<<split;
            if(!isLeaf()){
                oss << std::endl;
                oss << children[0].toString(level+1) << std::endl;
                oss << children[1].toString(level+1) << std::endl;
            }
            return oss.str();
        }

        size_t getSize() {
            return sizeof(Node) + (isLeaf() ? 0 : children[0].getSize() + children[1].getSize());
        }

        DataType value;
        BoundingBox::Scalar split;
        char splitplane;
        Node* children; // always 2 children, access the second with operator[]
    };

    struct NodeArray {
        Node array[2];
    };

    struct QueryRecord{
        QueryRecord(DataType _value, BoundingBox::VolumeType _volume) :
            value(_value), volume(_volume) {}
        DataType value;
        BoundingBox::VolumeType volume;
    };

    void constructTree(Node* node, BoundingBox dimensions, const TensorType& data, bool subtree_possibly_constant = true){
        if(EXPECT_TAKEN(dimensions.volume() > 1)){
            // get max dimension
            auto max_dimension = dimensions.maxExtend();
            auto max_extend = dimensions.extend()[max_dimension];

            if(subtree_possibly_constant && max_extend <= m_subtree_check_threshold) {
                bool differs = false;
                auto value = data(dimensions.lower[2], dimensions.lower[1], dimensions.lower[0]);
                for (auto z = dimensions.lower[2]; z < dimensions.upper[2]; ++z) {
                    for (auto y = dimensions.lower[1]; y < dimensions.upper[1]; ++y) {
                        for (auto x = dimensions.lower[0]; x < dimensions.upper[0]; ++x) {
                            differs = !equals(value, data(z,y,x));
                            if(differs)
                                break;
                        }
                        if(differs)
                            break;
                    }
                    if(differs)
                        break;
                }
                if(!differs){
                    //we can skip recursion, all values are the same.
                    node->children = nullptr;
                    node->value = value;
                    return;
                }
                subtree_possibly_constant = false;
            }

            BoundingBox::Scalar split = dimensions.lower[max_dimension] + max_extend / 2;
            //*node = Node(0, split, max_dimension);
            node->split = split;
            node->splitplane = max_dimension;
            // allocate an array of 2 Node objects
            node->children = reinterpret_cast<Node*>(m_memory_pool.alloc()); // equals new Node[2];
            node->children[1].children = nullptr; // initialize also the 2nd element
            auto split_boxes = dimensions.split(max_dimension, split);

            constructTree(&node->children[0], std::get<0>(split_boxes), data, subtree_possibly_constant);
            constructTree(&node->children[1], std::get<1>(split_boxes), data, subtree_possibly_constant);

            assert(std::get<0>(split_boxes).volume() + std::get<1>(split_boxes).volume() == dimensions.volume());
            node->value = (node->children[0].value * std::get<0>(split_boxes).volume() 
                        + node->children[1].value * std::get<1>(split_boxes).volume()) / dimensions.volume();
            if(EXPECT_NOT_TAKEN(node->children[0].isLeaf() && node->children[1].isLeaf() && equals(node->value, node->children[0].value) && equals(node->value, node->children[1].value))){
                m_memory_pool.free(reinterpret_cast<NodeArray*>(node->children));
                node->children = nullptr;
            }
        } else {
            // leaf node
            node->value = data(dimensions.lower[2], dimensions.lower[1], dimensions.lower[0]);
        }
    }

    void fill_one_stencil(
            typename DataType::Scalar* ptr,
            const std::vector<std::tuple<int, int>>& scale_levels,
            int patch_size_x, int patch_size_y, int patch_size_z,
            int coord_z, int coord_y, int coord_x
    ) {
        for (auto scale_level_idx = 0; scale_level_idx < scale_levels.size(); ++scale_level_idx) {
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

            int z1, z2, y1, y2, x1, x2;

            for (auto z = 0; z < patch_size_z; ++z) {
                z1 = patch_start_z + z * scale_kernel_z;
                z2 = z1 + scale_kernel_z;
                for (auto y = 0; y < patch_size_y; ++y) {
                    y1 = patch_start_y + y * scale_kernel_y;
                    y2 = y1 + scale_kernel_y;
                    for (auto x = 0; x < patch_size_x; ++x) {
                        x1 = patch_start_x + x * scale_kernel_x;
                        x2 = x1 + scale_kernel_x;

                        auto value = queryAverage(BoundingBox({x1, y1, z1}, {x2, y2, z2}));

                        for (unsigned char c = 0; c < Dimensionality; ++c) {
                            size_t ptr_idx =
                                    scale_level_idx * patch_size_z * patch_size_y * patch_size_x * Dimensionality +
                                    z * patch_size_y * patch_size_x * Dimensionality +
                                    y * patch_size_x * Dimensionality +
                                    x * Dimensionality + c;
                            ptr[ptr_idx] = value[c];
                        }
                    }
                }
            }
        }
    }

    QueryRecord queryNode(BoundingBox dimensions, const Node* node, BoundingBox queryBox) const {
        #ifdef DEBUG_TREE_COUNTERS
        ++current_depth;
        if(EXPECT_NOT_TAKEN(current_depth > max_depth)){
            max_depth = current_depth;
        }
        #endif
        // is that really necessary??
        if(queryBox.isNegative() || m_dataBBox.outside(queryBox)){
            #ifdef DEBUG_TREE_COUNTERS
            ++outside_response;
            traversals_per_outside_response += current_depth;
            #endif
            return QueryRecord(DataType(0),0);
        }
        if(queryBox.contains(dimensions)) {
            #ifdef DEBUG_TREE_COUNTERS
            if(node->isLeaf()){
                ++leaf_node_response;
                traversals_per_leaf_node_response += current_depth;
            } else {
                ++inner_node_response;
                traversals_per_inner_node_response += current_depth;
            }
            #endif
            return QueryRecord(node->value, dimensions.volume());
        } else {
            if (EXPECT_NOT_TAKEN(node->isLeaf())) {
                #ifdef DEBUG_TREE_COUNTERS
                ++leaf_node_response;
                traversals_per_leaf_node_response += current_depth;
                #endif
                return QueryRecord(node->value, queryBox.volume());
            }

            auto split_boxes = dimensions.split(node->splitplane, node->split);
            if(std::get<0>(split_boxes).contains(queryBox)){
                #ifdef DEBUG_TREE_COUNTERS
                ++recursive_response;
                auto rv = queryNode(std::get<0>(split_boxes), &node->children[0], queryBox);
                --current_depth;
                return rv;
                #else
                return queryNode(std::get<0>(split_boxes), &node->children[0], queryBox);
                #endif
            }
            if(std::get<1>(split_boxes).contains(queryBox)){
                #ifdef DEBUG_TREE_COUNTERS
                ++recursive_response;
                auto rv = queryNode(std::get<1>(split_boxes), &node->children[1], queryBox);
                --current_depth;
                return rv;
                #else
                return queryNode(std::get<1>(split_boxes), &node->children[1], queryBox);
                #endif
            }
            #ifdef DEBUG_TREE_COUNTERS
            ++split_response;
            traversals_per_split_response += current_depth;
            #endif
            // we need to handle the query on this level
            auto leftQR = queryNode(std::get<0>(split_boxes), &node->children[0], std::get<0>(split_boxes).overlap(queryBox));
            #ifdef DEBUG_TREE_COUNTERS
            --current_depth;
            #endif
            auto rightQR = queryNode(std::get<1>(split_boxes), &node->children[1], std::get<1>(split_boxes).overlap(queryBox));
            #ifdef DEBUG_TREE_COUNTERS
            --current_depth;
            #endif

            BoundingBox::VolumeType processed_volume = leftQR.volume + rightQR.volume;
            DataType result = (leftQR.value * leftQR.volume + rightQR.value * rightQR.volume) / processed_volume;
            return QueryRecord(result, processed_volume);
        }
    }

    void writeToArray(BoundingBox dimensions, const Node* node, ArrayType & array) const{
        assert(dimensions.lower < dimensions.upper);
        if(EXPECT_NOT_TAKEN(node->isLeaf())){
            for(size_t x = dimensions.lower[0]; x < dimensions.upper[0]; ++x){
                for(size_t y = dimensions.lower[1]; y < dimensions.upper[1]; ++y){
                    for(size_t z = dimensions.lower[2]; z < dimensions.upper[2]; ++z){
                        array[py::make_tuple(z,y,x)] = node->value;
                    }
                }
            }
        } else {
            auto split_boxes = dimensions.split(node->splitplane, node->split);
            if(!std::get<0>(split_boxes).isNegative()){
                writeToArray(std::get<0>(split_boxes), &node->children[0], array);
            }
            if(!std::get<1>(split_boxes).isNegative()){
                writeToArray(std::get<1>(split_boxes), &node->children[1], array);
            }
        }
    }

    /**
     * @brief general purpose implementation of equality comparison
     * @details overwriten for special types
     **/
    bool equals(DataType one, DataType other){
        return one == other;
    }

    Node* m_root;
    minipool<NodeArray> m_memory_pool;
    BoundingBox m_dataBBox;
    bool m_alignment_z_centered;
    unsigned int m_subtree_check_threshold;
};

// template<>
// bool AverageKDTree<float>::equals(float one, float other){
//     return ComparisonType(one).AlmostEquals(ComparisonType(other));
// }

// template<>
// bool AverageKDTree<double>::equals(double one, double other){
//     return ComparisonType(one).AlmostEquals(ComparisonType(other));
// }

// template<typename type, int size>
// bool AverageKDTree< Eigen::Matrix<type, size, 1> >::equals(Eigen::Matrix<type, size, 1> one, Eigen::Matrix<type, size, 1> other){
//     return (one.array() == other.array()).all();
// }
template<>
bool AverageKDTree< Eigen::Vector2f >::equals(Eigen::Vector2f one, Eigen::Vector2f other){
    return one.isApprox(other);
}

PYBIND11_MODULE(average_octree, m) {
    py::class_<AverageKDTree<Eigen::Matrix<float, 1, 1>>>(m, "AverageKDTree")
    .def(py::init<const py::array_t<float>&, bool, unsigned int>(), py::arg("volume"), py::arg("alignment_z_centered") = true, py::arg("subtree_check_threshold") = 32)
    .def("query_average", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::queryAverageSlice)
    .def("generate_scales", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::generate_scales)
    .def("generate_scales_list", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::generate_scales_list)
    .def("toNumpy", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::toNumpy)
    .def("print", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::print)
    #ifdef DEBUG_TREE_COUNTERS
    .def_static("getStats", &getStats)
    #endif // DEBUG_TREE_COUNTERS
    .def("size", &AverageKDTree<Eigen::Matrix<float, 1, 1>>::size);

    py::class_<AverageKDTree<Eigen::Vector2f>>(m, "AverageKDTree2D")
    .def(py::init<const py::array_t<float>&, bool, unsigned int>(), py::arg("volume"), py::arg("alignment_z_centered") = true, py::arg("subtree_check_threshold") = 32)
    .def("query_average", &AverageKDTree<Eigen::Vector2f>::queryAverageSlice)
    .def("generate_scales", &AverageKDTree<Eigen::Vector2f>::generate_scales)
    .def("generate_scales_list", &AverageKDTree<Eigen::Vector2f>::generate_scales_list)
    .def("toNumpy", &AverageKDTree<Eigen::Vector2f>::toNumpy)
    .def("print", &AverageKDTree<Eigen::Vector2f>::print)
    #ifdef DEBUG_TREE_COUNTERS
    .def_static("getStats", &getStats)
    #endif // DEBUG_TREE_COUNTERS
    .def("size", &AverageKDTree<Eigen::Vector2f>::size);

}
