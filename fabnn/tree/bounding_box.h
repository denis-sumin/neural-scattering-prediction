//
// Created by Denis Sumin on 30.11.2019.
//

#ifndef FABRICATIONENHANCEMENT_BOUNDING_BOX_H
#define FABRICATIONENHANCEMENT_BOUNDING_BOX_H

#include <Eigen/Dense>

#endif //FABRICATIONENHANCEMENT_BOUNDING_BOX_H


using vec3 = Eigen::Vector3i;
using vec3u = Eigen::Matrix<unsigned int, 3, 1>;


bool operator < (const vec3 &a, const vec3 &b) {
    return a[0] < b[0] && a[1] < b[1] && a[2] < b[2];
}

bool operator <= (const vec3 &a, const vec3 &b) {
    return a[0] <= b[0] && a[1] <= b[1] && a[2] <= b[2];
}


/// @brief Return the index [0,1,2] of the largest value in a 3D vector.
/// @note This methods assumes operator[] exists and avoids branching.
/// @details If two components of the input vector are equal and larger than the
/// third component, the largest index of the two is always returned.
/// If all three vector components are equal the largest index, i.e. 2, is
/// returned. In other words the return value corresponds to the largest index
/// of the largest vector components.
template<typename Vec3T>
size_t
MaxIndex(const Vec3T& v)
{
#ifndef _MSC_VER // Visual C++ doesn't guarantee thread-safe initialization of local statics
    static
#endif
    const size_t hashTable[8] = { 2, 1, 9, 1, 2, 9, 0, 0 };//9 is a dummy value
    const size_t hashKey =
            ((v[0] > v[1]) << 2) + ((v[0] > v[2]) << 1) + (v[1] > v[2]);// ?*4+?*2+?*1
    return hashTable[hashKey];
}


struct BoundingBox {
    using VolumeType = Eigen::Matrix<int, 3, 1, 0, 3, 1>::Scalar;
    using Scalar = Eigen::Matrix<int, 3, 1, 0, 3, 1>::Scalar;

    BoundingBox() : lower(0, 0, 0), upper(0, 0, 0) {}
    BoundingBox(vec3 _lower, vec3 _upper) :
            lower(_lower), upper(_upper) {
        assert(lower <= upper);
    }

    inline vec3 extend() const{
        return upper - lower;
    }

    char maxExtend() const {
        vec3 diff = extend();
        return MaxIndex(diff);
    }

    inline bool isNegative() const{
        return (extend().array() <= 0).any();
    }

    /**
     * @brief split this box into two halves
     * @param dimension [0-2] dimension along to split
     * @param split Scalar value along dimension
     * @remark may produce one empty box if split is at the boundary/outside of this box
     **/
    std::tuple<BoundingBox, BoundingBox> split(char dimension, Scalar split){
        BoundingBox left(*this);
        BoundingBox right(*this);

        assert(lower[dimension] < split && split < upper[dimension]-1);

        // max/min -> prevent split outside this box
        left.upper[dimension] = std::max(lower[dimension], std::min(split, upper[dimension]));
        right.lower[dimension] = std::max(lower[dimension], std::min(split, upper[dimension]));
        assert(left.outside(right) && right.outside(left));

        return std::make_tuple(left, right);
    }

    inline VolumeType volume() const{
        vec3 diff = extend();
        return diff[0] * diff[1] * diff[2];
    }

    inline bool contains(const BoundingBox &other) const {
        return lower <= other.lower && other.upper <= upper;
    }

    inline bool outside(const BoundingBox &other) const {
        return other.upper < lower || upper < other.lower;
    }

    BoundingBox overlap(const BoundingBox &other) const {
        BoundingBox common({0, 0, 0}, {0, 0, 0});
        for (unsigned char i = 0; i < 3; ++i) {
            common.lower[i] = other.lower[i] < lower[i] ? lower[i] : other.lower[i];
            common.upper[i] = upper[i] < other.upper[i] ? upper[i] : other.upper[i];
        }
        for (unsigned char i = 0; i < 3; ++i) {
            common.upper[i] = common.lower[i] <= common.upper[i] ? common.upper[i] : common.lower[i];
        }
        return common;
    }

    vec3 lower;  // including
    vec3 upper;  // excluding
};
