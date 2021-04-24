/**
 * @file colorspace_conversion.h
 * @date 05.11.18
 * @author Tobias Rittig
 * @brief Handle the conversion between different colorspaces (RGB, XYZ, Lab)
 **/
#pragma once

#include <openvdb/openvdb.h>

#define PI 3.14159265359

// taken from skimage/color/colorconv.py
static openvdb::Mat3R xyz_from_rgb(0.412453, 0.357580, 0.180423,
                                    0.212671, 0.715160, 0.072169,
                                    0.019334, 0.119193, 0.950227);

static openvdb::Mat3R rgb_from_xyz = xyz_from_rgb.inverse();

// reference xyz coordinates for illuminant="D65", observer="2"
static openvdb::Vec3R ref_white(0.95047, 1., 1.08883);


/**
 * @brief convert a linear RGB value into Lab color space
 **/
static openvdb::Vec3R linear_rgb2lab(const openvdb::Vec3R& rgb){
    openvdb::Vec3R xyz = xyz_from_rgb * rgb;

    xyz /= ref_white;

    double threshold = (6.0 / 29.0);
    double threshold2 = threshold * threshold;
    double threshold3 = threshold * threshold2;
    for(unsigned char i = 0; i < 3; ++i){
        if(xyz[i] > threshold3){
            xyz[i] = std::pow(xyz[i], 1. / 3.);
        } else {
            xyz[i] = xyz[i] / (3 * threshold2) + 4. / 29.;
        }
    }

    return {116. * xyz[1] - 16.,
            500 * (xyz[0] - xyz[1]),
            200 * (xyz[1] - xyz[2])};
}


/**
 * @brief convert a Lab value into Lch color space
 **/
static openvdb::Vec3f lab2lch(const openvdb::Vec3f& lab){
    auto c = std::sqrt(lab[1] * lab[1] + lab[2] * lab[2]);
    auto h = std::atan2(lab[2], lab[1]);
    if (h < 0) {
        h += 2 * PI;
    }
    return {lab[0], c, h};
}


/**
 * @brief convert a Lch value into Lab color space
 **/
static openvdb::Vec3f lch2lab(const openvdb::Vec3f& lch){
    auto c = lch[1], h = lch[2];
    auto a = c * std::cos(h);
    auto b = c * std::sin(h);
    return {lch[0], a, b};
}


/**
 * @brief convert a linear RGB value into Lch color space
 **/
static openvdb::Vec3f linear_rgb2lch(const openvdb::Vec3f& rgb){
    return lab2lch(linear_rgb2lab(rgb));
}


/**
 * @brief convert a linear RGB value into Lch color space
 **/
static openvdb::Vec3f lch2linear_rgb(const openvdb::Vec3f& lch){
    auto lab = lch2lab(lch);

    openvdb::Vec3f xyz;

    xyz[1] = (lab[0] + 16.0f) / 116.0f;
    xyz[0] = (lab[1] / 500) + xyz[1];
    xyz[2] = xyz[1] - (lab[2] / 200);

    double threshold = (6.0 / 29.0);
    for(unsigned char i = 0; i < 3; ++i){
        if(xyz[i] > threshold){
            xyz[i] *= xyz[i] * xyz[i];
        } else {
            xyz[i] = (xyz[i] - 16.0 / 116.0) / 7.787;
        }
    }
    xyz *= ref_white;

    return rgb_from_xyz * xyz;
}


/**
 * @brief Jacobian of (linear) RGB to Lab transformation
 **/
static openvdb::Mat3R linear_rgb2lab_grad(const openvdb::Vec3R& rgb){
    openvdb::Vec3R xyz = xyz_from_rgb * rgb;

    double threshold = (6.0 / 29.0);
    threshold *= threshold * threshold;

    openvdb::Mat3R result;
    // iterate over rows
    for(unsigned char i = 0; i < 3; ++i){
        if(xyz[i] > threshold){
            result.setRow(i, std::pow(xyz[i], -2.0/3.0) * (xyz_from_rgb.row(i) / (3.0 * std::pow(ref_white[i], 1. / 3.))));
        } else {
            result.setRow(i, (29. / 6.) * (29. / 6.) * (xyz_from_rgb.row(i) / (3.0 * ref_white[i])));
        }
    }

    openvdb::Mat3R J76(
        // X   Y   Z
        0., 116., 0., // L
        500., -500., 0., // a
        0., 200., -200. // b
    );
    
    return J76 * result;
}

/**
 * @brief Convert (linear) RGB values to HSV
 **/
static openvdb::Vec3f linear_rgb2hsv(const openvdb::Vec3f& rgb){
    unsigned char maxIndex = openvdb::math::MaxIndex(rgb),
                minIndex = openvdb::math::MinIndex(rgb);
    float minValue = rgb[minIndex],
        maxValue = rgb[maxIndex],
        maxMin = maxValue-minValue;

    openvdb::Vec3f hsv(0.0f);
    if(maxIndex != minIndex){
        switch (maxIndex)
        {
            default:
            case 0:
                hsv[0] = (0 + (rgb[1] - rgb[2]) / maxMin);
                break;
            case 1:
                hsv[0] = (2 + (rgb[2] - rgb[0]) / maxMin);
                break;
            case 2:
                hsv[0] = (4 + (rgb[0] - rgb[1]) / maxMin);
                break;
        }
        hsv[0] *= (M_PI / 3.0);
        if(hsv[0] < 0.0)
            hsv[0] += 2*M_PI; 
    }
    if(maxValue > 0.0f){
        hsv[1] = maxMin / maxValue;
    }
    hsv[2] = maxValue;
    return hsv;
}

/**
 * @brief Quick implementation of HSV to (linear) RGB conversion
 **/
static openvdb::Vec3f hsv2linear_rgb(const openvdb::Vec3f& hsv){
    struct converter{
        const openvdb::Vec3f& hsv;
        converter(const openvdb::Vec3f& hsv): hsv(hsv) {}

        inline float conv(unsigned char n){
            float k = fmod((n + 3.0f * hsv[0] / M_PI), 6.0f);
            return hsv[2] - hsv[2] * hsv[1] * std::max(0.0f, std::min(std::min(k, 4.0f-k), 1.0f));
        }
    } converter(hsv);
    return {converter.conv(5), converter.conv(3), converter.conv(1)};
}
