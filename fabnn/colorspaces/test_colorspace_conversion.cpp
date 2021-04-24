/**
 * @file test_colorspace_conversion.cpp
 * @date 22.02.19
 * @author Tobias Rittig
 * @brief Verify our colorspace conversions
 **/

#include "colorspace_conversion.h"
#include <iostream>

using namespace std;


int main(int argc, char const *argv[])
{
    string init = R"(
numpy.set_printoptions(precision=10, suppress=True, floatmode='fixed', linewidth=300, threshold=1500)
from skimage.color import lab2lch, lch2lab, rgb2hsv
from experiments.tiny_patch_optimization import linear_rgb2lab
print('RGB: {}'.format(rgb[0,0]))
print('Lab: {}'.format(linear_rgb2lab(rgb)[0,0]))
print('LCh: {}'.format(lab2lch(linear_rgb2lab(rgb))[0,0]))
print('HSV: {}'.format(rgb2hsv(rgb)[0,0]))
)";

    if(argc != 4){
        cerr << "Usage: "<<argv[0] << " <R> <G> <B>"<<endl;
        return -1;
    }
    openvdb::Vec3f rgb(atof(argv[1]), atof(argv[2]), atof(argv[3]));

    ostringstream pyinit;
    pyinit << "import numpy" << endl << "rgb = numpy.array([[(" << rgb[0] << "," << rgb[1] << "," << rgb[2] << ")]], dtype=numpy.float64)" << endl;
    pyinit << init;

    cout.precision(10);
    cout << fixed;
    cout << "RGB: " << rgb << endl;
    cout << "Lab: " << linear_rgb2lab(rgb) << endl;
    cout << "forward RGB -> LCh" << endl;
    cout << "LCh: " << linear_rgb2lch(rgb) << endl;
    cout << "backward LCh -> RGB" << endl;
    cout << "RGB: " << lch2linear_rgb(linear_rgb2lch(rgb)) << endl;
    cout << "forward RGB -> HSV" << endl;
    cout << "HSV: " << linear_rgb2hsv(rgb) << endl;
    cout << "backward HSV -> RGB" << endl;
    cout << "RGB: " << hsv2linear_rgb(linear_rgb2hsv(rgb)) << endl;
    cout << "RGB: " << hsv2linear_rgb({0.0f, 1.0f, 0.3f}) << endl;

    cout << endl << " ======================= " << endl << "     PYTHON     " << endl << " ======================= " << endl << endl;

    return system(("python -c \""+pyinit.str()+"\"").c_str());
}
