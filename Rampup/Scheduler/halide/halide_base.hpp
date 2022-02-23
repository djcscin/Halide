#ifndef __HALIDE_BASE__
#define __HALIDE_BASE__

#include "Halide.h"

namespace {
    using namespace Halide;

    class HalideBase {
    public:
        GeneratorParam<int> scheduler{"scheduler", 0};
        GeneratorParam<double> par_size_factor{"par_size_factor", 1};
        GeneratorParam<double> vec_size_factor{"vec_size_factor", 1};
    };
};

#endif
