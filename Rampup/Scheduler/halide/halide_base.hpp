#ifndef __HALIDE_BASE__
#define __HALIDE_BASE__

#include "Halide.h"

namespace {
    using namespace Halide;

    class HalideBase {
    public:
        GeneratorParam<int> scheduler{"scheduler", 1};
        GeneratorParam<double> size_factor{"size_factor", 1};

        GeneratorParam<bool> out_define_schedule{"out_define_schedule", true};
        GeneratorParam<bool> out_define_compute{"out_define_compute", true};
        GeneratorParam<LoopLevel> intm_compute_level{"intm_compute_level", LoopLevel::inlined()};
        GeneratorParam<LoopLevel> intm_store_level{"intm_store_level", LoopLevel::inlined()};
    };
};

#endif
