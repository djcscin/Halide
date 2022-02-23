#ifndef __CONSTANTS__
#define __CONSTANTS__

#include "Halide.h"

namespace {
    const uint16_t max14_u16 = (1 << 14) - 1;
    const float max14_f32 = max14_u16;

    const uint16_t max16_u16 = (1 << 16) - 1;
    const float max16_f32 = max16_u16;
};

#endif
