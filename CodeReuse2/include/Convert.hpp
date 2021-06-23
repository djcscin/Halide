#ifndef _CONVERT_
#define _CONVERT_

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

Expr convert_from_unit(Type type, Expr unit) {
    if (type == UInt(8)) {
        return unit * 255.f;
    } else if (type == UInt(16)) {
        return unit * 4095.f;
    } else {
        return -1;
    }
}

Expr clamp_convert_from_unit(Type type, Expr unit) {
    return convert_from_unit(type, clamp(unit, 0.f, 1.f));
}

Expr u16_to_u8(Expr x) {
    return u8(x * 255.f / 4095.f);
}

float k_ycbcr(Type type) {
    return (type == UInt(8)) ? 128.f : 2048.f;
}

#endif
