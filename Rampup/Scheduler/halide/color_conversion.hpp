#ifndef __COLOR_CONVERSION__
#define __COLOR_CONVERSION__

#include "Halide.h"

namespace {
    using namespace Halide;

    Expr rgb_to_gray(Expr r, Expr g, Expr b) {
        return clamp(0.299f*r + 0.587f*g + 0.114f*b, 0.f, 1.f);
    }
    Expr rgb_to_cb(Expr r, Expr g, Expr b) {
        return clamp(0.5f - 0.168736f*r - 0.331264f*g + 0.5f*b, 0.f, 1.f);
    }
    Expr rgb_to_cr(Expr r, Expr g, Expr b) {
        return clamp(0.5f + 0.5f*r - 0.418688f*g - 0.081312f*b, 0.f, 1.f);
    }

    Expr ycbcr_to_r(Expr y, Expr cb, Expr cr) {
        return clamp(y + 1.402f * (cr-0.5f), 0.f, 1.f);
    }
    Expr ycbcr_to_g(Expr y, Expr cb, Expr cr) {
        return clamp(y - 0.344136f * (cb-0.5f) - 0.714136f * (cr-0.5f), 0.f, 1.f);
    }
    Expr ycbcr_to_b(Expr y, Expr cb, Expr cr) {
        return clamp(y + 1.772f * (cb-0.5f), 0.f, 1.f);
    }

};

#endif
