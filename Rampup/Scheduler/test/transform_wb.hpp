#ifndef __TRANSFORM_WB__
#define __TRANSFORM_WB__

namespace {
    using namespace Halide;

    void transform_wb(uint8_t cfa_pattern, const Buffer<float> & wb_rgb, Buffer<float> & wb_cfa_pattern) {
        switch (cfa_pattern) {
        case RGGB:
            wb_cfa_pattern(0) = wb_rgb(0);
            wb_cfa_pattern(1) = wb_rgb(1);
            wb_cfa_pattern(2) = wb_rgb(1);
            wb_cfa_pattern(3) = wb_rgb(2);
            break;

        case GRBG:
            wb_cfa_pattern(0) = wb_rgb(1);
            wb_cfa_pattern(1) = wb_rgb(0);
            wb_cfa_pattern(2) = wb_rgb(2);
            wb_cfa_pattern(3) = wb_rgb(1);
            break;

        case GBRG:
            wb_cfa_pattern(0) = wb_rgb(1);
            wb_cfa_pattern(1) = wb_rgb(2);
            wb_cfa_pattern(2) = wb_rgb(0);
            wb_cfa_pattern(3) = wb_rgb(1);
            break;

        case BGGR:
            wb_cfa_pattern(0) = wb_rgb(2);
            wb_cfa_pattern(1) = wb_rgb(1);
            wb_cfa_pattern(2) = wb_rgb(1);
            wb_cfa_pattern(3) = wb_rgb(0);
            break;

        default:
            break;
        }
    }

};

#endif
