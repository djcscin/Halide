#ifndef __TEST__
#define __TEST__

#include "HalideBuffer.h"
#include "dng_io.h"
#include "read_metadata.hpp"
#include "transform_wb.hpp"
#include "halide_image_io.h"
#include "run_benchmark.hpp"

using namespace Halide::Runtime;
using namespace Halide::Tools;

#include "normalization.h"
#include "black_level_subtraction.h"
#include "bilinear_resize.h"
#include "lens_shading_correction.h"
#include "white_balance.h"
#include "demosaic.h"
#include "color_correction.h"
#include "reinhard_tone_mapping.h"
#include "gamma_correction.h"
#include "denormalization.h"
#include "rgb_to_ycbcr.h"
#include "bilateral_denoise.h"
#include "mix.h"
#include "ycbcr_to_rgb.h"

enum Test {
    BD = 0,
    BR,
    BLS,
    CC,
    DMS,
    DNORM,
    GC,
    LSC,
    MIX,
    NORM,
    RTM,
    R2Y,
    WB,
    Y2R,
};

template<Test OP>
int test(int argc, char ** argv) {
    if(argc < 8) {
        puts("Usage: ./test_* path_input path_lsc_map path_input_metadata gamma sigma_spatial sigma_range path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const char * path_lsc_map = argv[2];
    const char * path_input_metadata = argv[3];
    const float gamma = atof(argv[4]);
    const float sigma_spatial = atof(argv[5]);
    const float sigma_range = atof(argv[6]);
    const char * path_output = argv[7];

    Raw<uint16_t> input = load_dng<uint16_t>(path_input);
    const int width = input.buffer.width();
    const int height = input.buffer.height();
    const int numel = input.buffer.number_of_elements();
    Buffer<float> lsc_map = load_image(path_lsc_map);
    Buffer<float> wb_rgb(3);
    Buffer<float> wb4(4);
    Buffer<float> ccm(3,3);
    read_metadata(path_input_metadata, wb_rgb, ccm);
    transform_wb(input.cfa_pattern, wb_rgb, wb4);

    Buffer<uint16_t> output(width, height, 3);

    {
        Buffer<float> black_level_f32(4);
        for(int i=0; i<4; ++i) black_level_f32(i) = float(input.black_level(i)) / input.white_level;

        Buffer<float> im_norm(width, height);
        auto norm = [&]() {
            normalization(input.buffer, input.white_level, im_norm);
        };
        if(OP == NORM) {
            run_benchmark(numel, norm);
        } else {
            norm();
        }

        Buffer<float> im_bls(width, height);
        auto bls = [&]() {
            black_level_subtraction(im_norm, black_level_f32, im_bls);
        };
        if(OP == BLS) {
            run_benchmark(numel, bls);
        } else {
            bls();
        }

        Buffer<float> lsc_map_bigger(width/2, height/2, 4);
        auto br = [&]() {
            bilinear_resize(lsc_map, lsc_map.width(), lsc_map.height(), width/2, height/2, lsc_map_bigger);
        };
        if(OP == BR) {
            run_benchmark(numel, br);
        } else {
            br();
        }

        Buffer<float> im_lsc(width, height);
        auto lsc = [&]() {
            lens_shading_correction(im_bls, lsc_map_bigger, im_lsc);
        };
        if(OP == LSC) {
            run_benchmark(numel, lsc);
        } else {
            lsc();
        }

        Buffer<float> im_wb(width, height);
        auto wb = [&]() {
            white_balance(im_lsc, wb4, im_wb);
        };
        if(OP == WB) {
            run_benchmark(numel, wb);
        } else {
            wb();
        }

        Buffer<float> im_dms(width, height, 3);
        auto dms = [&]() {
            demosaic(im_wb, width, height, input.cfa_pattern, im_dms);
        };
        if(OP == DMS) {
            run_benchmark(numel, dms);
        } else {
            dms();
        }

        Buffer<float> im_r2y(width, height, 3);
        auto r2y = [&]() {
            rgb_to_ycbcr(im_dms, im_r2y);
        };
        if(OP == R2Y) {
            run_benchmark(numel, r2y);
        } else {
            r2y();
        }

        Buffer<float> im_dns(width, height, 2);
        im_dns.set_min({0, 0, 1});
        auto bd = [&]() {
            bilateral_denoise(im_r2y, im_dms, width, height, sigma_spatial, sigma_range, im_dns);
        };
        if(OP == BD) {
            run_benchmark(numel, bd);
        } else {
            bd();
        }

        Buffer<float> im_mix(width, height, 3);
        auto lmix = [&]() {
            mix(im_r2y, im_dns, im_mix);
        };
        if(OP == MIX) {
            run_benchmark(numel, lmix);
        } else {
            lmix();
        }

        Buffer<float> im_y2r(width, height, 3);
        auto y2r = [&]() {
            ycbcr_to_rgb(im_mix, im_y2r);
        };
        if(OP == Y2R) {
            run_benchmark(numel, y2r);
        } else {
            y2r();
        }

        Buffer<float> im_cc(width, height, 3);
        auto cc = [&]() {
            color_correction(im_y2r, ccm, im_cc);
        };
        if(OP == CC) {
            run_benchmark(numel, cc);
        } else {
            cc();
        }

        Buffer<float> im_tm(width, height, 3);
        auto rtm = [&]() {
            reinhard_tone_mapping(im_cc, width, height, im_tm);
        };
        if(OP == RTM) {
            run_benchmark(numel, rtm);
        } else {
            rtm();
        }

        Buffer<float> im_gc(width, height, 3);
        auto gc = [&]() {
            gamma_correction(im_tm, gamma, im_gc);
        };
        if(OP == GC) {
            run_benchmark(numel, gc);
        } else {
            gc();
        }

        auto dnorm = [&]() {
            denormalization(im_gc, 65535, output);
        };
        if(OP == DNORM) {
            run_benchmark(numel, dnorm);
        } else {
            dnorm();
        }
    }

    save_image(output, path_output);

    return 0;
}

#endif
