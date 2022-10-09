#ifndef __TEST__
#define __TEST__

#include "buffer.hpp"
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
    COPY_GPU,
    COPY_CPU,
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
        const struct halide_device_interface_t * device_interface = INTERFACE;
        const struct halide_device_interface_t * texture_interface = INTERFACE_TEX;

        Buffer<float> black_level_f32(4);
        for(int i=0; i<4; ++i)
            black_level_f32(i) = float(input.black_level(i)) / input.white_level;

        auto copy_gpu = [&]() {
            copy_to_gpu(input.buffer, texture_interface);
            copy_to_gpu(lsc_map, texture_interface);
            copy_to_gpu(wb4, texture_interface);
            copy_to_gpu(ccm, texture_interface);
            copy_to_gpu(black_level_f32, texture_interface);
        };
        if(OP == COPY_GPU) {
            run_benchmark(numel, copy_gpu);
        } else {
            copy_gpu();
        }

        Buffer<float> im_norm = create_buffer<float>(device_interface, {width, height});
        auto norm = [&]() {
            normalization(input.buffer, input.white_level, im_norm);
            im_norm.device_sync();
        };
        if(OP == NORM) {
            run_benchmark(numel, norm);
        } else {
            norm();
        }

        Buffer<float> im_bls = create_buffer<float>(device_interface, {width, height});
        auto bls = [&]() {
            black_level_subtraction(im_norm, black_level_f32, im_bls);
            im_bls.device_sync();
        };
        if(OP == BLS) {
            run_benchmark(numel, bls);
        } else {
            bls();
        }
        release_buffer(im_norm);

        Buffer<float> lsc_map_bigger = create_buffer<float>(device_interface, {width/2, height/2, 4});
        auto br = [&]() {
            bilinear_resize(lsc_map, lsc_map.width(), lsc_map.height(), width/2, height/2, lsc_map_bigger);
            lsc_map_bigger.device_sync();
        };
        if(OP == BR) {
            run_benchmark(numel, br);
        } else {
            br();
        }

        Buffer<float> im_lsc = create_buffer<float>(device_interface, {width, height});
        auto lsc = [&]() {
            lens_shading_correction(im_bls, lsc_map_bigger, im_lsc);
            im_lsc.device_sync();
        };
        if(OP == LSC) {
            run_benchmark(numel, lsc);
        } else {
            lsc();
        }
        release_buffer(im_bls);
        release_buffer(lsc_map_bigger);

        Buffer<float> im_wb = create_buffer<float>(device_interface, {width, height});
        auto wb = [&]() {
            white_balance(im_lsc, wb4, im_wb);
            im_wb.device_sync();
        };
        if(OP == WB) {
            run_benchmark(numel, wb);
        } else {
            wb();
        }
        release_buffer(im_lsc);

        Buffer<float> im_dms = create_buffer<float>(device_interface, {width, height, 3});
        auto dms = [&]() {
            demosaic(im_wb, width, height, input.cfa_pattern, im_dms);
            im_dms.device_sync();
        };
        if(OP == DMS) {
            run_benchmark(numel, dms);
        } else {
            dms();
        }
        release_buffer(im_wb);

        Buffer<float> im_r2y = create_buffer<float>(device_interface, {width, height, 3});
        auto r2y = [&]() {
            rgb_to_ycbcr(im_dms, im_r2y);
            im_r2y.device_sync();
        };
        if(OP == R2Y) {
            run_benchmark(numel, r2y);
        } else {
            r2y();
        }

        Buffer<float> im_dns = create_buffer<float>(device_interface, {width, height, 2}, {0, 0, 1});
        auto bd = [&]() {
            bilateral_denoise(im_r2y, im_dms, width, height, sigma_spatial, sigma_range, im_dns);
            im_dns.device_sync();
        };
        if(OP == BD) {
            run_benchmark(numel, bd);
        } else {
            bd();
        }
        release_buffer(im_dms);

        Buffer<float> im_mix = create_buffer<float>(device_interface, {width, height, 3});
        auto lmix = [&]() {
            mix(im_r2y, im_dns, im_mix);
            im_mix.device_sync();
        };
        if(OP == MIX) {
            run_benchmark(numel, lmix);
        } else {
            lmix();
        }
        release_buffer(im_r2y);
        release_buffer(im_dns);

        Buffer<float> im_y2r = create_buffer<float>(device_interface, {width, height, 3});
        auto y2r = [&]() {
            ycbcr_to_rgb(im_mix, im_y2r);
            im_y2r.device_sync();
        };
        if(OP == Y2R) {
            run_benchmark(numel, y2r);
        } else {
            y2r();
        }
        release_buffer(im_mix);

        Buffer<float> im_cc = create_buffer<float>(device_interface, {width, height, 3});
        auto cc = [&]() {
            color_correction(im_y2r, ccm, im_cc);
            im_cc.device_sync();
        };
        if(OP == CC) {
            run_benchmark(numel, cc);
        } else {
            cc();
        }
        release_buffer(im_y2r);

        Buffer<float> im_tm = create_buffer<float>(device_interface, {width, height, 3});
        auto rtm = [&]() {
            reinhard_tone_mapping(im_cc, width, height, im_tm);
            im_tm.device_sync();
        };
        if(OP == RTM) {
            run_benchmark(numel, rtm);
        } else {
            rtm();
        }
        release_buffer(im_cc);

        Buffer<float> im_gc = create_buffer<float>(device_interface, {width, height, 3});
        auto gc = [&]() {
            gamma_correction(im_tm, gamma, im_gc);
            im_gc.device_sync();
        };
        if(OP == GC) {
            run_benchmark(numel, gc);
        } else {
            gc();
        }
        release_buffer(im_tm);

        auto dnorm = [&]() {
            denormalization(im_gc, 65535, output);
            output.device_sync();
        };
        if(OP == DNORM) {
            run_benchmark(numel, dnorm);
        } else {
            dnorm();
        }
        release_buffer(im_gc);

        auto copy_cpu = [&]() {
            copy_to_cpu(output);
        };
        if(OP == COPY_CPU) {
            run_benchmark(numel, copy_cpu);
        } else {
            copy_cpu();
        }
    }

    save_image(output, path_output);

    return 0;
}

#endif
