#ifndef __ISP__
#define __ISP__

#include "halide_base.hpp"
#include "constants.hpp"
#include "normalization.hpp"
#include "black_level_subtraction.hpp"
#include "bilinear_resize.hpp"
#include "lens_shading_correction.hpp"
#include "white_balance.hpp"
#include "demosaic.hpp"
#include "rgb_to_ycbcr.hpp"
#include "bilateral_denoise.hpp"
#include "mix.hpp"
#include "ycbcr_to_rgb.hpp"
#include "color_correction.hpp"
#include "reinhard_tone_mapping.hpp"
#include "gamma_correction.hpp"
#include "denormalization.hpp"

namespace {
    using namespace Halide;

    class ISP : public Generator<ISP>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func black_level_f32{"black_level_f32"};
        Func bilateral_denoise_input{"bilateral_denoise_input"};
        std::unique_ptr<Normalization> normalization;
        std::unique_ptr<BlackLevelSubtraction> black_level_subtraction;
        std::unique_ptr<BilinearResize> bilinear_resize;
        std::unique_ptr<LensShadingCorrection> lens_shading_correction;
        std::unique_ptr<WhiteBalance> white_balance;
        std::unique_ptr<Demosaic> demosaic;
        std::unique_ptr<RGB2YCbCr> rgb_to_ycbcr;
        std::unique_ptr<BilateralDenoise> bilateral_denoise;
        std::unique_ptr<Mix> mix;
        std::unique_ptr<YCbCr2RGB> ycbcr_to_rgb;
        std::unique_ptr<ColorCorrection> color_correction;
        std::unique_ptr<ReinhardToneMapping> reinhard_tone_mapping;
        std::unique_ptr<GammaCorrection> gamma_correction;
        std::unique_ptr<Denormalization> denormalization;

        int mscheduler;

    public:
        Input<Buffer<uint16_t>> input{"input", 2};
        Input<Buffer<float>> lsc_map{"lsc_map", 3};
        Input<Buffer<float>> wb{"wb", 1};
        Input<Buffer<float>> ccm{"ccm", 2};
        Input<Buffer<uint16_t>> black_level{"black_level", 1};
        Input<uint16_t> white_level{"white_level"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};
        Input<float> gamma{"gamma"};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};
        Output<Buffer<uint16_t>> output{"output_isp", 3};

        void generate() {
            mscheduler = (scheduler == 1)?16:scheduler;

            black_level_f32(c) = f32(black_level(c)) / white_level; // range (0,white_level) -> (0.f,1.f)

            normalization = create<Normalization>(); // range (0,white_level) -> (0.f,1.f)
            normalization->out_define_schedule.set(false);
            normalization->apply(input, white_level);

            black_level_subtraction = create<BlackLevelSubtraction>();
            black_level_subtraction->out_define_schedule.set(false);
            black_level_subtraction->apply(normalization->output, black_level_f32);

            bilinear_resize = create<BilinearResize>();
            bilinear_resize->out_define_schedule.set(false);
            bilinear_resize->apply(lsc_map, lsc_map.width(), lsc_map.height(), input.width()/2, input.height()/2);

            lens_shading_correction = create<LensShadingCorrection>();
            lens_shading_correction->out_define_schedule.set(false);
            lens_shading_correction->apply(black_level_subtraction->output, bilinear_resize->output);

            white_balance = create<WhiteBalance>();
            white_balance->out_define_schedule.set(false);
            white_balance->apply(lens_shading_correction->output, wb);

            demosaic = create<Demosaic>(); // Bayer 2D (width, height) -> 3D (width, height, 3)
            demosaic->apply(white_balance->output, input.width(), input.height(), cfa_pattern);

            rgb_to_ycbcr = create<RGB2YCbCr>();
            rgb_to_ycbcr->out_define_schedule.set(false);
            rgb_to_ycbcr->apply(demosaic->output);

            bilateral_denoise_input(x, y, c) = rgb_to_ycbcr->output(x, y, c);

            bilateral_denoise = create<BilateralDenoise>();
            bilateral_denoise->apply(bilateral_denoise_input, demosaic->output, input.width(), input.height(), sigma_spatial, sigma_range);

            mix = create<Mix>();
            mix->out_define_schedule.set(false);
            mix->apply(rgb_to_ycbcr->output, bilateral_denoise->output);

            ycbcr_to_rgb = create<YCbCr2RGB>();
            ycbcr_to_rgb->out_define_schedule.set(false);
            ycbcr_to_rgb->apply(mix->output);

            color_correction = create<ColorCorrection>();
            color_correction->out_define_schedule.set(false);
            color_correction->apply(ycbcr_to_rgb->output, ccm);

            reinhard_tone_mapping = create<ReinhardToneMapping>();
            reinhard_tone_mapping->out_define_schedule.set(false);
            reinhard_tone_mapping->apply(color_correction->output, input.width(), input.height());

            gamma_correction = create<GammaCorrection>();
            gamma_correction->out_define_schedule.set(false);
            gamma_correction->apply(reinhard_tone_mapping->output, gamma);

            denormalization = create<Denormalization>(); // range (0.f, 1.f) -> (0, 65535)
            denormalization->out_define_schedule.set(false);
            denormalization->apply(gamma_correction->output, max16_u16);

            output(x, y, c) = denormalization->output(x, y, c);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000}});
                lsc_map.set_estimates({{0,17},{0,13},{0, 4}});
                wb.set_estimates({{0,4}});
                black_level.set_estimates({{0,4}});
                white_level.set_estimate(1023);
                gamma.set_estimate(2.2f);
                sigma_spatial.set_estimate(5.f);
                sigma_range.set_estimate(0.05f);
                cfa_pattern.set_estimate(RGGB);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads_x = 64;
                const int num_threads_y = 16;
                const int num_threads_1d = std::min(num_threads_x*num_threads_y,512);
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"};
                black_level_f32.compute_root().store_in(MemoryType::GPUTexture)
                    .bound(c, 0, 4)
                    .vectorize(c, 4)
                    .gpu_single_thread()
                ;
                output.compute_root()
                    .split(x, xo, xi, num_threads_x*vector_size)
                    .split(xi, xi, xi2, vector_size)
                    .split(y, yo, yi, num_threads_y)
                    .reorder(xi2, xi, yi, xo, yo, c)
                    .gpu_blocks(xo, yo, c)
                    .gpu_threads(xi, yi)
                    .vectorize(xi2)
                ;
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                Var xo("xo"), xi("xi"), yo("yo"), yi("yi"), yc("yc");
                Var xoo("xoo"), yoo("yoo");
                black_level_f32.compute_root()
                    .bound(c, 0, 4)
                    .vectorize(c, 4)
                ;
                output.compute_root()
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, y)
                    .parallel(y)
                ;
                reinhard_tone_mapping->intm_compute_level.set({output, y});
                bilinear_resize->intm_compute_level.set({demosaic->output, yo});
                rgb_to_ycbcr->output.in(bilateral_denoise_input).compute_at(bilateral_denoise->output, yoo)
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, y)
                    .unroll(c)
                    .vectorize(xi)
                ;
            }
        }
    };
};

#endif