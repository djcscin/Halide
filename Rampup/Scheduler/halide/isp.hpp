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
            normalization->out_define_schedule.set(mscheduler < 15);
            normalization->apply(input, white_level);

            black_level_subtraction = create<BlackLevelSubtraction>();
            black_level_subtraction->out_define_schedule.set(mscheduler < 14);
            black_level_subtraction->apply(normalization->output, black_level_f32);

            bilinear_resize = create<BilinearResize>();
            bilinear_resize->out_define_schedule.set(mscheduler < 16);
            bilinear_resize->apply(lsc_map, lsc_map.width(), lsc_map.height(), input.width()/2, input.height()/2);

            lens_shading_correction = create<LensShadingCorrection>();
            lens_shading_correction->out_define_schedule.set(mscheduler < 13);
            lens_shading_correction->apply(black_level_subtraction->output, bilinear_resize->output);

            white_balance = create<WhiteBalance>();
            white_balance->out_define_compute.set(mscheduler != 11);
            white_balance->out_define_schedule.set(mscheduler < 12);
            white_balance->apply(lens_shading_correction->output, wb);

            demosaic = create<Demosaic>(); // Bayer 2D (width, height) -> 3D (width, height, 3)
            demosaic->apply(white_balance->output, input.width(), input.height(), cfa_pattern);

            rgb_to_ycbcr = create<RGB2YCbCr>();
            rgb_to_ycbcr->out_define_schedule.set(mscheduler < 8);
            rgb_to_ycbcr->apply(demosaic->output);

            bilateral_denoise_input(x, y, c) = rgb_to_ycbcr->output(x, y, c);

            bilateral_denoise = create<BilateralDenoise>();
            bilateral_denoise->apply(bilateral_denoise_input, demosaic->output, input.width(), input.height(), sigma_spatial, sigma_range);

            mix = create<Mix>();
            mix->out_define_schedule.set(mscheduler < 7);
            mix->apply(rgb_to_ycbcr->output, bilateral_denoise->output);

            ycbcr_to_rgb = create<YCbCr2RGB>();
            ycbcr_to_rgb->out_define_schedule.set(mscheduler < 6);
            ycbcr_to_rgb->apply(mix->output);

            color_correction = create<ColorCorrection>();
            color_correction->out_define_schedule.set(mscheduler < 5);
            color_correction->apply(ycbcr_to_rgb->output, ccm);

            reinhard_tone_mapping = create<ReinhardToneMapping>();
            reinhard_tone_mapping->out_define_schedule.set(mscheduler < 4);
            reinhard_tone_mapping->apply(color_correction->output, input.width(), input.height());

            gamma_correction = create<GammaCorrection>();
            gamma_correction->out_define_schedule.set(mscheduler < 3);
            gamma_correction->apply(reinhard_tone_mapping->output, gamma);

            denormalization = create<Denormalization>(); // range (0.f, 1.f) -> (0, 65535)
            denormalization->out_define_schedule.set(mscheduler < 3);
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
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                Var xo("xo"), xi("xi"), yo("yo"), yi("yi"), yc("yc");
                Var xoo("xoo"), yoo("yoo");

                black_level_f32.compute_root()
                    .bound(c, 0, 4)
                    .vectorize(c, 4)
                ;

                if((mscheduler == 8) || (mscheduler >= 11)) {
                    rgb_to_ycbcr->output.in(bilateral_denoise_input).compute_at(bilateral_denoise->output, yoo)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .unroll(c)
                        .vectorize(xi)
                    ;
                }

                if(mscheduler == 10) {
                    rgb_to_ycbcr->output.in(mix->output).compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .unroll(c)
                        .vectorize(xi)
                        .parallel(y)
                    ;
                }

                if(mscheduler == 11) {
                    white_balance->output.compute_at(demosaic->output, yo);
                }

                switch (mscheduler)
                {
                case 3:
                    // denormalization->output and gamma_correction->output inline

                case 2:
                    output.compute_root()
                        .fuse(y, c, yc).parallel(yc)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    break;

                default:
                case 1:
                case 16:
                    // case 15 + bilinear_resize->output inline
                    bilinear_resize->intm_compute_level.set({demosaic->output, yo});
                    bilinear_resize->kernel_y_compute_level.set({demosaic->output, yo});

                case 15:
                    // case 14 + normalization->output inline
                case 14:
                    // case 13 + black_level_subtraction->output inline
                case 13:
                    // case 12 + lens_shading_correction->output inline
                case 12:
                    // case 8 + white_balance->output inline
                case 11:
                    // case 8 + white_balance->output compute at demosaic->output
                case 10:
                    // case 8 + rgb_to_ycbcr->output used in max_gray and output compute root
                case 9:
                    // case 7 + rgb_to_ycbcr->output inline
                case 8:
                    // case 7 + rgb_to_ycbcr->output used in bilateral_denoise compute at bilateral_denoise->output
                case 7:
                    // case 6 + mix->output inline
                case 6:
                    // case 5 + ycbcr_to_rgb->output inline
                case 5:
                    // case 4 + color_correction->output inline
                case 4:
                    // case 3 + reinhard_tone_mapping->output inline
                    output.compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    reinhard_tone_mapping->intm_compute_level.set({output, y});
                    break;
                }
            }
        }
    };
};

#endif
