#include "Halide.h"
#include "CFA.hpp"
#include "HalideDemosaic.hpp"
#include "HalideLscMap.hpp"
#include "HalideDenoise.hpp"
#include "HalideRGB2YCbCr.hpp"
#include "HalideMix.hpp"
#include "HalideYCbCr2RGB.hpp"

using namespace Halide;

class HalideISP : public Generator<HalideISP> {
    public:
        Input<Buffer<uint16_t>> img_input{"img_input", 2};
        Input<Buffer<float>> lsc_map_input{"lsc_map_input", 3};
        Input<Buffer<float>> white_balance{"white_balance", 1};
        Input<Buffer<uint16_t>> black_level{"black_level", 1};
        Input<uint16_t> white_level{"white_level"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};

        Output<Buffer<uint8_t>> img_output{"isp_img_output", 3};

        GeneratorParam<bool> has_sigma_luma{"has_sigma_luma", false};
        GeneratorParam<bool> output_demosaic{"output_demosaic", false};

        void configure() {
            if(has_sigma_luma) {
                sigma_spatial_luma = add_input<float>("sigma_spatial_luma");
                sigma_range_luma = add_input<float>("sigma_range_luma");
            }

            if(output_demosaic) {
                img_output_demosaic = add_output<Buffer<uint8_t>>("img_output_demosaic", 3);
            }
        }

        void generate() {
            lsc_map = create<HalideLscMap>();
            lsc_map->composable.set(false);
            // Todos os GeneratorParam's têm que ser setados antes do apply
            // exceto os de LoopLevel (onde vai ser o compute_at e o store_at)
            lsc_map->apply(lsc_map_input, img_input.width(), img_input.height());

            demosaic = create<HalideDemosaic>();
            demosaic->img_output_type.set(UInt(16));
            demosaic->composable.set(true);
            demosaic->apply(img_input, lsc_map->lsc_map_output, white_balance, black_level, white_level, cfa_pattern);

            rgb_ycbcr = create<HalideRGB2YCbCr>();
            rgb_ycbcr->img_output_type.set(UInt(16));
            rgb_ycbcr->define_schedule.set(false);
            rgb_ycbcr->apply(demosaic->img_output);

            chroma_dns = create<HalideDenoise>();
            chroma_dns->img_output_type.set(UInt(16));
            chroma_dns->composable.set(true);
            chroma_dns->min_channel.set(1);
            chroma_dns->num_channels.set(2);
            chroma_dns->apply(rgb_ycbcr->img_output, img_input.width(), img_input.height(), sigma_spatial, sigma_range);

            mix = create<HalideMix>();
            mix->define_schedule.set(false);
            if(has_sigma_luma) {
                luma_dns = create<HalideDenoise>();
                luma_dns->img_output_type.set(UInt(16));
                luma_dns->composable.set(true);
                luma_dns->min_channel.set(0);
                luma_dns->num_channels.set(1);
                luma_dns->apply(rgb_ycbcr->img_output, img_input.width(), img_input.height(), *sigma_spatial_luma, *sigma_range_luma);

                mix->apply(luma_dns->img_output, chroma_dns->img_output);
            } else {
                mix->apply(rgb_ycbcr->img_output, chroma_dns->img_output);
            }

            ycbcr_rgb = create<HalideYCbCr2RGB>();
            ycbcr_rgb->img_output_type.set(UInt(8));
            ycbcr_rgb->define_schedule.set(false);
            ycbcr_rgb->apply(mix->img_output);

            img_output = ycbcr_rgb->img_output;

            if(output_demosaic) {
                (*img_output_demosaic)(x, y, c) = u16_to_u8(demosaic->img_output(x, y, c));
            }
        }

        void schedule() {
            img_input.dim(0).set_min(0);
            img_input.dim(1).set_min(0);
            lsc_map_input.dim(2).set_bounds(0, 4);
            white_balance.dim(0).set_bounds(0, 3);
            black_level.dim(0).set_bounds(0, 4);

            if(auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}});
                lsc_map_input.set_estimates({{0, 4000}, {0, 3000}, {0, 4}});
                white_balance.set_estimates({{0, 3}});
                black_level.set_estimates({{0, 4}});
                white_level.set_estimate(16*1023);
                cfa_pattern.set_estimate(RGGB);
                sigma_spatial.set_estimate(5.0f);
                sigma_range.set_estimate(0.06f);
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();
                img_output
                    .compute_root()
                    .bound(c, 0, 3)
                    .unroll(c)
                    .split(y, yo, yi, 32).parallel(yo)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, yi, yo)
                ;

                if(output_demosaic) {
                    demosaic->img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .align_bounds(y, 2, 0)
                        .align_bounds(x, 2, 0)
                        .unroll(c)
                        .split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    demosaic->intm_compute_level.set({demosaic->img_output, yo_i});
                    demosaic->intm_store_level.set({demosaic->img_output, yo_o});
                    rgb_ycbcr->img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .split(y, yo, yi, 32).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    (*img_output_demosaic)
                        .compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .split(y, yo, yi, 32).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                        .compute_with(rgb_ycbcr->img_output, xo)
                    ;
                } else {
                    rgb_ycbcr->img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .align_bounds(y, 2, 0)
                        .align_bounds(x, 2, 0)
                        .unroll(c)
                        .split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    demosaic->intm_compute_level.set({rgb_ycbcr->img_output, yo_i});
                    demosaic->intm_store_level.set({rgb_ycbcr->img_output, yo_o});
                }

                chroma_dns->img_output
                    .compute_at(img_output, yo)
                    .bound(c, 1, 2)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, y)
                ;
                chroma_dns->intm_compute_level.set({chroma_dns->img_output, y});
                chroma_dns->intm_store_level.set({img_output, yo});

                if(has_sigma_luma) {
                    luma_dns->img_output
                        .compute_at(img_output, yo)
                        .bound(c, 0, 1)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    luma_dns->intm_compute_level.set({luma_dns->img_output, y});
                    luma_dns->intm_store_level.set({img_output, yo});
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xo{"xo"}, xi{"xi"};
        Var yo{"yo"}, yi{"yi"};
        Var yo_o{"yo_o"}, yo_i{"yo_i"};

        Input<float> * sigma_spatial_luma;
        Input<float> * sigma_range_luma;
        Output<Buffer<uint8_t>> * img_output_demosaic;

        std::unique_ptr<HalideLscMap> lsc_map;
        std::unique_ptr<HalideDemosaic> demosaic;
        std::unique_ptr<HalideRGB2YCbCr> rgb_ycbcr;
        std::unique_ptr<HalideDenoise> chroma_dns;
        std::unique_ptr<HalideDenoise> luma_dns;
        std::unique_ptr<HalideMix> mix;
        std::unique_ptr<HalideYCbCr2RGB> ycbcr_rgb;
};
HALIDE_REGISTER_GENERATOR(HalideISP, isp)
