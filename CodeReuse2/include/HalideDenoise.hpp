#include "Halide.h"
#include "Convert.hpp"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideDenoise : public Generator<HalideDenoise> {
    public:
        Input<Func> img_input{"img_input", 3};
        Input<int32_t> img_width{"img_width"};
        Input<int32_t> img_height{"img_height"};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};

        Output<Func> img_output{"img_dns_output", 3};

        GeneratorParam<Type> img_output_type{"img_output_type", UInt(8)};
        GeneratorParam<int32_t> min_channel{"min_channel", 0};
        GeneratorParam<int32_t> num_channels{"num_channels", 3};

        GeneratorParam<bool> composable{"composable", false};
        GeneratorParam<LoopLevel> intm_compute_level{"intm_compute_level", LoopLevel::inlined()};
        GeneratorParam<LoopLevel> intm_store_level{"intm_store_level", LoopLevel::inlined()};

        Expr gaussian(Expr i, Expr sigma) {
            return exp(-(i*i)/(2.f*sigma*sigma));
        }

        void generate() {
            input_bound(x, y, c) = i16(BoundaryConditions::repeat_edge(img_input,
                                    {{0, img_width}, {0, img_height}, {0, 3}})(x, y, c));

            Expr gaussian_width = max(i32(2.5f*sigma_spatial), 1);
            Expr kernel_size = 2*gaussian_width + 1;
            kernel = RDom(-gaussian_width, kernel_size, "kernel");

            weights_spatial(i) = gaussian(i, sigma_spatial);
            weights_range(i) = gaussian(i, convert_from_unit(img_input.type(), sigma_range));

            diff_y(x, y, i, c) = absd(input_bound(x, y, c), input_bound(x, y + i, c));
            norm_y(x, y, i) = u16(diff_y(x, y, i, 0) + diff_y(x, y, i, 1) + diff_y(x, y, i, 2));
            weights_y(x, y, i) = weights_spatial(i) * weights_range(norm_y(x, y, i));

            sum_y(x, y, c) += select(c == min_channel-1, weights_y(x, y, kernel),
                                weights_y(x, y, kernel) * input_bound(x, y + kernel, c));

            output_y(x, y, c) = i16(sum_y(x, y, c)/sum_y(x, y, min_channel-1));

            diff_x(x, y, i, c) = absd(input_bound(x, y, c), input_bound(x + i, y, c));
            norm_x(x, y, i) = u16(diff_x(x, y, i, 0) + diff_x(x, y, i, 1) + diff_x(x, y, i, 2));
            weights_x(x, y, i) = weights_spatial(i) * weights_range(norm_x(x, y, i));

            sum_x(x, y, c) += select(c == min_channel-1, weights_x(x, y, kernel),
                                weights_x(x, y, kernel) * output_y(x + kernel, y, c));

            output_x(x, y, c) = sum_x(x, y, c)/sum_x(x, y, min_channel-1);

            if (img_input.type() != img_output_type) {
                if (img_output_type == UInt(8)) {
                    if (img_input.type() == UInt(16)) {
                        img_output(x, y, c) = u16_to_u8(output_x(x, y, c));
                    }
                }
            } else {
                img_output(x, y, c) = cast(img_output_type, output_x(x, y, c));
            }
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                sigma_spatial.set_estimate(1.5f);
                sigma_range.set_estimate(0.03f);
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();

                weights_spatial
                    .compute_root()
                    .split(i, xo, xi, vector_size).vectorize(xi)
                ;
                weights_range
                    .compute_root()
                    .split(i, xo, xi, vector_size).vectorize(xi)
                ;
                if(!composable) {
                    img_output
                        .compute_root()
                        .bound(c, min_channel, num_channels)
                        .split(y, yo, yi, 32).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    intm_compute_level.set({img_output, yi});
                    intm_store_level.set({img_output, yo});
                }
                sum_x
                    .compute_at(intm_compute_level)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                sum_x.update()
                    .unroll(c)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, kernel, y)
                ;
                weights_x
                    .compute_at(sum_x, kernel)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                output_y
                    .compute_at(intm_compute_level)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                sum_y
                    .compute_at(intm_compute_level)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                sum_y.update()
                    .unroll(c)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, kernel, y)
                ;
                weights_y
                    .compute_at(sum_y, kernel)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                input_bound.compute_at(intm_compute_level).store_at(intm_store_level);
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"}, i{"i"}, j{"j"};
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"};
        Var xo_o{"xo_o"}, xo_i{"xo_i"};
        RDom kernel;

        Func input_bound{"input_bound"};
        Func weights_spatial{"weights_spatial"}, weights_range{"weights_range"};
        Func diff_y{"diff_y"}, norm_y{"norm_y"}, weights_y{"weights_y"}, sum_y{"sum_y"}, output_y{"output_y"};
        Func diff_x{"diff_x"}, norm_x{"norm_x"}, weights_x{"weights_x"}, sum_x{"sum_x"}, output_x{"output_x"};

};
