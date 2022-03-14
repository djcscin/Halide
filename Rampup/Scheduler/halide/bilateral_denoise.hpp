#ifndef __BILATERAL_DENOISE__
#define __BILATERAL_DENOISE__

#include "halide_base.hpp"
#include "constants.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class BilateralDenoise : public Generator<BilateralDenoise>, HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"}, i{"i"}, j{"j"};
        RDom kernel;
        Func input_bound{"input_bound"}, guide_bound{"guide_bound"};
        Func weights_spatial{"weights_spatial"}, weights_range{"weights_range"};
        Func diff_y{"diff_y"}, norm_y{"norm_y"}, weights_y{"weights_y"}, sum_y{"sum_y"}, output_y{"output_y"};
        Func diff_x{"diff_x"}, norm_x{"norm_x"}, weights_x{"weights_x"}, sum_x{"sum_x"}, output_x{"output_x"};

        Expr gaussian(Expr i, Expr sigma) {
            return exp(-(i*i)/(2.f*sigma*sigma));
        }
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<Func> guide{"guide", Float(32), 3};
        Input<int32_t> width{"width"};
        Input<int32_t> height{"height"};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};

        Output<Func> output{"output_bd", Float(32), 3};

        GeneratorParam<int32_t> channel_min{"channel_min", 1};
        GeneratorParam<int32_t> channel_extent{"channel_extent", 2};

        void generate() {
            input_bound = BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}, {0, 3}});
            guide_bound = BoundaryConditions::repeat_edge(guide, {{0, width}, {0, height}, {0, 3}});

            Expr gaussian_width = max(i32(3.f*sigma_spatial), 1);
            Expr kernel_size = 2*gaussian_width + 1;
            kernel = RDom(-gaussian_width, kernel_size, "kernel");

            weights_spatial(i) = gaussian(i, sigma_spatial);
            weights_range(i) = gaussian(f32(i)/(3.f*max14_f32), sigma_range);

            diff_y(x, y, i, c) = absd(guide_bound(x, y, c), guide_bound(x, y + i, c));
            norm_y(x, y, i) = u16_sat((diff_y(x, y, i, 0) + diff_y(x, y, i, 1) + diff_y(x, y, i, 2))*max14_f32);
            weights_y(x, y, i) = weights_spatial(i) * weights_range(norm_y(x, y, i));

            // sum_y(x, y, c) = 0.f;
            sum_y(x, y, c) += select(c == channel_min-1, weights_y(x, y, kernel),
                                weights_y(x, y, kernel) * input_bound(x, y + kernel, c));

            output_y(x, y, c) = sum_y(x, y, c)/sum_y(x, y, channel_min-1);

            diff_x(x, y, i, c) = absd(guide_bound(x, y, c), guide_bound(x + i, y, c)); //inline?
            norm_x(x, y, i) = u16_sat((diff_x(x, y, i, 0) + diff_x(x, y, i, 1) + diff_x(x, y, i, 2))*max14_f32); //inline?
            weights_x(x, y, i) = weights_spatial(i) * weights_range(norm_x(x, y, i)); //compute?

            // sum_x(x, y, c) = 0.f;
            sum_x(x, y, c) += select(c == channel_min-1, weights_x(x, y, kernel), //compute
                                weights_x(x, y, kernel) * output_y(x + kernel, y, c));

            output_x(x, y, c) = sum_x(x, y, c)/sum_x(x, y, channel_min-1); //inline

            output(x, y, c) = clamp(output_x(x, y, c), 0.f, 1.f); //compute_root
        }

        void schedule() {
            if (auto_schedule) {
                input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                guide.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                sigma_spatial.set_estimate(5.f);
                sigma_range.set_estimate(0.05f);
                output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size = 256;
                Var xo{"xo"}, xi{"xi"};

                switch (scheduler)
                {
                case 3:
                    weights_x.compute_at(output, xo)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, i, xo, y)
                        .vectorize(xi)
                    ;
                    weights_y.compute_at(output, y)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, xo, i, y)
                        .vectorize(xi)
                    ;
                    goto scheduler_2;

                case 4:
                    weights_x.compute_at(sum_x, xo)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, i, xo, y)
                        .vectorize(xi)
                    ;
                    weights_y.compute_at(sum_y, kernel)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, xo, i, y)
                        .vectorize(xi)
                    ;
                    goto scheduler_2;

                case 5:
                case 1:
                default:
                    weights_x.compute_at(sum_x, kernel)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, i, xo, y)
                        .vectorize(xi)
                    ;
                    weights_y.compute_at(sum_y, xo)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, xo, i, y)
                        .vectorize(xi)
                    ;

                case 2:
                scheduler_2:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                        .vectorize(xi)
                    ;
                    weights_spatial.compute_root()
                        .split(i, xo, xi, parallel_size)
                        .parallel(xo)
                        .vectorize(xi, vector_size)
                    ;
                    weights_range.compute_root()
                        .split(i, xo, xi, parallel_size)
                        .parallel(xo)
                        .vectorize(xi, vector_size)
                    ;
                    sum_x.compute_at(output, xo)
                        .vectorize(x, vector_size)
                    ;
                    sum_x.update(0)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, kernel, xo, y)
                        .unroll(c)
                        .vectorize(xi)
                    ;
                    sum_y.compute_at(output, y)
                        .vectorize(x, vector_size)
                    ;
                    sum_y.update(0)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, kernel, y)
                        .unroll(c)
                        .vectorize(xi)
                    ;
                    break;
                }
            }
        }
    };
};

#endif
