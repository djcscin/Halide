#ifndef _HALIDE_LSC_MAP_
#define _HALIDE_LSC_MAP_

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideLscMap : public Generator<HalideLscMap> {
    public:
        Input<Buffer<float>> lsc_map_input{"lsc_map_input", 3};
        Input<int32_t> output_width{"output_width"};
        Input<int32_t> output_height{"output_height"};

        Output<Func> lsc_map_output{"lsc_map_output", Float(32), 2};

        GeneratorParam<bool> composable{"composable", false};
        GeneratorParam<LoopLevel> intm_compute_level{"intm_compute_level", LoopLevel::inlined()};
        GeneratorParam<LoopLevel> intm_store_level{"intm_store_level", LoopLevel::inlined()};

        void generate() {
            Expr input_width = lsc_map_input.width();
            Expr input_height = lsc_map_input.height();

            // (input_x + 0.5f) / input_width = (x + 0.5f) * 2.0f / output_width
            // input_x + 0.5f = (x + 0.5f) * input_width * 2.0f / output_width
            // input_x = (x + 0.5f) * input_width / output_width - 0.5f
            Expr input_x = (x + 0.5f) * input_width * 2.0f / output_width - 0.5f;
            Expr input_y = (y + 0.5f) * input_height * 2.0f / output_height - 0.5f;

            kernel_x = kernel(x, input_x, "kernel_x");
            kernel_y = kernel(y, input_y, "kernel_y");

            lsc_map_input_b = BoundaryConditions::repeat_edge(lsc_map_input);

            Expr ix = i32(floor(input_x));
            Expr iy = i32(floor(input_y));

            interpolation_y(x, y, c) = lsc_map_input_b(x, iy,     c) * kernel_y(y, 0)
                                     + lsc_map_input_b(x, iy + 1, c) * kernel_y(y, 1);
            interpolation_x(x, y, c) = interpolation_y(ix,     y, c) * kernel_x(x, 0)
                                     + interpolation_y(ix + 1, y, c) * kernel_x(x, 1);

            // interleave
            lsc_map_output(x, y) = select(
                (x % 2) == 0 && (y % 2) == 0, interpolation_x(x / 2, y / 2, 0),
                (x % 2) == 1 && (y % 2) == 0, interpolation_x(x / 2, y / 2, 1),
                (x % 2) == 0 && (y % 2) == 1, interpolation_x(x / 2, y / 2, 2),
                                              interpolation_x(x / 2, y / 2, 3)
            );
        }

        void schedule() {
            lsc_map_input.dim(2).set_bounds(0, 4);

            if (auto_schedule) {
                lsc_map_input.set_estimates({{0, 17}, {0, 13}, {0, 4}});
                output_width.set_estimate(4000);
                output_height.set_estimate(3000);
                lsc_map_output.set_estimates({{0, 4000}, {0, 3000}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();
                if (!composable) {
                    lsc_map_output
                        .compute_root()
                        .align_bounds(y, 2, 0)
                        .align_bounds(x, 2, 0)
                        .split(y, yo, yi, 2).unroll(yi)
                        .parallel(yo)
                        .vectorize(x, vector_size)
                    ;
                    intm_compute_level.set({lsc_map_output, yo});
                    intm_store_level.set({lsc_map_output, yo});
                }
                interpolation_y
                    .compute_at(intm_compute_level)
                    .store_at(intm_store_level)
                    .vectorize(x, vector_size, TailStrategy::RoundUp)
                ;
                kernel_x.compute_root();
                kernel_y.compute_root();
                lsc_map_input_b.compute_root();
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var yo{"yo"}, yi{"yi"};

        Func kernel_x, kernel_y;
        Func lsc_map_input_b{"lsc_map_input_b"};
        Func interpolation_x{"interpolation_x"}, interpolation_y{"interpolation_y"};

        Func kernel(Var & var, Expr input, std::string name) {
            Func output(name);

            output(var, c) = undef<float>();
            output(var, 0) = ceil(input) - input;
            output(var, 1) = 1.0f - output(var, 0);

            return output;
        }
};

#endif
