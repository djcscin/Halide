#ifndef __BILINEAR_RESIZE__
#define __BILINEAR_RESIZE__

#include "halide_base.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class BilinearResize : public Generator<BilinearResize>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func kernel_x, kernel_y;
        Func input_bound{"input_bound"}, interpolation_x{"interpolation_x_br"}, interpolation_y{"interpolation_y_br"};
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<int> input_width{"input_width"};
        Input<int> input_height{"input_height"};
        Input<int> output_width{"output_width"};
        Input<int> output_height{"output_height"};
        Output<Func> output{"output_br", Float(32), 3};

        void generate() {
            // (input_x + 0.5f) / input_width = (x + 0.5f) / output_width
            // input_x + 0.5f = (x + 0.5f) * input_width / output_width
            // input_x = (x + 0.5f) * input_width / output_width - 0.5f
            Expr input_x = (x + 0.5f) * input_width / output_width - 0.5f;
            Expr input_y = (y + 0.5f) * input_height / output_height - 0.5f;
            Expr ix = i32(floor(input_x));
            Expr iy = i32(floor(input_y));

            kernel_x = kernel(x, input_x, "kernel_x");
            kernel_y = kernel(y, input_y, "kernel_y");

            input_bound = BoundaryConditions::repeat_edge(input, {{0, input_width}, {0, input_height}});
            interpolation_y(x, y, c) = input_bound(x, iy,     c) * kernel_y(y, 0)
                                     + input_bound(x, iy + 1, c) * kernel_y(y, 1);
            interpolation_x(x, y, c) = interpolation_y(ix,     y, c) * kernel_x(x, 0)
                                     + interpolation_y(ix + 1, y, c) * kernel_x(x, 1);

            output(x, y, c) = interpolation_x(x, y, c);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,17},{0,13},{0,4}});
                input_width.set_estimate(17);
                input_height.set_estimate(13);
                output_width.set_estimate(4000);
                output_height.set_estimate(3000);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                const int parallel_size = 128;

                Var xi{"xi"}, xo{"xo"};
                Var yi{"yi"}, yo{"yo"};
                Var yc{"yc"};

                output.compute_root()
                    .fuse(y, c, yc).parallel(yc)
                    .vectorize(x, vector_size)
                ;

                interpolation_y.compute_at(output, yc)
                    .vectorize(x, vector_size)
                ;

                kernel_x.compute_root();
                kernel_x.update(0)
                    .split(x, xo, xi, parallel_size)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                kernel_x.update(1)
                    .split(x, xo, xi, parallel_size)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;

                kernel_y.compute_root();
                kernel_y.update(0)
                    .split(y, yo, yi, parallel_size)
                    .parallel(yo)
                    .vectorize(yi, vector_size)
                ;
                kernel_y.update(1)
                    .split(y, yo, yi, parallel_size)
                    .parallel(yo)
                    .vectorize(yi, vector_size)
                ;
            }
        }

    private:
        Func kernel(Var & var, Expr input, std::string name) {
            Func output(name);

            output(var, c) = undef<float>();
            output(var, 0) = ceil(input) - input;
            output(var, 1) = 1.0f - output(var, 0);

            return output;
        }
    };

};

#endif
