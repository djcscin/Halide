#ifndef __RGB_2_YCBCR__
#define __RGB_2_YCBCR__

#include "halide_base.hpp"
#include "color_conversion.hpp"

namespace {
    using namespace Halide;

    class RGB2YCbCr : public Generator<RGB2YCbCr>, HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
    public:
        Input<Func> input{"input", Float(32), 3};

        Output<Func> output{"output_r2y", Float(32), 3};

        void generate() {
            Expr r = input(x, y, 0);
            Expr g = input(x, y, 1);
            Expr b = input(x, y, 2);

            Expr yy = rgb_to_gray(r, g, b);
            Expr cb = rgb_to_cb(r, g, b);
            Expr cr = rgb_to_cr(r, g, b);

            output(x, y, c) = mux(c, {yy, cb, cr});
        }

        void schedule() {
            if (auto_schedule) {
                input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xo{"xo"}, xi{"xi"};
                output.compute_root()
                    .bound(c, 0, 3)
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, y)
                    .unroll(c)
                    .parallel(y)
                    .vectorize(xi)
                ;
            }
        }
    };
};

#endif
