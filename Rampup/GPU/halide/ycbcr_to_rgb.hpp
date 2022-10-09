#ifndef __YCBCR_2_RGB__
#define __YCBCR_2_RGB__

#include "halide_base.hpp"
#include "color_conversion.hpp"

namespace {
    using namespace Halide;

    class YCbCr2RGB : public Generator<YCbCr2RGB>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
    public:
        Input<Func> input{"input", Float(32), 3};

        Output<Func> output{"output_y2r", Float(32), 3};

        void generate() {
            Expr yy = input(x, y, 0);
            Expr cb = input(x, y, 1);
            Expr cr = input(x, y, 2);

            Expr r = ycbcr_to_r(yy, cb, cr);
            Expr g = ycbcr_to_g(yy, cb, cr);
            Expr b = ycbcr_to_b(yy, cb, cr);

            output(x, y, c) = mux(c, {r, g, b});
        }

        void schedule() {
            if (auto_schedule) {
                input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"};
                Var co{"co"}, ci{"ci"};
                switch (scheduler)
                {
                case 2:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi2, xi, yi, xo, yo, c)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .vectorize(xi2)
                        .unroll(c)
                    ;
                    break;

                default:
                case 1:
                case 3:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi2, c, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .vectorize(xi2)
                        .unroll(c)
                    ;
                    break;

                case 4:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads)
                        .split(y, yo, yi, num_threads)
                        .reorder(c, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .split(c, co, ci, 3).vectorize(ci)
                    ;
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xo{"xo"}, xi{"xi"};
                if(out_define_schedule) {
                    output
                        .bound(c, 0, 3)
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .unroll(c)
                        .vectorize(xi)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(y)
                        ;
                    }
                }
            }
        }
    };
};

#endif
