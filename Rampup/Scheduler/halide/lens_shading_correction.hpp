#ifndef __LENS_SHADING_CORRECTION__
#define __LENS_SHADING_CORRECTION__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class LensShadingCorrection : public Generator<LensShadingCorrection>, public HalideBase {
    private:
        Var x{"x"}, y{"y"};
    public:
        Input<Func> input{"input", Float(32), 2};
        Input<Func> lsc_map{"lsc_map", Float(32), 3};
        Output<Func> output{"output_lsc", Float(32), 2};

        void generate() {
            output(x, y) = min(1.f, input(x, y) * lsc_map(x/2, y/2, (x % 2) + (y % 2)*2));
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000}});
                lsc_map.set_estimates({{0,2000},{0,1500},{0,4}});
                output.set_estimates({{0,4000},{0,3000}});
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                Var xo("xo"), xi("xi"), yo("yo"), yi("yi");

                switch (scheduler)
                {
                case 2:
                    output.compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    break;

                case 3:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, 2).unroll(xi)
                        .split(y, yo, yi, 2).unroll(yi)
                        .parallel(yo)
                        .vectorize(xo, vector_size)
                    ;
                    break;

                case 1:
                default:
                    if(out_define_schedule) {
                        output
                            .align_bounds(x, 2, 0)
                            .align_bounds(y, 2, 0)
                            .split(x, xo, xi, 2).unroll(xi)
                            .split(y, yo, yi, 2).unroll(yi)
                            .vectorize(xo, vector_size)
                        ;
                        if(out_define_compute) {
                            output.compute_root()
                                .parallel(yo)
                            ;
                        }
                    }
                    break;
                }

            }
        }
    };

};

#endif
