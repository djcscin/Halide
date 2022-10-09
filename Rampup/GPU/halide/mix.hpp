#ifndef __MIX__
#define __MIX__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class Mix : public Generator<Mix>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
    public:
        Input<Func> luma_input{"luma_input", Float(32), 3};
        Input<Func> chroma_input{"chroma_input", Float(32), 3};

        Output<Func> output{"output_mix", Float(32), 3};

        void generate() {
            output(x, y, c) = mux(c, {luma_input(x, y, 0), chroma_input(x, y, 1), chroma_input(x, y, 2)});
        }

        void schedule() {
            if(auto_schedule) {
                luma_input.set_estimates({{0, 4000}, {0, 3000}, {0, 1}});
                chroma_input.set_estimates({{0, 4000}, {0, 3000}, {1, 2}});
                output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"};
                switch (scheduler)
                {
                default:
                case 1:
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
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                if(out_define_schedule) {
                    output
                        .bound(c, 0, 3)
                        .unroll(c)
                        .vectorize(x, vector_size)
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
