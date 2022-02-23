#ifndef __MIX__
#define __MIX__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class Mix : public Generator<Mix>, HalideBase {
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
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                output.compute_root()
                    .bound(c, 0, 3)
                    .unroll(c)
                    .parallel(y)
                    .vectorize(x, vector_size)
                ;
            }
        }
    };
};

#endif
