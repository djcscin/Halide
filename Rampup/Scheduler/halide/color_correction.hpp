#ifndef __COLOR_CORRECTION__
#define __COLOR_CORRECTION__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class ColorCorrection : public Generator<ColorCorrection>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<Buffer<float>> ccm{"ccm", 2};
        Output<Func> output{"output_cc", Float(32), 3};

        void generate() {
            Expr cc = input(x, y, 0) * ccm(0, c) + input(x, y, 1) * ccm(1, c) + input(x, y, 2) * ccm(2, c);
            output(x, y, c) = clamp(cc, 0.f, 1.f);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000},{0,3}});
                ccm.set_estimates({{0,3},{0,3}});
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xo{"xo"}, xi{"xi"};

                output.compute_root()
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, y)
                    .parallel(y)
                ;
            }
        }
    };

};

#endif
