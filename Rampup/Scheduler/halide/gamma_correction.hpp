#ifndef __GAMMA_CORRECTION__
#define __GAMMA_CORRECTION__

#include "halide_base.hpp"
#include "constants.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class GammaCorrection : public Generator<GammaCorrection>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func lut_gamma{"lut_gamma"};
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<float> gamma{"gamma"};
        Output<Func> output{"output_gc", Float(32), 3};

        void generate() {
            lut_gamma(c) = pow(f32(c)/max16_f32, 1.f/gamma);
            output(x, y, c) = lut_gamma(u16_sat(input(x, y, c) * max16_f32));
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000},{0,3}});
                gamma.set_estimate(2.2f);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size = 256;
                Var yc{"yc"};
                Var co{"co"}, ci{"ci"};

                output.compute_root()
                    .fuse(y, c, yc).parallel(yc)
                    .vectorize(x, vector_size)
                ;
                lut_gamma.compute_root()
                    .split(c, co, ci, parallel_size)
                    .parallel(co)
                    .vectorize(ci, vector_size)
                ;
            }
        }
    };

};

#endif
