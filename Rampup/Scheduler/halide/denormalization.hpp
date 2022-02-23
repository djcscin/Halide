#ifndef __DENORMALIZATION__
#define __DENORMALIZATION__

#include "halide_base.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class Denormalization : public Generator<Denormalization>, public HalideBase {
    private:
        Var x{"x"};
    public:
        Input<Func> input{"input"};
        Input<uint16_t> white_level{"white_level"};
        Output<Buffer<uint16_t>> output{"output_denorm"};

        void generate() {
            output(x, _) = u16_sat(f32(input(x, _)) * white_level);
        }

        void schedule() {
            if(auto_schedule) {
                if(output.dimensions() == 3) {
                    input.set_estimates({{0,4000},{0,3000},{0,3}});
                    white_level.set_estimate(255);
                    output.set_estimates({{0,4000},{0,3000},{0,3}});
                }
            } else {
                if(output.dimensions() == 3) {
                    const int vector_size = get_target().natural_vector_size<float>();
                    Var yc{"yc"};
                    Var y = output.args()[1];
                    Var c = output.args()[2];

                    output.compute_root()
                        .fuse(y, c, yc).parallel(yc)
                        .vectorize(x, vector_size)
                    ;
                }
            }
        }
    };

};

#endif
