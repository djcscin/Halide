#ifndef __NORMALIZATION__
#define __NORMALIZATION__

#include "halide_base.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class Normalization : public Generator<Normalization>, public HalideBase {
    private:
        Var x{"x"};
    public:
        Input<Buffer<uint16_t>> input{"input"};
        Input<uint16_t> white_level{"white_level"};
        Output<Func> output{"output_norm"};

        void generate() {
            output(x, _) = min(1.f, f32(input(x, _)) / white_level);
        }

        void schedule() {
            if(auto_schedule) {
                if(input.dimensions() == 2) {
                    input.set_estimates({{0,4000},{0,3000}});
                    white_level.set_estimate(1023);
                    output.set_estimates({{0,4000},{0,3000}});
                }
            } else {
                if(input.dimensions() == 2) {
                    const int vector_size = get_target().natural_vector_size<float>();
                    Var y = output.args()[1];

                    output.compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                }
            }
        }
    };

};

#endif
