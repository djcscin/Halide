#ifndef __WHITE_BALANCE__
#define __WHITE_BALANCE__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class WhiteBalance : public Generator<WhiteBalance>, public HalideBase {
    private:
        Var x{"x"}, y{"y"};
    public:
        Input<Func> input{"input", Float(32), 2};
        Input<Buffer<float>> white_balance{"white_balance", 1};
        Output<Func> output{"output_wb", Float(32), 2};

        void generate() {
            output(x, y) = min(1.f, input(x, y) * white_balance((x % 2) + (y % 2)*2));
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000}});
                white_balance.set_estimates({{0,4}});
                output.set_estimates({{0,4000},{0,3000}});
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));

                output.compute_root()
                    .parallel(y)
                    .vectorize(x, vector_size)
                ;
            }
        }
    };

};

#endif
