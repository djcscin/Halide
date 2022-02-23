#ifndef __REINHARD_TONE_MAPPING__
#define __REINHARD_TONE_MAPPING__

#include "halide_base.hpp"
#include "color_conversion.hpp"

namespace {
    using namespace Halide;

    class ReinhardToneMapping : public Generator<ReinhardToneMapping>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func gray{"gray"}, max_gray{"max_gray"}, gain{"gain"};
        RDom r_max_gray;
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<int> width{"width"};
        Input<int> height{"height"};
        Output<Func> output{"output_rtm", Float(32), 3};

        void generate() {
            gray(x, y) = rgb_to_gray(input(x, y, 0), input(x, y, 1), input(x, y, 2));

            r_max_gray = RDom(0, width, 0, height, "r_max_gray");
            max_gray() = 1.e-5f;
            max_gray() = max(max_gray(), gray(r_max_gray.x, r_max_gray.y));

            Expr l = gray(x, y);
            Expr l_max = max_gray();
            gain(x, y) = (1.f + (l / (l_max * l_max))) / (1.f + l);

            output(x, y, c) = input(x, y, c) * gain(x, y);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000},{0,3}});
                width.set_estimate(4000);
                height.set_estimate(3000);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xo{"xo"}, xi{"xi"};

                output.compute_root()
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, y)
                    .parallel(y)
                ;
                max_gray.compute_root();
            }
        }
    };

};

#endif
