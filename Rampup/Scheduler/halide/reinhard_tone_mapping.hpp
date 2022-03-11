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
                const int parallel_size = 8;
                Var xo{"xo"}, xi{"xi"};
                RVar ryo{"ryo"}, ryi{"ryi"};
                Func max_gray_intm;

                switch (scheduler)
                {
                case 2:
                    output.compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    max_gray.compute_root();
                    break;

                case 3:
                    output.compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    max_gray.compute_root();
                    gain.compute_at(output, xo)
                        .vectorize(x, vector_size)
                    ;
                    break;

                case 4:
                    output.compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    max_gray.compute_root();
                    gain.compute_at(output, y)
                        .vectorize(x, vector_size)
                    ;
                    break;

                case 5:
                    output.compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    max_gray_intm = max_gray.update().rfactor(r_max_gray.y, y);
                    // max_gray_intm(y) = 1e-5;
                    // max_gray_intm(y) = max(gray(r_max_gray.x, y), max_gray_intm(y));
                    // max_gray() = 1e-5;
                    // max_gray() = max(max_gray_intm(r_max_gray.y), max_gray());
                    max_gray_intm.compute_root()
                        .parallel(y)
                    ;
                    max_gray_intm.update()
                        .parallel(y)
                    ;
                    max_gray.compute_root();
                    break;

                case 6:
                case 1:
                default:
                    output.compute_root()
                        .split(x, xo, xi, vector_size)
                        .reorder(xi, c, xo, y)
                        .parallel(y)
                    ;
                    max_gray_intm = max_gray.update().split(r_max_gray.y, ryo, ryi, parallel_size).rfactor(ryo, y);
                    // max_gray_intm(y) = 1e-5;
                    // max_gray_intm(y) = max(gray(r_max_gray.x, y*parallel_size + ryi), max_gray_intm(y));
                    // max_gray() = 1e-5;
                    // max_gray() = max(max_gray_intm(r_max_gray.y), max_gray());
                    max_gray_intm.compute_root()
                        .parallel(y)
                    ;
                    max_gray_intm.update()
                        .parallel(y)
                    ;
                    max_gray.compute_root();
                    break;
                }

            }
        }
    };

};

#endif
