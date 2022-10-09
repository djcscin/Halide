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
            } else if (get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int vector_size = 4;
                const int parallel_size = 256;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"};
                Var co{"co"}, ci{"ci"}, ci2{"ci2"};

                output.compute_root()
                    .split(x, xo, xi, num_threads*vector_size)
                    .split(xi, xi, xi2, vector_size)
                    .split(y, yo, yi, num_threads)
                    .reorder(xi2, xi, yi, xo, yo)
                    .gpu_blocks(xo, yo)
                    .gpu_threads(xi, yi)
                    .vectorize(xi2)
                ;

                switch (scheduler)
                {
                case 2:
                    break;

                case 3: //compute at CPU
                    lut_gamma.compute_root()
                        .split(c, co, ci, parallel_size)
                        .parallel(co)
                        .vectorize(ci, vector_size)
                    ;
                    break;

                default:
                case 1:
                case 4: //compute at GPU globally
                    lut_gamma.compute_root()
                        .split(c, co, ci, num_threads*num_threads*vector_size)
                        .split(ci, ci, ci2, vector_size)
                        .gpu_blocks(co)
                        .gpu_threads(ci)
                        .vectorize(ci2)
                    ;
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size = 256;
                Var yc{"yc"};
                Var co{"co"}, ci{"ci"};

                if(out_define_schedule) {
                    output
                        .vectorize(x, vector_size)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .fuse(y, c, yc).parallel(yc)
                        ;
                    }
                }
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
