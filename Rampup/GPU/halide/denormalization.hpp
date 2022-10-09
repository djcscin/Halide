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
            } else if(get_target().has_gpu_feature()) {
                if(output.dimensions() == 3) {
                    const int num_threads_x = 128;
                    const int num_threads_y = 8;
                    const int vector_size = 4;
                    Var xo{"xo"}, xi{"xi"}, xi2{"xi2"}, xi3{"xi3"};
                    Var yo{"yo"}, yi{"yi"}, yi2{"yi2"}, yi3{"yi3"};
                    Var y = output.args()[1];
                    Var c = output.args()[2];
                    switch (scheduler)
                    {
                    default:
                    case 1:
                    case 2:
                        output.compute_root()
                            .split(x, xo, xi, std::min(num_threads_x*num_threads_y,512)*vector_size)
                            .split(xi, xi, xi2, vector_size)
                            //.reorder(xi2, xi, xo, y, c)
                            .gpu_blocks(xo, y, c)
                            .gpu_threads(xi)
                            .vectorize(xi2)
                        ;
                        break;

                    case 3:
                        output.compute_root()
                            .split(x, xo, xi, num_threads_x*vector_size)
                            .split(xi, xi, xi2, vector_size)
                            .split(y, yo, yi, num_threads_y)
                            .reorder(xi2, xi, yi, xo, yo, c)
                            .gpu_blocks(xo, yo, c)
                            .gpu_threads(xi, yi)
                            .vectorize(xi2)
                        ;
                        break;

                    case 4:
                        output.compute_root()
                            .split(x, xo, xi, num_threads_x)
                            .split(y, yo, yi, num_threads_y*vector_size)
                            .split(yi, yi, yi2, vector_size)
                            .reorder(yi2, xi, yi, xo, yo, c)
                            .gpu_blocks(xo, yo, c)
                            .gpu_threads(xi, yi)
                            .vectorize(yi2)
                        ;
                        break;

                    case 5:
                        output.compute_root()
                            .split(x, xo, xi, num_threads_x)
                            .split(y, yo, yi, num_threads_y)
                            .reorder(c, xi, yi, xo, yo)
                            .gpu_blocks(xo, yo)
                            .gpu_threads(xi, yi)
                            .bound(c, 0, 3)
                            .vectorize(c, 3)
                        ;
                        break;
                    }
                }
            } else {
                if(output.dimensions() == 3) {
                    const int vector_size = get_target().natural_vector_size<float>();
                    Var yc{"yc"};
                    Var y = output.args()[1];
                    Var c = output.args()[2];

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
                }
            }
        }
    };

};

#endif
