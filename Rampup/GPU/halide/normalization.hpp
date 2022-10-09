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
            } else if(get_target().has_gpu_feature()) {
                if(input.dimensions() == 2) {
                    const int num_threads = 32;
                    const int vector_size = 4;
                    Var xo{"xo"}, xi{"xi"}, xi2{"xi2"}, xi3{"xi3"};
                    Var yo{"yo"}, yi{"yi"}, yi2{"yi2"}, yi3{"yi3"};
                    Var y = output.args()[1];
                    if(get_target().has_feature(Target::OpenCL)) {
                        input.store_in(MemoryType::GPUTexture);
                    }
                    switch (scheduler)
                    {
                    default:
                    case 1:
                    case 2:
                        output.compute_root()
                            .split(x, xo, xi, std::min(num_threads*num_threads,512)*vector_size)
                            .split(xi, xi, xi2, vector_size)
                            //.reorder(xi2, xi, xo, y)
                            .gpu_blocks(xo, y)
                            .gpu_threads(xi)
                            .vectorize(xi2)
                        ;
                        break;

                    case 3:
                        output.compute_root()
                            .split(x, xo, xi, num_threads*vector_size)
                            .split(xi, xi, xi2, vector_size)
                            .split(y, yo, yi, num_threads)
                            .reorder(xi2, xi, yi, xo, yo)
                            .gpu_blocks(xo, yo)
                            .gpu_threads(xi, yi)
                            .vectorize(xi2)
                        ;
                        break;

                    case 4:
                        output.compute_root()
                            .split(x, xo, xi, num_threads)
                            .split(y, yo, yi, num_threads*vector_size)
                            .split(yi, yi, yi2, vector_size)
                            .reorder(yi2, xi, yi, xo, yo)
                            .gpu_blocks(xo, yo)
                            .gpu_threads(xi, yi)
                            .vectorize(yi2)
                        ;
                        break;
                    }
                }
            } else {
                if(input.dimensions() == 2) {
                    const int vector_size = get_target().natural_vector_size<float>();
                    Var y = output.args()[1];
                    if(out_define_schedule) {
                        output
                            .vectorize(x, vector_size)
                        ;
                        if(out_define_compute) {
                            output.compute_root()
                                .parallel(y)
                            ;
                        }
                    }
                }
            }
        }
    };

};

#endif
