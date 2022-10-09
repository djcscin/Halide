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
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"}, xi3{"xi3"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"}, yi3{"yi3"};
                if(get_target().has_feature(Target::OpenCL)) {
                    white_balance.store_in(MemoryType::GPUTexture);
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
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, std::min(num_threads*num_threads,256)*vector_size*2)
                        .split(xi, xi, xi2, vector_size*2)
                        .split(xi2, xi2, xi3, 2)
                        .split(y, yo, yi, 2)
                        .reorder(xi3, yi, xi2, xi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi)
                        .vectorize(xi2)
                        .unroll(yi)
                        .unroll(xi3)
                    ;
                    break;

                case 4:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, std::min(num_threads*num_threads,512)*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, 2)
                        .reorder(yi, xi2, xi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi)
                        .vectorize(xi2)
                        .unroll(yi)
                    ;
                    break;

                case 5:
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

                case 6:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, num_threads*vector_size*2)
                        .split(xi, xi, xi2, vector_size*2)
                        .split(xi2, xi2, xi3, 2)
                        .split(y, yo, yi, num_threads*2)
                        .split(yi, yi, yi2, 2)
                        .reorder(xi3, yi2, xi2, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(xi3)
                    ;
                    break;

                case 7:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads*2)
                        .split(yi, yi, yi2, 2)
                        .reorder(yi2, xi2, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                    ;
                    break;

                case 8:
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

                case 9:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, num_threads*2)
                        .split(xi, xi, xi2, 2)
                        .split(y, yo, yi, num_threads*vector_size*2)
                        .split(yi, yi, yi2, vector_size*2)
                        .split(yi2, yi2, yi3, 2)
                        .reorder(xi2, yi3, yi2, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                        .unroll(yi3)
                        .unroll(xi2)
                    ;
                    break;

                case 10:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, num_threads)
                        .split(y, yo, yi, num_threads*vector_size*2)
                        .split(yi, yi, yi2, vector_size*2)
                        .split(yi2, yi2, yi3, 2)
                        .reorder(yi3, yi2, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                        .unroll(yi3)
                    ;
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                Var xo("xo"), xi("xi"), yo("yo"), yi("yi");
                if(out_define_schedule) {
                    output
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .split(x, xo, xi, 2).unroll(xi)
                        .split(y, yo, yi, 2).unroll(yi)
                        .vectorize(xo, vector_size)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(yo)
                        ;
                    }
                }
            }
        }
    };

};

#endif
