#ifndef __BILINEAR_RESIZE__
#define __BILINEAR_RESIZE__

#include "halide_base.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class BilinearResize : public Generator<BilinearResize>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func kernel_x, kernel_y;
        Func input_bound{"input_bound"}, interpolation_x{"interpolation_x_br"}, interpolation_y{"interpolation_y_br"};
    public:
        Input<Buffer<>> input{"input", Float(32), 3};
        Input<int> input_width{"input_width"};
        Input<int> input_height{"input_height"};
        Input<int> output_width{"output_width"};
        Input<int> output_height{"output_height"};
        Output<Func> output{"output_br", Float(32), 3};

        void generate() {
            // (input_x + 0.5f) / input_width = (x + 0.5f) / output_width
            // input_x + 0.5f = (x + 0.5f) * input_width / output_width
            // input_x = (x + 0.5f) * input_width / output_width - 0.5f
            Expr input_x = (x + 0.5f) * input_width / output_width - 0.5f;
            Expr input_y = (y + 0.5f) * input_height / output_height - 0.5f;
            Expr ix = i32(floor(input_x));
            Expr iy = i32(floor(input_y));

            kernel_x = kernel(x, input_x, "fkernel_x");
            kernel_y = kernel(y, input_y, "fkernel_y");

            input_bound = BoundaryConditions::repeat_edge(input, {{0, input_width}, {0, input_height}});
            interpolation_y(x, y, c) = input_bound(x, iy,     c) * kernel_y(y, 0)
                                     + input_bound(x, iy + 1, c) * kernel_y(y, 1);
            interpolation_x(x, y, c) = interpolation_y(ix,     y, c) * kernel_x(x, 0)
                                     + interpolation_y(ix + 1, y, c) * kernel_x(x, 1);

            output(x, y, c) = interpolation_x(x, y, c);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,17},{0,13},{0,4}});
                input_width.set_estimate(17);
                input_height.set_estimate(13);
                output_width.set_estimate(4000);
                output_height.set_estimate(3000);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int num_threads_x = size_factor;
                const int vector_size = 4;
                const int unroll_size = 4;
                const int parallel_size_1d = 256;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"}, xi3{"xi3"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"}, yi3{"yi3"};

                if(get_target().has_feature(Target::OpenCL)) {
                    input.store_in(MemoryType::GPUTexture);
                }

                switch(scheduler) {
                case 3: // compute at CPU
                    kernel_x.compute_root()
                        .split(x, xo, xi, parallel_size_1d)
                        .split(xi, xi, xi2, vector_size)
                        .reorder(xi2, c, xi, xo)
                        .parallel(xo)
                        .vectorize(xi2)
                    ;
                    kernel_y.compute_root()
                        .split(y, yo, yi, parallel_size_1d)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, yi, yo)
                        .parallel(yo)
                        .vectorize(yi2)
                    ;
                    break;

                case 4: // compute global at GPU
                    kernel_x.compute_root()
                        .split(x, xo, xi, num_threads*num_threads)
                        .reorder(c, xi, xo)
                        .gpu_blocks(xo)
                        .gpu_threads(xi)
                    ;
                    kernel_y.compute_root()
                        .split(y, yo, yi, num_threads*num_threads)
                        .reorder(c, yi, yo)
                        .gpu_blocks(yo)
                        .gpu_threads(yi)
                    ;
                    break;

                case 5: // compute at GPU blocks
                    kernel_x.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, num_threads)
                        .reorder(c, xi, xo)
                        .gpu_threads(xi)
                    ;
                    kernel_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(y, yo, yi, num_threads*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, yi, yo)
                        .gpu_threads(yi)
                        .vectorize(yi2)
                    ;
                    break;

                case 6: // compute at GPU threads
                    kernel_x.compute_at(output, xi).store_in(MemoryType::Register);
                    kernel_y.compute_at(output, xi).store_in(MemoryType::Register)
                        .vectorize(y, vector_size)
                    ;
                    break;

                default: // do not compute kernel_x or kernel_y
                    break;
                }

                switch (scheduler)
                {
                default:
                case 1:
                case 2:
                case 3:
                case 4:
                case 5:
                case 6:
                case 13:
                case 14:
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

                case 7:
                case 8:
                case 9:
                case 10:
                case 11:
                case 12:
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
                }

                switch (scheduler)
                {
                case 8: // compute at GPU globally
                    interpolation_y.compute_root()
                        .split(x, xo, xi, vector_size*num_threads_x)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads*num_threads/num_threads_x)
                        .reorder(xi2, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                    ;
                    break;

                case 9: // compute at GPU blocks
                    interpolation_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi2, xi, yi, xo, yo)
                        .unroll(yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                    ;
                    break;

                case 10: // compute at GPU blocks
                    interpolation_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, vector_size)
                        .split(y, yo, yi, num_threads)
                        // .reorder(xi, xo, yi, yo)
                        .unroll(yo)
                        .gpu_threads(yi)
                        .vectorize(xi)
                    ;
                    break;

                case 11: // compute at GPU threads with vectorize
                    interpolation_y.compute_at(output, xi).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .unroll(y)
                    ;
                    break;

                case 12: // compute at GPU threads without vectorize
                    interpolation_y.compute_at(output, xi).store_in(MemoryType::GPUShared)
                        .unroll(y)
                    ;
                    break;

                case 13: // compute at GPU threads
                    interpolation_y.compute_at(output, xi).store_in(MemoryType::GPUShared)
                        .split(y, yo, yi, vector_size)
                        .vectorize(yi)
                    ;
                    break;

                case 14: // compute at GPU blocks
                    interpolation_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, num_threads)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi, yi, xo, yo)
                        .unroll(yo)
                        .gpu_threads(xi, yo)
                    ;
                    break;

                default:
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size(Float(32));
                const int parallel_size = 128;

                Var xi{"xi"}, xo{"xo"};
                Var yi{"yi"}, yo{"yo"};
                Var yc{"yc"};

                if(out_define_schedule) {
                    output
                        .vectorize(x, vector_size)
                        .fuse(y, c, yc)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(yc)
                        ;
                    }
                    intm_compute_level.set({output, yc});
                }
                interpolation_y.compute_at(intm_compute_level)
                    .vectorize(x, vector_size)
                ;
                kernel_x.compute_root();
                kernel_x.update(0)
                    .split(x, xo, xi, parallel_size)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                kernel_x.update(1)
                    .split(x, xo, xi, parallel_size)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                kernel_y.compute_root();
                kernel_y.update(0)
                    .split(y, yo, yi, parallel_size)
                    .parallel(yo)
                    .vectorize(yi, vector_size)
                ;
                kernel_y.update(1)
                    .split(y, yo, yi, parallel_size)
                    .parallel(yo)
                    .vectorize(yi, vector_size)
                ;
            }
        }
    private:
        Func kernel(Var & var, Expr input, std::string name) {
            Func output(name);

            if(get_target().has_gpu_feature()) {
                output(var, c) = select(c == 0, ceil(input) - input, 1.f - ceil(input) + input);
            } else {
                output(var, c) = undef<float>();
                output(var, 0) = ceil(input) - input;
                output(var, 1) = 1.0f - output(var, 0);
            }

            return output;
        }
    };
};

#endif
