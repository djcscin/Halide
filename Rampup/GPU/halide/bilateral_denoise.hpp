#ifndef __BILATERAL_DENOISE__
#define __BILATERAL_DENOISE__

#include "halide_base.hpp"
#include "constants.hpp"

namespace {
    using namespace Halide;
    using namespace Halide::ConciseCasts;

    class BilateralDenoise : public Generator<BilateralDenoise>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"}, i{"i"}, j{"j"};
        RDom kernel;
        Func input_bound{"input_bound"}, guide_bound{"guide_bound"};
        Func weights_spatial{"weights_spatial"}, weights_range{"weights_range"};
        Func diff_y{"diff_y"}, norm_y{"norm_y"}, weights_y{"weights_y"}, sum_y{"sum_y"}, output_y{"output_y"};
        Func diff_x{"diff_x"}, norm_x{"norm_x"}, weights_x{"weights_x"}, sum_x{"sum_x"}, output_x{"output_x"};

        Expr gaussian(Expr i, Expr sigma) {
            return exp(-(i*i)/(2.f*sigma*sigma));
        }
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<Func> guide{"guide", Float(32), 3};
        Input<int32_t> width{"width"};
        Input<int32_t> height{"height"};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};

        Output<Func> output{"output_bd", Float(32), 3};

        GeneratorParam<int32_t> channel_min{"channel_min", 1};
        GeneratorParam<int32_t> channel_extent{"channel_extent", 2};
        GeneratorParam<LoopLevel> sum_x_compute_level{"sum_x_compute_level", LoopLevel::inlined()};
        GeneratorParam<LoopLevel> sum_y_compute_level{"sum_y_compute_level", LoopLevel::inlined()};

        void generate() {
            input_bound = BoundaryConditions::repeat_edge(input, {{0, width}, {0, height}, {0, 3}});
            guide_bound = BoundaryConditions::repeat_edge(guide, {{0, width}, {0, height}, {0, 3}});

            Expr gaussian_width = clamp(i32(3.f*sigma_spatial), 1, 15);
            Expr kernel_size = 2*gaussian_width + 1;
            kernel = RDom(-gaussian_width, kernel_size, "kernel");

            weights_spatial(i) = gaussian(i, sigma_spatial);
            weights_range(i) = gaussian(f32(i)/(3.f*max14_f32), sigma_range);

            diff_y(x, y, i, c) = absd(guide_bound(x, y, c), guide_bound(x, y + i, c));
            norm_y(x, y, i) = u16_sat((diff_y(x, y, i, 0) + diff_y(x, y, i, 1) + diff_y(x, y, i, 2))*max14_f32);
            weights_y(x, y, i) = weights_spatial(i) * weights_range(norm_y(x, y, i));

            // sum_y(x, y, c) = 0.f;
            sum_y(x, y, c) += select(c == channel_min-1, weights_y(x, y, kernel),
                                weights_y(x, y, kernel) * input_bound(x, y + kernel, clamp(c, channel_min, channel_min-1 + channel_extent)));

            output_y(x, y, c) = sum_y(x, y, c)/sum_y(x, y, channel_min-1);

            diff_x(x, y, i, c) = absd(guide_bound(x, y, c), guide_bound(x + i, y, c)); //inline?
            norm_x(x, y, i) = u16_sat((diff_x(x, y, i, 0) + diff_x(x, y, i, 1) + diff_x(x, y, i, 2))*max14_f32); //inline?
            weights_x(x, y, i) = weights_spatial(i) * weights_range(norm_x(x, y, i)); //compute?

            // sum_x(x, y, c) = 0.f;
            sum_x(x, y, c) += select(c == channel_min-1, weights_x(x, y, kernel), //compute
                                weights_x(x, y, kernel) * output_y(x + kernel, y, clamp(c, channel_min, channel_min-1 + channel_extent)));

            output_x(x, y, c) = sum_x(x, y, c)/sum_x(x, y, channel_min-1); //inline

            output(x, y, c) = clamp(output_x(x, y, c), 0.f, 1.f); //compute_root
        }

        void schedule() {
            if (auto_schedule) {
                input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                guide.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                sigma_spatial.set_estimate(5.f);
                sigma_range.set_estimate(0.05f);
                output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if(get_target().has_gpu_feature()) {
                const int mscheduler = (scheduler==0)?12:scheduler;
                const int parallel_size_1d = 256;
                const int num_threads_x = 64;
                const int num_threads_y = 16;
                const int num_threads_1d = std::min(num_threads_x*num_threads_y,512);
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"};

                weights_spatial.compute_root()
                    .split(i, xo, xi, parallel_size_1d)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                weights_range.compute_root()
                    .split(i, xo, xi, parallel_size_1d)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                switch (mscheduler)
                {
                case 2:
                case 3:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_1d*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .reorder(xi2, c, xi, xo, y)
                        .gpu_blocks(xo, y)
                        .gpu_threads(xi)
                        .vectorize(xi2)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_1d*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .reorder(xi2, c, xi, xo, y)
                        .gpu_blocks(xo, y)
                        .gpu_threads(xi)
                        .vectorize(xi2)
                    ;
                    break;

                case 4:
                case 5:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x*num_threads_y)
                        .reorder(c, xi, xo, y)
                        .gpu_blocks(xo, y)
                        .gpu_threads(xi)
                        .vectorize(c)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x*num_threads_y)
                        .reorder(c, xi, xo, y)
                        .gpu_blocks(xo, y)
                        .gpu_threads(xi)
                        .vectorize(c)
                    ;
                    break;

                case 6:
                case 7:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads_y/((scheduler==6)?2:1))
                        .reorder(xi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x*vector_size/((scheduler==6)?2:1))
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads_y)
                        .reorder(xi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                    ;
                    break;

                case 8:
                case 9:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y*vector_size/((scheduler==8)?2:1))
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x/((scheduler==8)?2:1))
                        .split(y, yo, yi, num_threads_y*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    break;

                case 10:
                case 11:
                    output.compute_root()
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y)
                        .reorder(c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(c)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y)
                        .reorder(c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(c)
                    ;
                    break;

                default:
                case 1:
                case 12:
                    output.compute_root() //tx*ty*vs*c
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x/2)
                        .split(y, yo, yi, num_threads_y*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    // tx*(ty*vs+2*gw+1)*c = tx*(ty*vs+2*15+1)*c = tx*(ty*vs+31)*c ~= 2*tx*ty*vs*c = 2*tx*ty*4*2 = 16*tx*ty = 16K = 64KB
                    input_bound.compute_at(output_y, xo)
                        .store_in(MemoryType::GPUShared)
                        .split(_0, xo, xi, num_threads_x/2)
                        .split(_1, yo, yi, num_threads_y*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, _2, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    break;

                case 13:
                    output.compute_root() //tx*ty*vs*c
                        .bound(c, channel_min, channel_extent)
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y*vector_size)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    output_y.compute_root()
                        .split(x, xo, xi, num_threads_x)
                        .split(y, yo, yi, num_threads_y*vector_size/2)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    input_bound.compute_at(output_y, xo)
                        .store_in(MemoryType::GPUShared)
                        .split(_0, xo, xi, num_threads_x)
                        .split(_1, yo, yi, num_threads_y*vector_size/2)
                        .split(yi, yi, yi2, vector_size)
                        .reorder(yi2, _2, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(yi2)
                    ;
                    break;
                }
                sum_x.compute_at(output, xi)
                    .vectorize(x)
                    .vectorize(y)
                ;
                sum_x.update()
                    .reorder(x, y, c, kernel)
                    .vectorize(x)
                    .vectorize(y)
                    .unroll(c)
                ;
                sum_y.compute_at(output_y, xi)
                    .vectorize(x)
                    .vectorize(y)
                ;
                sum_y.update()
                    .reorder(x, y, c, kernel)
                    .vectorize(x)
                    .vectorize(y)
                    .unroll(c)
                ;
                if((mscheduler < 12) && ((mscheduler%2) == 0)) {
                    sum_x.store_in(MemoryType::GPUShared);
                    sum_y.store_in(MemoryType::GPUShared);
                } else {
                    sum_x.store_in(MemoryType::Register);
                    sum_y.store_in(MemoryType::Register);
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size_1d = 256;
                const int parallel_size_2d_y = 32;
                const int parallel_size_2d_x = 64*vector_size;
                Var xo{"xo"}, xi{"xi"}, xoo{"xoo"};
                Var yo{"yo"}, yi{"yi"}, yoo{"yoo"};
                RVar kernel_o{"kernel_o"}, kernel_i{"kernel_i"};

                weights_spatial.compute_root()
                    .split(i, xo, xi, parallel_size_1d)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                weights_range.compute_root()
                    .split(i, xo, xi, parallel_size_1d)
                    .parallel(xo)
                    .vectorize(xi, vector_size)
                ;
                if(out_define_schedule) {
                    output
                        .bound(c, channel_min, channel_extent)
                        .split(x, xoo, xo, parallel_size_2d_x)
                        .split(xo, xo, xi, vector_size)
                        .split(y, yo, yi, parallel_size_2d_y)
                        .fuse(xoo, yo, yoo)
                        .reorder(xi, c, xo, yi, yoo)
                        .vectorize(xi)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(yoo)
                        ;
                    }
                    sum_x_compute_level.set({output, xo});
                    sum_y_compute_level.set({output, yi});
                    intm_compute_level.set({output, yoo});
                }
                sum_x.compute_at(sum_x_compute_level)
                    .vectorize(x, vector_size)
                ;
                sum_x.update(0)
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, kernel, xo, y)
                    .unroll(c)
                    .vectorize(xi)
                ;
                sum_y.compute_at(sum_y_compute_level)
                    .vectorize(x, vector_size)
                ;
                sum_y.update(0)
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, kernel, y)
                    .unroll(c)
                    .vectorize(xi)
                ;
                guide_bound.compute_at(intm_compute_level);
                input_bound.compute_at(intm_compute_level);
            }
        }
    };
};

#endif
