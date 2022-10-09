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
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int num_pixels = 4*num_threads;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"};
                RVar rxo{"rxo"}, rxi{"rxi"}, rxi2{"rxi2"};
                RVar ryo{"ryo"}, ryi{"ryi"}, ryi2{"ryi2"};
                Func max_gray_intm, max_gray_intm2;
                int n;
                max_gray.compute_root().gpu_single_thread();
                switch (scheduler)
                {
                case 2:
                    max_gray.update()
                        .split(r_max_gray.x, rxo, rxi, num_threads)
                        .split(r_max_gray.y, ryo, ryi, num_threads)
                        .reorder(rxi, ryi, rxo, ryo)
                        .atomic()
                        .gpu_blocks(rxo, ryo)
                        .gpu_threads(rxi, ryi)
                    ;
                    break;

                case 3:
                    max_gray_intm = max_gray.update()
                        .split(r_max_gray.x, rxo, rxi, num_pixels)
                        .split(r_max_gray.y, ryo, ryi, num_pixels)
                        .rfactor({{rxo, x}, {ryo, y}})
                    ;
                    max_gray.update().gpu_single_thread();
                    max_gray_intm.in().compute_root()
                        .store_in(MemoryType::GPUTexture)
                        .gpu_blocks(x, y)
                    ;
                    max_gray_intm.compute_at(max_gray_intm.in(), x)
                        .store_in(MemoryType::GPUShared)
                    ;
                    max_gray_intm.update()
                        .split(rxi, rxi2, rxi, num_threads)
                        .split(ryi, ryi2, ryi, num_threads)
                        .reorder(rxi2, ryi2, rxi, ryi)
                        .atomic()
                        .gpu_threads(rxi, ryi)
                    ;
                    break;

                case 4:
                    max_gray_intm = max_gray.update()
                        .split(r_max_gray.x, rxo, rxi, num_pixels)
                        .split(r_max_gray.y, ryo, ryi, num_pixels)
                        .rfactor({{rxo, x}, {ryo, y}})
                    ;
                    max_gray.update()
                        .atomic()
                        .gpu_blocks(rxo, ryo)
                    ;
                    max_gray_intm.compute_at(max_gray, rxo)
                        .store_in(MemoryType::GPUShared)
                    ;
                    max_gray_intm.update()
                        .split(rxi, rxi2, rxi, num_threads)
                        .split(ryi, ryi2, ryi, num_threads)
                        .reorder(rxi2, ryi2, rxi, ryi)
                        .atomic()
                        .gpu_threads(rxi, ryi)
                    ;
                    break;

                case 5:
                    max_gray_intm = max_gray.update()
                        .split(r_max_gray.x, rxo, rxi, num_pixels)
                        .split(r_max_gray.y, ryo, ryi, num_pixels)
                        .rfactor({{rxo, x}, {ryo, y}})
                    ;
                    max_gray.update()
                        .atomic()
                        .gpu_blocks(rxo, ryo)
                    ;
                    max_gray_intm.compute_at(max_gray, rxo)
                        .store_in(MemoryType::GPUShared)
                    ;
                    n = 1;
                    while(n < num_threads) {
                        Var dx("dx"+std::to_string(n)), dy("dy"+std::to_string(n));
                        Func max_gray_intm_block = max_gray_intm.update()
                            .split(rxi, rxi, rxi2, 2)
                            .split(ryi, ryi, ryi2, 2)
                            .rfactor({{rxi2, dx}, {ryi2, dy}})
                        ;
                        max_gray_intm.update()
                            .unroll(rxi2)
                            .unroll(ryi2)
                        ;
                        for(int i = max_gray_intm_block.dimensions() - 4; i >= 2; i -= 2)
                        {
                            Var dx0 = max_gray_intm_block.args()[i];
                            Var dy0 = max_gray_intm_block.args()[i+1];
                            max_gray_intm_block.in().fuse(dx0, dx, dx);
                            max_gray_intm_block.in().fuse(dy0, dy, dy);
                        }
                        max_gray_intm_block.in().compute_at(max_gray, rxo)
                            .store_in(MemoryType::GPUShared)
                            .gpu_threads(dx, dy)
                        ;
                        max_gray_intm_block.compute_at(max_gray_intm_block.in(), dx)
                            .store_in(MemoryType::Register)
                        ;
                        max_gray_intm = max_gray_intm_block;
                        n *= 2;
                    }
                    max_gray_intm.update()
                        .unroll(rxi)
                        .unroll(ryi)
                    ;
                    break;

                default:
                case 6:
                case 7:
                case 8:
                    max_gray_intm = max_gray.update()
                        .split(r_max_gray.x, rxo, rxi, num_pixels)
                        .split(r_max_gray.y, ryo, ryi, num_pixels)
                        .rfactor({{rxo, x}, {ryo, y}})
                    ;
                    max_gray.update()
                        .atomic()
                        .gpu_blocks(rxo, ryo)
                    ;
                    max_gray_intm.compute_at(max_gray, rxo)
                        .store_in(MemoryType::GPUShared)
                    ;
                    n = 1;
                    while(n < num_threads) {
                        Var dx("dx"+std::to_string(n));
                        Func max_gray_intm_block = max_gray_intm.update()
                            .split(rxi, rxi, rxi2, 2)
                            .rfactor(rxi2, dx)
                        ;
                        max_gray_intm.update()
                            .unroll(rxi2)
                        ;
                        for(int i = max_gray_intm_block.dimensions() - 2; i >= 2; i--)
                        {
                            Var dx0 = max_gray_intm_block.args()[i];
                            max_gray_intm_block.in().fuse(dx0, dx, dx);
                        }
                        max_gray_intm_block.in().compute_at(max_gray, rxo)
                            .store_in(MemoryType::GPUShared)
                            .gpu_threads(dx)
                        ;
                        max_gray_intm_block.compute_at(max_gray_intm_block.in(), dx)
                            .store_in(MemoryType::Register)
                        ;
                        max_gray_intm = max_gray_intm_block;
                        n *= 2;
                    }
                    n = 1;
                    while(n < num_threads) {
                        Var dy("dy"+std::to_string(n));
                        Func max_gray_intm_block = max_gray_intm.update()
                            .split(ryi, ryi, ryi2, 2)
                            .rfactor(ryi2, dy)
                        ;
                        max_gray_intm.update()
                            .unroll(ryi2)
                        ;
                        for(int i = max_gray_intm_block.dimensions() - 2; i >= 2; i--)
                        {
                            Var dy0 = max_gray_intm_block.args()[i];
                            max_gray_intm_block.in().fuse(dy0, dy, dy);
                        }
                        max_gray_intm_block.in().compute_at(max_gray, rxo)
                            .store_in(MemoryType::GPUShared)
                            .gpu_threads(dy)
                        ;
                        max_gray_intm_block.compute_at(max_gray_intm_block.in(), dy)
                            .store_in(MemoryType::Register)
                        ;
                        max_gray_intm = max_gray_intm_block;
                        n *= 2;
                    }
                    max_gray_intm.update()
                        .unroll(rxi)
                        .unroll(ryi)
                    ;
                    break;
                }
                output.compute_root()
                    .bound(c, 0, 3)
                    .split(x, xo, xi, num_threads*vector_size)
                    .split(xi, xi, xi2, vector_size)
                    .split(y, yo, yi, num_threads)
                    .reorder(xi2, c, xi, yi, xo, yo)
                    .gpu_threads(xi, yi)
                    .gpu_blocks(xo, yo)
                    .vectorize(xi2)
                ;
                switch(scheduler) {
                    default:
                    case 2:
                    case 3:
                    case 4:
                    case 5:
                    case 6:
                        break;

                    case 7:
                        gain.compute_at(output, xi)
                            .store_in(MemoryType::GPUShared)
                            .vectorize(x)
                        ;
                        break;

                    case 8:
                        gain.compute_at(output, xi)
                            .store_in(MemoryType::Register)
                            .vectorize(x)
                        ;
                        break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size = 2;
                Var xo{"xo"}, xi{"xi"};
                RVar ryo{"ryo"}, ryi{"ryi"};
                Func max_gray_intm, max_gray_intm_in;
                if(out_define_schedule) {
                    output
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(y)
                        ;
                    }
                    intm_compute_level.set({output, y});
                }
                max_gray_intm = max_gray.update().split(r_max_gray.y, ryo, ryi, parallel_size).rfactor(ryo, y);
                max_gray_intm_in = max_gray_intm.in();
                max_gray_intm_in.compute_root()
                    .parallel(y)
                ;
                max_gray_intm.compute_at(max_gray_intm_in, y);
                max_gray.compute_root();
                gain.compute_at(intm_compute_level)
                    .vectorize(x, vector_size)
                ;
            }
        }
    };

};

#endif
