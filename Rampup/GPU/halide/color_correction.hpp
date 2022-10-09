#ifndef __COLOR_CORRECTION__
#define __COLOR_CORRECTION__

#include "halide_base.hpp"

namespace {
    using namespace Halide;

    class ColorCorrection : public Generator<ColorCorrection>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
    public:
        Input<Func> input{"input", Float(32), 3};
        Input<Buffer<float>> ccm{"ccm", 2};
        Output<Func> output{"output_cc", Float(32), 3};

        void generate() {
            Expr cc = input(x, y, 0) * ccm(0, c) + input(x, y, 1) * ccm(1, c) + input(x, y, 2) * ccm(2, c);
            output(x, y, c) = clamp(cc, 0.f, 1.f);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000},{0,3}});
                ccm.set_estimates({{0,3},{0,3}});
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"};
                Var yo{"yo"}, yi{"yi"};
                Var co{"co"}, ci{"ci"};
                if(get_target().has_feature(Target::OpenCL)) {
                    ccm.store_in(MemoryType::GPUTexture);
                }
                switch (scheduler)
                {
                case 2:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi2, xi, yi, xo, yo, c)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .vectorize(xi2)
                    ;
                    break;

                default:
                case 1:
                case 3:
                case 4:
                case 5:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads/((scheduler == 4)?2:1))
                        .reorder(xi2, c, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .vectorize(xi2)
                    ;
                    if(scheduler > 3) { // 4, 5
                        input.in().compute_at(output, xi)
                            .split(_0, xi, xi2, vector_size).vectorize(xi2)
                        ;
                        if(scheduler == 4)
                            input.in().store_in(MemoryType::GPUShared);
                        else
                            input.in().store_in(MemoryType::Register);
                    }
                    break;

                case 6:
                case 7:
                case 8:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads)
                        .split(y, yo, yi, num_threads)
                        .reorder(c, xi, yi, xo, yo)
                        .gpu_threads(xi, yi)
                        .gpu_blocks(xo, yo)
                        .split(c, co, ci, 3).vectorize(ci)
                    ;
                    if(scheduler > 6) { // 7, 8
                        input.in().compute_at(output, xi)
                            .split(_2, co, ci, 3).vectorize(ci)
                        ;
                        if(scheduler == 4)
                            input.in().store_in(MemoryType::GPUShared);
                        else
                            input.in().store_in(MemoryType::Register);
                    }
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xo{"xo"}, xi{"xi"};

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
                }
            }
        }
    };

};

#endif
