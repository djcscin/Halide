#ifndef __DEMOSAIC__
#define __DEMOSAIC__

#include "halide_base.hpp"
#include "CFA.hpp"

namespace {
    using namespace Halide;

    class Demosaic : public Generator<Demosaic>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func deinterld_bound{"deinterld_bound"}, input_bound{"input_bound"};
        Func interpolation_y{"interpolation_y"}, interpolation_x{"interpolation_x"};
        RDom r_max_gray;
    public:
        Input<Func> input{"input", Float(32), 2};
        Input<int> width{"width"};
        Input<int> height{"height"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};
        Output<Func> output{"output_demosaic", Float(32), 3};

        void generate() {
            input_bound = BoundaryConditions::mirror_interior(input, {{0, width}, {0, height}});
            deinterld_bound(x, y, c) = select(
                cfa_pattern == RGGB, deinterleave_rggb(input_bound)(x, y, c),
                cfa_pattern == GRBG, deinterleave_grbg(input_bound)(x, y, c),
                cfa_pattern == BGGR, deinterleave_bggr(input_bound)(x, y, c),
                cfa_pattern == GBRG, deinterleave_gbrg(input_bound)(x, y, c),
                                    0.0f
            );

            interpolation_y(x, y, c) = 0.5f*deinterld_bound(x, y - 1, c) + deinterld_bound(x, y, c) + 0.5f*deinterld_bound(x, y + 1, c);
            interpolation_x(x, y, c) = 0.5f*interpolation_y(x - 1, y, c) + interpolation_y(x, y, c) + 0.5f*interpolation_y(x + 1, y, c);

            Expr interpolation = interpolation_x(x, y, c);
            Expr out = select(c == 1, 0.5f*interpolation, interpolation);
            output(x, y, c) = clamp(out, 0.f, 1.f);
        }

        void schedule() {
            if(auto_schedule) {
                input.set_estimates({{0,4000},{0,3000}});
                width.set_estimate(4000);
                height.set_estimate(3000);
                output.set_estimates({{0,4000},{0,3000},{0,3}});
            } else if(get_target().has_gpu_feature()) {
                const int num_threads = 32 * size_factor;
                const int vector_size = 4;
                Var xo{"xo"}, xi{"xi"}, xi2{"xi2"}, xi3{"xi3"};
                Var yo{"yo"}, yi{"yi"}, yi2{"yi2"}, yi3{"yi3"};
                switch (scheduler)
                {
                default:
                case 1:
                case 2:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads)
                        .reorder(xi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(c)
                    ;
                    break;

                case 5: // compute interpolation_y at GPU blocks
                case 6:
                    interpolation_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads*2)
                        .split(yi, yi, yi2, 2)
                        .reorder(xi2, yi2, c, xi, yi, xo, yo)
                        .unroll(xo)
                        .unroll(yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(c)
                    ;
                    if(scheduler == 6) {
                        interpolation_y.specialize(cfa_pattern == RGGB);
                        interpolation_y.specialize(cfa_pattern == GRBG);
                        interpolation_y.specialize(cfa_pattern == GBRG);
                        interpolation_y.specialize(cfa_pattern == BGGR);
                    }
                    goto scheduler_3_4;

                case 7: // compute interpolation_y at GPU threads
                case 8:
                    interpolation_y.compute_at(output, xi).store_in(MemoryType::Register)
                        .split(x, xi, xi2, vector_size)
                        .split(y, yi, yi2, 2)
                        .reorder(xi2, yi2, c, xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(c)
                    ;
                    if(scheduler == 8) {
                        interpolation_y.specialize(cfa_pattern == RGGB);
                        interpolation_y.specialize(cfa_pattern == GRBG);
                        interpolation_y.specialize(cfa_pattern == GBRG);
                        interpolation_y.specialize(cfa_pattern == BGGR);
                    }

                case 3:
                case 4:
                scheduler_3_4:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads*2)
                        .split(yi, yi, yi2, 2)
                        .reorder(xi2, yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(c)
                    ;
                    if(scheduler == 4) {
                        output.specialize(cfa_pattern == RGGB);
                        output.specialize(cfa_pattern == GRBG);
                        output.specialize(cfa_pattern == GBRG);
                        output.specialize(cfa_pattern == BGGR);
                    }
                    break;

                case 9:
                case 10:
                    output.compute_root()
                        .align_bounds(x, 2, 0)
                        .align_bounds(y, 2, 0)
                        .bound(c, 0, 3)
                        .split(x, xo, xi, num_threads*vector_size - 2) // reduce 2 from dimension x
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads/2)
                        .split(yi, yi, yi2, 2)
                        .reorder(xi2, yi2, c, xi, yi, xo, yo)
                        .gpu_blocks(xo, yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(c)
                    ;
                    interpolation_y.compute_at(output, xo).store_in(MemoryType::GPUShared)
                        .split(x, xo, xi, num_threads*vector_size)
                        .split(xi, xi, xi2, vector_size)
                        .split(y, yo, yi, num_threads/2)
                        .split(yi, yi, yi2, 2)
                        .reorder(xi2, yi2, c, xi, yi, xo, yo)
                        .unroll(xo)
                        .unroll(yo)
                        .gpu_threads(xi, yi)
                        .vectorize(xi2)
                        .unroll(yi2)
                        .unroll(c)
                    ;
                    if(scheduler == 10) {
                        interpolation_y.specialize(cfa_pattern == RGGB);
                        interpolation_y.specialize(cfa_pattern == GRBG);
                        interpolation_y.specialize(cfa_pattern == GBRG);
                        interpolation_y.specialize(cfa_pattern == BGGR);
                    }
                    break;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                const int parallel_size = 8;
                Var xi{"xi"}, xo{"xo"}, xio{"xio"}, xii{"xii"};
                Var yi{"yi"}, yo{"yo"};

                if(out_define_schedule) {
                    output
                        .bound(c, 0, 3)
                        .unroll(c)
                        .align_bounds(y, 2, 0)
                        .split(y, yo, yi, 2)
                        .align_bounds(x, 2, 0)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    if(out_define_compute) {
                        output.compute_root()
                            .parallel(yo)
                        ;
                    }
                    intm_compute_level.set({output, yo});
                    intm_store_level.set({output, yo});
                }
                interpolation_y.compute_at(intm_compute_level)
                    .unroll(c)
                    .unroll(y)
                    .split(x, xo, xi, vector_size)
                    .split(xi, xio, xii, 2)
                    .unroll(xii)
                    .vectorize(xio)
                    .reorder(xii, xio, c, xo, y)
                ;
                interpolation_y.specialize(cfa_pattern == RGGB);
                interpolation_y.specialize(cfa_pattern == GRBG);
                interpolation_y.specialize(cfa_pattern == GBRG);
                interpolation_y.specialize(cfa_pattern == BGGR);
            }
        }

    private:
        Func deinterleave_rggb(Func input) {
            Func output{"deinterleave_rggb"};

            Expr r = (((x % 2) == 0) && ((y % 2) == 0))*input(x, y);
            Expr g = (((x % 2) == 1) && ((y % 2) == 0))*input(x, y) +
                     (((x % 2) == 0) && ((y % 2) == 1))*input(x, y);
            Expr b = (((x % 2) == 1) && ((y % 2) == 1))*input(x, y);

            output(x, y, c) = mux(c, {r, g, b});
            return output;
        }
        Func deinterleave_grbg(Func input) {
            Func output{"deinterleave_grbg"};

            Expr r = (((x % 2) == 1) && ((y % 2) == 0))*input(x, y);
            Expr g = (((x % 2) == 0) && ((y % 2) == 0))*input(x, y) +
                     (((x % 2) == 1) && ((y % 2) == 1))*input(x, y);
            Expr b = (((x % 2) == 0) && ((y % 2) == 1))*input(x, y);

            output(x, y, c) = mux(c, {r, g, b});
            return output;
        }
        Func deinterleave_bggr(Func input) {
            Func output{"deinterleave_bggr"};

            Expr r = (((x % 2) == 1) && ((y % 2) == 1))*input(x, y);
            Expr g = (((x % 2) == 1) && ((y % 2) == 0))*input(x, y) +
                     (((x % 2) == 0) && ((y % 2) == 1))*input(x, y);
            Expr b = (((x % 2) == 0) && ((y % 2) == 0))*input(x, y);

            output(x, y, c) = mux(c, {r, g, b});
            return output;
        }
        Func deinterleave_gbrg(Func input) {
            Func output{"deinterleave_gbrg"};

            Expr r = (((x % 2) == 0) && ((y % 2) == 1))*input(x, y);
            Expr g = (((x % 2) == 0) && ((y % 2) == 0))*input(x, y) +
                     (((x % 2) == 1) && ((y % 2) == 1))*input(x, y);
            Expr b = (((x % 2) == 1) && ((y % 2) == 0))*input(x, y);

            output(x, y, c) = mux(c, {r, g, b});
            return output;
        }
    };

};

#endif
