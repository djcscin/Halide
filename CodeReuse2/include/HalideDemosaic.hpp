#ifndef __HALIDE_DEMOSAIC__
#define __HALIDE_DEMOSAIC__

#include "Halide.h"
#include "CFA.hpp"
#include "Convert.hpp"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideDemosaic : public Generator<HalideDemosaic> {
    public:
        Input<Buffer<uint16_t>> img_input{"img_input", 2};
        Input<Func> lsc_map{"lsc_map", Float(32), 2};
        Input<Buffer<float>> white_balance{"white_balance", 1};
        Input<Buffer<uint16_t>> black_level{"black_level", 1};
        Input<uint16_t> white_level{"white_level"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};

        Output<Func> img_output{"img_dms_output", 3};

        GeneratorParam<Type> img_output_type{"img_output_type", UInt(8)};

        GeneratorParam<bool> composable{"composable", false};
        GeneratorParam<LoopLevel> intm_compute_level{"intm_compute_level", LoopLevel::inlined()};
        GeneratorParam<LoopLevel> intm_store_level{"intm_store_level", LoopLevel::inlined()};

        void generate() {
            input_bound = BoundaryConditions::mirror_interior(img_input);
            deinterld_bound(x, y, c) = select(
                cfa_pattern == RGGB, deinterleave_rggb(input_bound)(x, y, c),
                cfa_pattern == GRBG, deinterleave_grbg(input_bound)(x, y, c),
                cfa_pattern == BGGR, deinterleave_bggr(input_bound)(x, y, c),
                cfa_pattern == GBRG, deinterleave_gbrg(input_bound)(x, y, c),
                                    u16(0)
            );
            lsc_map_bound = BoundaryConditions::mirror_interior(lsc_map, {{0, img_input.width()}, {0, img_input.height()}});

            deinterld_lsc(x, y, c) = deinterld_bound(x, y, c) * lsc_map_bound(x, y);

            white_balancing(x, y, c) = deinterld_lsc(x, y, c) * white_balance(c);

            Expr unit = white_balancing(x, y, c) / f32(white_level);
            input_corrected(x, y, c) = u16(clamp_convert_from_unit(img_output_type, unit));

            interpolation_y(x, y, c) = input_corrected(x, y - 1, c) + 2*input_corrected(x, y, c) + input_corrected(x, y + 1, c);
            interpolation_x(x, y, c) = interpolation_y(x - 1, y, c) + 2*interpolation_y(x, y, c) + interpolation_y(x + 1, y, c);

            Expr rb = 4;
            Expr g = 8;
            interpolation(x, y, c) = interpolation_x(x, y, c) / select(c == 1, g, rb);

            img_output(x, y, c) = cast(img_output_type, interpolation(x, y, c));
        }

        void schedule() {
            img_input.dim(0).set_min(0);
            img_input.dim(1).set_min(0);
            white_balance.dim(0).set_bounds(0, 3);
            black_level.dim(0).set_bounds(0, 4);

            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}});
                lsc_map.set_estimates({{0, 4000}, {0, 3000}});
                white_balance.set_estimates({{0, 3}});
                black_level.set_estimates({{0, 4}});
                white_level.set_estimate(16*1023);
                cfa_pattern.set_estimate(RGGB);
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();
                if (!composable) {
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .align_bounds(y, 2, 0)
                        .align_bounds(x, 2, 0)
                        .unroll(c)
                        .split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    intm_compute_level.set({img_output, yo_i});
                    intm_store_level.set({img_output, yo_o});
                }
                interpolation_y
                    .compute_at(intm_compute_level)
                    .unroll(c)
                    .unroll(y)
                    .vectorize(x, vector_size, TailStrategy::RoundUp)
                ;
                interpolation_y.specialize(cfa_pattern == RGGB);
                interpolation_y.specialize(cfa_pattern == GRBG);
                interpolation_y.specialize(cfa_pattern == BGGR);
                interpolation_y.specialize(cfa_pattern == GBRG);
                input_bound.compute_at(intm_compute_level).store_at(intm_store_level);
                lsc_map_bound.compute_at(intm_compute_level).store_at(intm_store_level).compute_with(input_bound, _1);
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"};
        Var xo_o{"xo_o"}, xo_i{"xo_i"};
        Var yo_o{"yo_o"}, yo_i{"yo_i"};
        Func input_bound{"input_bound"}, deinterleaved{"deinterleaved"}, deinterld_bound{"deinterld_bound"};
        Func white_balancing{"white_balancing"}, deinterld_lsc{"deinterld_lsc"}, lsc_map_bound{"lsc_map_bound"};
        Func input_corrected{"input_corrected"};
        Func interpolation_y{"interpolation_y"}, interpolation_x{"interpolation_x"}, interpolation{"interpolation"};

    Expr black_level_subtraction(Expr img, Expr bl) {
        return u16_sat(i32(img) - bl);
    }

    Func deinterleave_rggb(Func input) {
        Func output{"deinterleave_rggb"};

        Expr r = (((x % 2) == 0) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(0));
        Expr g = (((x % 2) == 1) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(1)) +
                 (((x % 2) == 0) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(2));
        Expr b = (((x % 2) == 1) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(3));

        output(x, y, c) = mux(c, {r, g, b});

        return output;
    }

    Func deinterleave_grbg(Func input) {
        Func output{"deinterleave_grbg"};

        Expr r = (((x % 2) == 1) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(1));
        Expr g = (((x % 2) == 0) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(0)) +
                 (((x % 2) == 1) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(3));
        Expr b = (((x % 2) == 0) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(2));

        output(x, y, c) = mux(c, {r, g, b});

        return output;
    }

    Func deinterleave_bggr(Func input) {
        Func output{"deinterleave_bggr"};

        Expr r = (((x % 2) == 1) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(3));
        Expr g = (((x % 2) == 1) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(1)) +
                 (((x % 2) == 0) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(2));
        Expr b = (((x % 2) == 0) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(0));

        output(x, y, c) = mux(c, {r, g, b});

        return output;
    }

    Func deinterleave_gbrg(Func input) {
        Func output{"deinterleave_gbrg"};

        Expr r = (((x % 2) == 0) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(2));
        Expr g = (((x % 2) == 0) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(0)) +
                 (((x % 2) == 1) && ((y % 2) == 1))*black_level_subtraction(input(x, y), black_level(3));
        Expr b = (((x % 2) == 1) && ((y % 2) == 0))*black_level_subtraction(input(x, y), black_level(1));

        output(x, y, c) = mux(c, {r, g, b});

        return output;
    }

};

#endif
