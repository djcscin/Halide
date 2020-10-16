/* run all versions using
for lut in false true; do make LUT=$lut; done
*/

#include "Halide.h"
#include "CFA.hpp"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideDemosaic : public Generator<HalideDemosaic> {
    public:
        Input<Buffer<uint16_t>> img_input{"img_input", 2};
        Input<Buffer<float>> white_balance{"white_balance", 1};
        Input<Buffer<uint16_t>> black_level{"black_level", 1};
        Input<uint16_t> white_level{"white_level"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<bool> lut{"lut", true}; // LUT-> LookUp Table

        void generate() {
            input_bound = BoundaryConditions::mirror_interior(img_input);
            deinterld_bound(x, y, c) = select(
                cfa_pattern == RGGB, deinterleave_rggb(input_bound)(x, y, c),
                cfa_pattern == GRBG, deinterleave_grbg(input_bound)(x, y, c),
                cfa_pattern == BGGR, deinterleave_bggr(input_bound)(x, y, c),
                cfa_pattern == GBRG, deinterleave_gbrg(input_bound)(x, y, c),
                                    u16(0)
            );

            interpolation_y(x, y, c) = i32(deinterld_bound(x, y - 1, c)) + 2*deinterld_bound(x, y, c) + deinterld_bound(x, y + 1, c);
            interpolation_x(x, y, c) = interpolation_y(x - 1, y, c) + 2*interpolation_y(x, y, c) + interpolation_y(x + 1, y, c);

            Expr rb = 4;
            Expr g = 8;

            if(lut) {
                lut_interpolation(value, c) = value / select(c == 1, g, rb);

                lut_white_balancing(value, c) = lut_interpolation(value, c) * white_balance(c);

                Expr unit = lut_white_balancing(value, c) / f32(white_level);
                lut_output(value, c) = u8_sat(unit * 255.0f);

                img_output(x, y, c) = lut_output(interpolation_x(x, y, c), c);
            } else {
                interpolation(x, y, c) = interpolation_x(x, y, c) / select(c == 1, g, rb);

                white_balancing(x, y, c) = interpolation(x, y, c) * white_balance(c);

                Expr unit = white_balancing(x, y, c) / f32(white_level);
                img_output(x, y, c) = u8_sat(unit * 255.0f);
            }
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}});
                white_balance.set_estimates({{0, 3}});
                black_level.set_estimates({{0, 4}});
                white_level.set_estimate(16*1023);
                cfa_pattern.set_estimate(RGGB);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<int32_t>();

                if(lut) {
                    lut_output
                        .compute_root()
                        .unroll(c)
                        .split(value, xo, xi, vector_size).vectorize(xi)
                        .split(xo, xo_o, xo_i, 256).parallel(xo_o)
                        .reorder(xi, xo_i, c, xo_o)
                    ;
                }
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
                interpolation_y
                    .compute_at(img_output, yo_i)
                    .unroll(c)
                    .unroll(y)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                interpolation_y.specialize(cfa_pattern == RGGB);
                interpolation_y.specialize(cfa_pattern == GRBG);
                interpolation_y.specialize(cfa_pattern == BGGR);
                interpolation_y.specialize(cfa_pattern == GBRG);
                input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);

            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"};
        Var xo_o{"xo_o"}, xo_i{"xo_i"};
        Var yo_o{"yo_o"}, yo_i{"yo_i"};
        Func input_bound{"input_bound"}, deinterleaved{"deinterleaved"}, deinterld_bound{"deinterld_bound"};
        Func interpolation_y{"interpolation_y"}, interpolation_x{"interpolation_x"}, interpolation{"interpolation"};
        Func white_balancing{"white_balancing"};
        Var value{"value"};
        Func lut_interpolation{"lut_interpolation"}, lut_white_balancing{"lut_white_balancing"}, lut_output{"lut_output"};

    Expr black_level_subtraction(Expr img, Expr bl) {
        return u16_sat(i32(img) - bl);
    }

    Func deinterleave_rggb(Func input) {
        Func output{"deinterleave_rggb"};

// INPUT
// RGGB
// RGRGRGRGRGRGRG
// GBGBGBGBGBGBGB
// RGRGRGRGRGRGRG
// GBGBGBGBGBGBGB

// OUTPUT
// Canal 0
// R0R0R0R0R0R0R0
// 00000000000000
// R0R0R0R0R0R0R0
// 00000000000000
// Canal 1
// 0G0G0G0G0G0G0G
// G0G0G0G0G0G0G0
// 0G0G0G0G0G0G0G
// G0G0G0G0G0G0G0
// Canal 2
// 00000000000000
// 0B0B0B0B0B0B0B
// 00000000000000
// 0B0B0B0B0B0B0B

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

    // Exercício:
    // Fazer para GBRG
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
HALIDE_REGISTER_GENERATOR(HalideDemosaic, demosaic);
