#ifndef __DEMOSAIC__
#define __DEMOSAIC__

#include "halide_base.hpp"
#include "CFA.hpp"

namespace {
    using namespace Halide;

    class Demosaic : public Generator<Demosaic>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func deinterld{"deinterld"}, deinterld_bound{"deinterld_bound"};
        Func interpolation_y{"interpolation_y"}, interpolation_x{"interpolation_x"};
        RDom r_max_gray;
    public:
        Input<Func> input{"input", Float(32), 2};
        Input<int> width{"width"};
        Input<int> height{"height"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};
        Output<Func> output{"output_demosaic", Float(32), 3};

        void generate() {
            deinterld(x, y, c) = select(
                cfa_pattern == RGGB, deinterleave_rggb(input)(x, y, c),
                cfa_pattern == GRBG, deinterleave_grbg(input)(x, y, c),
                cfa_pattern == BGGR, deinterleave_bggr(input)(x, y, c),
                cfa_pattern == GBRG, deinterleave_gbrg(input)(x, y, c),
                                     0.0f
            );

            deinterld_bound = BoundaryConditions::mirror_interior(deinterld, {{0, width}, {0, height}});
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
            } else {
                const int vector_size = get_target().natural_vector_size<float>();
                Var xi{"xi"}, xo{"xo"}, xo_o{"xo_o"}, xo_i{"xo_i"};
                Var yi{"yi"}, yo{"yo"};

                output.compute_root()
                    .bound(c, 0, 3)
                    .align_bounds(x, 2, 0)
                    .align_bounds(y, 2, 0)
                    .split(y, yo, yi, 2)
                    .split(x, xo, xi, vector_size)
                    .reorder(xi, c, xo, yi, yo)
                    .unroll(c)
                    .vectorize(xi)
                    .parallel(yo)
                ;
                interpolation_y.compute_at(output, yo)
                    .split(x, xo, xi, 2)
                    .split(y, yo, yi, 2)
                    .split(xo, xo_o, xo_i, vector_size)
                    .reorder(xi, xo_i, c, xo_o, yi, yo)
                    .unroll(c)
                    .unroll(xi)
                    .unroll(yi)
                    .vectorize(xo_i)
                ;
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
