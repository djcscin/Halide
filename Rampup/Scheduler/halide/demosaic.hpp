#ifndef __DEMOSAIC__
#define __DEMOSAIC__

#include "halide_base.hpp"
#include "CFA.hpp"

namespace {
    using namespace Halide;

    class Demosaic : public Generator<Demosaic>, public HalideBase {
    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func deinterld{"deinterld"}, deinterld_bound{"deinterld_bound"}, input_bound{"input_bound"};
        Func interpolation_y{"interpolation_y"}, interpolation_x{"interpolation_x"};
        RDom r_max_gray;
    public:
        Input<Func> input{"input", Float(32), 2};
        Input<int> width{"width"};
        Input<int> height{"height"};
        Input<uint8_t> cfa_pattern{"cfa_pattern"};
        Output<Func> output{"output_demosaic", Float(32), 3};

        void generate() {
            if(scheduler < 10) {
                deinterld(x, y, c) = select(
                    cfa_pattern == RGGB, deinterleave_rggb(input)(x, y, c),
                    cfa_pattern == GRBG, deinterleave_grbg(input)(x, y, c),
                    cfa_pattern == BGGR, deinterleave_bggr(input)(x, y, c),
                    cfa_pattern == GBRG, deinterleave_gbrg(input)(x, y, c),
                                        0.0f
                );
                deinterld_bound = BoundaryConditions::mirror_interior(deinterld, {{0, width}, {0, height}});
            } else {
                input_bound = BoundaryConditions::mirror_interior(input, {{0, width}, {0, height}});
                deinterld_bound(x, y, c) = select(
                    cfa_pattern == RGGB, deinterleave_rggb(input_bound)(x, y, c),
                    cfa_pattern == GRBG, deinterleave_grbg(input_bound)(x, y, c),
                    cfa_pattern == BGGR, deinterleave_bggr(input_bound)(x, y, c),
                    cfa_pattern == GBRG, deinterleave_gbrg(input_bound)(x, y, c),
                                        0.0f
                );
            }

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
                const int parallel_size =  (
                    (uint32_t(scheduler-7)<4)?4: //scheduler=7-10
                                              8  //scheduler=11-18,default
                );
                Var xi{"xi"}, xo{"xo"}, xio{"xio"}, xii{"xii"};
                Var yi{"yi"}, yo{"yo"}, yio{"yio"}, yii{"yii"};

                switch (scheduler)
                {
                case 2:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .parallel(y)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    break;

                case 3:
                case 4:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .align_bounds(y, 2, 0)
                        .split(y, yo, yi, 2)
                        .unroll(yi)
                        .parallel(yo)
                        .align_bounds(x, 2, 0)
                        .split(x, xo, xi, vector_size)
                        .split(xi, xio, xii, 2)
                        .unroll(xii)
                        .vectorize(xio)
                        .reorder(xii, xio, c, xo, yi, yo)
                    ;
                    if(scheduler == 4) {
                        output.specialize(cfa_pattern == RGGB);
                        output.specialize(cfa_pattern == GRBG);
                        output.specialize(cfa_pattern == GBRG);
                        output.specialize(cfa_pattern == BGGR);
                    }
                    break;

                case 5:
                case 6:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .align_bounds(y, 2, 0)
                        .split(y, yo, yi, 2)
                        .parallel(yo)
                        .align_bounds(x, 2, 0)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    interpolation_y.compute_at(output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size)
                        .split(xi, xio, xii, 2)
                        .unroll(xii)
                        .vectorize(xio)
                        .reorder(xii, xio, c, xo, y)
                    ;
                    if(scheduler == 6) {
                        interpolation_y.specialize(cfa_pattern == RGGB);
                        interpolation_y.specialize(cfa_pattern == GRBG);
                        interpolation_y.specialize(cfa_pattern == GBRG);
                        interpolation_y.specialize(cfa_pattern == BGGR);
                    }
                    break;

                case 7:
                case 8:
                case 9:
                case 10:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .align_bounds(y, 2, 0)
                        .split(y, yo, yi, parallel_size)
                        .parallel(yo)
                        .align_bounds(x, 2, 0)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    interpolation_y.compute_at(output, (scheduler<9)?yo:yi) // yo-7,8;yi-9,10
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    deinterld_bound.compute_at(output, yo)
                        .unroll(c)
                        .split(y, yo, yi, 2)
                        .unroll(yi)
                        .split(x, xo, xi, vector_size)
                        .split(xi, xio, xii, 2)
                        .unroll(xii)
                        .vectorize(xio)
                        .reorder(xii, xio, c, xo, yi, yo)
                    ;
                    if((scheduler % 2) == 0) { // scheduler == 8,10
                        deinterld_bound.specialize(cfa_pattern == RGGB);
                        deinterld_bound.specialize(cfa_pattern == GRBG);
                        deinterld_bound.specialize(cfa_pattern == GBRG);
                        deinterld_bound.specialize(cfa_pattern == BGGR);
                    }
                    break;

                case 13:
                case 14:
                case 17:
                case 18:
                    input_bound.compute_at(output, yio).store_at(output, yo);

                case 11:
                case 12:
                case 15:
                case 16:
                    output.compute_root()
                        .bound(c, 0, 3)
                        .unroll(c)
                        .align_bounds(y, 2, 0)
                        .split(y, yo, yi, parallel_size)
                        .split(yi, yio, yii, 2)
                        .parallel(yo)
                        .align_bounds(x, 2, 0)
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, yii, yio, yo)
                    ;
                    interpolation_y.compute_at(output, (scheduler<15)?yio:yii) //yio-11,12,13,14;yii-15,16,17,18
                        .split(x, xo, xi, vector_size)
                        .vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    deinterld_bound.compute_at(output, yio).store_at(output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size)
                        .split(xi, xio, xii, 2)
                        .unroll(xii)
                        .vectorize(xio)
                        .reorder(xii, xio, c, xo, y)
                    ;
                    if((scheduler % 2) == 0) { // scheduler == 12,14,16,18
                        deinterld_bound.specialize(cfa_pattern == RGGB);
                        deinterld_bound.specialize(cfa_pattern == GRBG);
                        deinterld_bound.specialize(cfa_pattern == GBRG);
                        deinterld_bound.specialize(cfa_pattern == BGGR);
                    }
                    break;

                default:
                case 1:
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
                    break;
                }
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
