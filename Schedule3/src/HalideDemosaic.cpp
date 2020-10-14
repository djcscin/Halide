/* run all schedulers using
for i in $(seq 0 47); do make SCHEDULER=$i; done
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

        GeneratorParam<uint> scheduler{"scheduler", 0};

        void generate() {
            if(scheduler < 30) {
                deinterleaved(x, y, c) = select(
                    cfa_pattern == RGGB, deinterleave_rggb(img_input)(x, y, c),
                    cfa_pattern == GRBG, deinterleave_grbg(img_input)(x, y, c),
                    cfa_pattern == BGGR, deinterleave_bggr(img_input)(x, y, c),
                    cfa_pattern == GBRG, deinterleave_gbrg(img_input)(x, y, c),
                                        u16(0)
                );
                deinterld_bound = BoundaryConditions::mirror_interior(deinterleaved, {{0, img_input.width()}, {0, img_input.height()}});
            } else {
                input_bound = BoundaryConditions::mirror_interior(img_input);
                deinterld_bound(x, y, c) = select(
                    cfa_pattern == RGGB, deinterleave_rggb(input_bound)(x, y, c),
                    cfa_pattern == GRBG, deinterleave_grbg(input_bound)(x, y, c),
                    cfa_pattern == BGGR, deinterleave_bggr(input_bound)(x, y, c),
                    cfa_pattern == GBRG, deinterleave_gbrg(input_bound)(x, y, c),
                                        u16(0)
                );
            }

            // 1                  1 2 1
            // 2 * 1 2 1 * 1/16 = 2 4 2 * 1/16
            // 1                  1 2 1
            interpolation_y(x, y, c) = i32(deinterld_bound(x, y - 1, c)) + 2*deinterld_bound(x, y, c) + deinterld_bound(x, y + 1, c);
            interpolation_x(x, y, c) = interpolation_y(x - 1, y, c) + 2*interpolation_y(x, y, c) + interpolation_y(x + 1, y, c);

            Expr rb = 4; // 4/16 = 1/4
            Expr g = 8;  // 2/16 = 1/8
            interpolation(x, y, c) = interpolation_x(x, y, c) / select(c == 1, g, rb);

            white_balancing(x, y, c) = interpolation(x, y, c) * white_balance(c);

            Expr unit = white_balancing(x, y, c) / f32(white_level);
            img_output(x, y, c) = u8_sat(unit * 255.0f);
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

                switch (scheduler)
                {
                case 0:
                    img_output.compute_root();
                    white_balancing.compute_root();
                    interpolation.compute_root();
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved.compute_root();
                    break;

                case 1:
                    // BEFORE
                    // for c:
                    //  for y:
                    //   for x:
                    //     interpolation(x, y, c)
                    // AFTER
                    // for y:
                    //  for x:
                    //   interpolation(x, y, 0)
                    // for y:
                    //  for x:
                    //   interpolation(x, y, 1)
                    // for y:
                    //  for x:
                    //   interpolation(x, y, 2)
                    img_output.compute_root();
                    white_balancing.compute_root();
                    interpolation
                        .compute_root()
                        .bound(c, 0, 3).unroll(c) // c tem valores de 0 a 2 e vou fazer unroll no c
                    ;
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved.compute_root();
                    break;

                case 2:
                    img_output.compute_root();
                    white_balancing.compute_root();
                    interpolation
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2, 0).split(x, xo, xi, 2).unroll(xi) // x é uma variável de tamanho par e mínimo é 0
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2).unroll(yi)
                    ;
                    break;

                case 3:
                    // BEFORE
                    // for y:
                    //  for x:
                    //    deinterleaved(x, y, [0,1,2]) com select do cfa_pattern
                    // AFTER
                    //  if cfa_pattern == RGGB
                    //   for y:
                    //    for x:
                    //     deinterleaved(x, y, [0,1,2]) do RGGB sem select
                    //  else if cfa_pattern == GRBG
                    //   for y:
                    //    for x:
                    //     deinterleaved(x, y, [0,1,2]) do GRBG sem select
                    // ...
                    img_output.compute_root();
                    white_balancing.compute_root();
                    interpolation
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 4:
                    // compute inline white_balancing
                    img_output.compute_root();
                    white_balancing.compute_inline();
                    interpolation
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 5:
                    // compute inline interpolation -> .bound(c, 0, 3).unroll(c) moved to img_output
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_root();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 6:
                    // compute inline interpolation_x
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_root();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 7:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // for c:
                    //  for y:
                    //   for x:
                    //     deinterld_bound(x, y, c)
                    // unroll c:
                    //  for y:
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_inline();
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 8:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // for c:
                    //  for y:
                    //   for x:
                    //     deinterld_bound(x, y, c)
                    // unroll c:
                    //  for y:
                    //   allocate interpolation_y
                    //   for c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_at(img_output, y);
                    deinterld_bound.compute_root();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 9:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // unroll c:
                    //  for y:
                    //   allocate interpolation_y
                    //   for c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_at(img_output, y);
                    deinterld_bound.compute_inline();
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 10:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // unroll c:
                    //  allocate deinterld_bounds
                    //  for y:
                    //   for c:
                    //    for y:
                    //     for x:
                    //      deinterld_bound(x, y, c)
                    //   allocate interpolation_y
                    //   for c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_at(img_output, y);
                    deinterld_bound.compute_at(img_output, y).store_at(img_output, c);
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 11:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // unroll c:
                    //  allocate deinterld_bounds
                    //  for y:
                    //   for c:
                    //    for y:
                    //     for x:
                    //      deinterld_bound(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_inline();
                    deinterld_bound.compute_at(img_output, y).store_at(img_output, c);
                    deinterleaved
                        .compute_root()
                        .unroll(c)
                        .align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi)
                        .align_bounds(y, 2).split(y, yo, yi, 2).unroll(yi)
                    ;
                    deinterleaved.specialize(cfa_pattern == RGGB);
                    deinterleaved.specialize(cfa_pattern == GRBG);
                    deinterleaved.specialize(cfa_pattern == BGGR);
                    deinterleaved.specialize(cfa_pattern == GBRG);
                    break;

                case 12:
                    // BEFORE:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterleaved(x, y, c)
                    // unroll c:
                    //  for y:
                    //   allocate interpolation_y
                    //   for c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    // AFTER:
                    // unroll c:
                    //  for yo:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    unroll y:
                    //     for xo:
                    //      unroll xi:
                    //       interpolation_y(x, y, c)
                    //   for yi: //2
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .align_bounds(x, 2, 0)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 13:
                    // unroll c:
                    //  for yo:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    unroll y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 14:
                    // unroll c:
                    //  for y:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, y)
                        .unroll(c)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 15:
                    // unroll c:
                    //  for yo:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    unroll y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 16:
                    // unroll c:
                    //  parallel yo:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 17:
                    // parallel yo:
                    //  unroll c:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .reorder(x, yi, c, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, c)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 18:
                    // parallel yo:
                    //  allocate interpolation_y
                    //  unroll c:
                    //   for y:
                    //    for x:
                    //     interpolation_y(x, y, c)
                    //  unroll c:
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .reorder(x, yi, c, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 19:
                    // parallel yo:
                    //  allocate interpolation_y
                    //  unroll c:
                    //   for y:
                    //    for x:
                    //     interpolation_y(x, y, c)
                    //  for yi:
                    //   unroll c:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .reorder(x, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 20:
                    // parallel yo:
                    //  allocate interpolation_y
                    //  for y:
                    //   unroll c:
                    //    for x:
                    //     interpolation_y(x, y, c)
                    //  for yi:
                    //   unroll c:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .reorder(x, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .reorder(x, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 21:
                    // parallel yo:
                    //  allocate interpolation_y
                    //  for y:
                    //   for x:
                    //    unroll c:
                    //     interpolation_y(x, y, c)
                    //  for yi:
                    //   for x:
                    //    unroll c:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .reorder(c, x, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .reorder(c, x, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 22:
                    // parallel yo:
                    //  allocate interpolation_y
                    //  for y:
                    //   unroll c:
                    //    for xo:
                    //     vectorize xi:
                    //      interpolation_y(x, y, c)
                    //  for yi:
                    //   unroll c:
                    //    for xo:
                    //     vectorize xi:
                    //      img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 23:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 24:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 25:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size*2).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size*2).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 26:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size*4).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size*4).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 27:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size, TailStrategy::RoundUp).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 28:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 29:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .parallel(yo)
                        .split(x, xo, xi, vector_size, TailStrategy::ShiftInwards).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .split(x, xo, xi, vector_size, TailStrategy::ShiftInwards).vectorize(xi)
                        .reorder(xi, xo, c, y)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    deinterleaved.compute_inline();
                    break;

                case 30:
                    // unroll c:
                    //  for yo:
                    //   unroll yi:
                    //    for xo:
                    //     unroll xi:
                    //      deinterld_bound(x, y, c)
                    // unroll c:
                    //  for y:
                    //   allocate interpolation_y
                    //   for c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for x:
                    //    img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_at(img_output, y);
                    deinterld_bound
                        .compute_root()
                        .unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2).unroll(yi)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    deinterld_bound.specialize(cfa_pattern == RGGB);
                    deinterld_bound.specialize(cfa_pattern == GRBG);
                    deinterld_bound.specialize(cfa_pattern == BGGR);
                    deinterld_bound.specialize(cfa_pattern == GBRG);
                    input_bound.compute_inline();
                    break;

                case 31:
                    // unroll c:
                    //  for yo:
                    //   allocate deinterld_bounds
                    //   for c:
                    //    for y:
                    //     for x:
                    //      deinterld_bound(x, y, c)
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //       interpolation_y(x, y, c)
                    //   for yi: //2
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y.compute_at(img_output, yo);
                    deinterld_bound
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .split(y, yo, yi, 2).unroll(yi)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    deinterld_bound.specialize(cfa_pattern == RGGB);
                    deinterld_bound.specialize(cfa_pattern == GRBG);
                    deinterld_bound.specialize(cfa_pattern == BGGR);
                    deinterld_bound.specialize(cfa_pattern == GBRG);
                    input_bound.compute_inline();
                    break;

                case 32:
                    // unroll c:
                    //  for yo:
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //      interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_inline();
                    break;

                case 33:
                    // unroll c:
                    //  allocate input_bound
                    //  for yo:
                    //   for c:
                    //    for y:
                    //     for x:
                    //      input_bound(x, y, c)
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //      interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo).store_at(img_output, c);
                    break;

                case 34:
                    // unroll c:
                    //  allocate input_bound
                    //  for yo:
                    //   for c:
                    //    for y:
                    //     for x:
                    //      input_bound(x, y, c)
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //      interpolation_y(x, y, c)
                    //   for yi:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_x.specialize(cfa_pattern == RGGB);
                    interpolation_x.specialize(cfa_pattern == GRBG);
                    interpolation_x.specialize(cfa_pattern == BGGR);
                    interpolation_x.specialize(cfa_pattern == GBRG);
                    interpolation_y.compute_inline();
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo).store_at(img_output, c);
                    break;

                case 35:
                    // allocate input_bound
                    // for yo:
                    //  for c:
                    //   for y:
                    //    for x:
                    //     input_bound(x, y, c)
                    //  allocate interpolation_y
                    //  unroll c:
                    //   for y:
                    //    for x:
                    //     interpolation_y(x, y, c)
                    //  for yi:
                    //   unroll c:
                    //    for x:
                    //     img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .reorder(x, c, yi, yo)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo).store_root();
                    break;

                case 36:
                    // parallel yo_o:
                    //  allocate input_bound
                    //  for yo_i:
                    //   for c:
                    //    for y:
                    //     for x:
                    //      input_bound(x, y, c)
                    //   allocate interpolation_y
                    //   unroll c:
                    //    for y:
                    //     for x:
                    //     interpolation_y(x, y, c)
                    //   for yi:
                    //    unroll c:
                    //     for x:
                    //      img_output(x, y, c)
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 4).parallel(yo_o)
                        .reorder(x, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 37:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 8).parallel(yo_o)
                        .reorder(x, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 38:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .reorder(x, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 39:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 32).parallel(yo_o)
                        .reorder(x, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 40:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 64).parallel(yo_o)
                        .reorder(x, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 41:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 42:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 43:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 44:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size*2).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 45:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size*4).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 46:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, 2).unroll(xi)
                        .split(xo, xo_o, xo_i, vector_size/2).vectorize(xo_i)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                case 47:
                    img_output
                        .compute_root()
                        .bound(c, 0, 3).unroll(c)
                        .align_bounds(y, 2, 0).split(y, yo, yi, 2)
                        .split(yo, yo_o, yo_i, 16).parallel(yo_o)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, xo, c, yi, yo_i, yo_o)
                    ;
                    white_balancing.compute_inline();
                    interpolation.compute_inline();
                    interpolation_x.compute_inline();
                    interpolation_y
                        .compute_at(img_output, yo_i)
                        .unroll(c)
                        .unroll(y)
                        .align_bounds(x, 2, 1).split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    interpolation_y.specialize(cfa_pattern == RGGB);
                    interpolation_y.specialize(cfa_pattern == GRBG);
                    interpolation_y.specialize(cfa_pattern == BGGR);
                    interpolation_y.specialize(cfa_pattern == GBRG);
                    deinterld_bound.compute_inline();
                    input_bound.compute_at(img_output, yo_i).store_at(img_output, yo_o);
                    break;

                default:
                    break;
                }
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

        output(x, y, c) = u16(0);

        return output;
    }

};
HALIDE_REGISTER_GENERATOR(HalideDemosaic, demosaic);
