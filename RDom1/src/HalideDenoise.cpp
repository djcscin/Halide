/* run all versions using
for p in true false; do for s in $(seq 0 11); do make PREDICATE=$p SCHEDULER=$s; done; done
*/
#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideDenoise : public Generator<HalideDenoise> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<float> sigma_spatial{"sigma_spatial"};
        Input<float> sigma_range{"sigma_range"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<uint> scheduler{"scheduler", 0};
        GeneratorParam<bool> predicate{"predicate", false};

        Expr gaussian(Expr i, Expr sigma) {
            return exp(-(i*i)/(2.f*sigma*sigma));
        }

        void generate() {
            Expr gaussian_width = max(i32(3.f*sigma_spatial), 1);
            Expr kernel_size = 2*gaussian_width + 1;
            // Se o kernel_size for 3 gaussian_width é 1 -> kernel = RDom(-1, 3, -1, 3) -> min, extent
            // RVar:
            // kernel.x  kernel.y
            // kernel[0] kernel[1] RDom retangular
            //       -1       -1
            //        0       -1
            //        1       -1
            //       -1        0
            //        0        0
            //        1        0
            //       -1        1
            //        0        1
            //        1        1
            // predicate: -> RDom não-retangular
            // abs(kernel[0]) + abs(kernel[1]) <= 1
            //        0       -1
            //       -1        0
            //        0        0
            //        1        0
            //        0        1
            kernel = RDom(-gaussian_width, kernel_size, -gaussian_width, kernel_size, "kernel");
            if(predicate) {
                // distância for menor do que 3*sigma_spatial
                // sqrt(kernel.x*kernel.x + kernel.y*kernel.y) < 3*sigma_spatial
                // kernel.x*kernel.x + kernel.y*kernel.y < 9*sigma_spatial*sigma_spatial
                // kernel.x or kernel[0]
                // kernel.y or kernel[1]
                kernel.where(kernel[0]*kernel.x + kernel[1]*kernel.y < 9.f*sigma_spatial*sigma_spatial);
            }

            input_bound(x, y, c) = i16(BoundaryConditions::mirror_interior(img_input)(x, y, c));
            diff(x, y, i, j, c) = absd(input_bound(x, y, c), input_bound(x + i, y + j, c));
            norm(x, y, i, j) = u16(diff(x, y, i, j, 0) + diff(x, y, i, j, 1) + diff(x, y, i, j, 2));

            weights_spatial(i) = gaussian(i, sigma_spatial);
            weights_range(i) = gaussian(i, sigma_range);
            weights(x, y, i, j) = weights_spatial(i) * weights_spatial(j) * weights_range(norm(x, y, i, j));

            // redução de domínio
            // blur
            // 1st
            // blur_x(x, y) = input_bound(x, y - 1) + input_bound(x, y) + input_bound(x, y + 1);
            // blur_y(x, y) = (blur_x(x - 1, y) + blur_x(x, y) + blur(x, y + 1))/9;
            // 2nd RDom
            // RDom r(-1, 3);
            // blur_x(x, y) = 0; // difinição pura subtendida
            // blur_x(x, y) += input_bound(x, y + r);
            // blur_y(x, y) = 0; // não precisa colocar
            // blur_y(x, y) += blur_x(x + r, y);
            // blur_y(x, y) /= 9;

            // blur_x(x, y) += input_bound(x, y + r); -> blur_x(x, y) = sum(input_bound(x, y+r), "sum_x");

            output(x, y, c) += select(c == -1, weights(x, y, kernel.x, kernel.y),
                                weights(x, y, kernel.x, kernel.y) * input_bound(x + kernel.x, y + kernel.y, c));
            // sum_weights(x, y) += weights(x, y, kernel.x, kernel.y);
            // sum_conv(x, y, c) += weights(x, y, kernel.x, kernel.y) * input_bound(x + kernel.x, y + kernel.y, c);
            // img_output(x, y, c) = sum_conv(x, y, c) / sum_weights(x, y);

            img_output(x, y, c) = u8_sat(output(x, y, c)/output(x, y, -1));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                sigma_spatial.set_estimate(1.5);
                sigma_range.set_estimate(10.0);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();

                switch (scheduler)
                {
                case 0:
                    // LUTs com compute_root
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // output têm estágios de update
                    // não pode ser inline
                    // weights não depende de c -> c mais interno
                    // unroll c do output pra eliminar select
                    // parallel img_output.y:
                    //  allocate output
                    //  for output.y:
                    //   for output.c:
                    //    for output.xo:
                    //     vectorize output.xi:
                    //      output(x, y, c)
                    //  for output.y:
                    //   for output.xo:
                    //    for kernel.y:
                    //     for kernel.x:
                    //      unroll output.c:
                    //       vectorize output.xi:
                    //        update output(x, y, c)
                    // for img_output.xo:
                    //  for img_output.c:
                    //   vectorize img_output.xi:
                    //    img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .parallel(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    output
                        .compute_at(img_output, y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, kernel.y, xo, y)
                        // a ordem default seria kernel.x, kernel.y, x, y, c
                        // com vetorização, xi, kernel.x, kernel.y, xo, y, c
                    ;
                    break;

                case 1:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.y:
                    //  allocate output
                    //  for output.y:
                    //   for output.c:
                    //    for output.xo:
                    //     vectorize output.xi:
                    //      output(x, y, c)
                    //  for output.y:
                    //   for kernel.y:
                    //    for output.xo:
                    //     for kernel.x:
                    //      unroll output.c:
                    //       vectorize output.xi:
                    //        update output(x, y, c)
                    // for img_output.xo:
                    //  for img_output.c:
                    //   vectorize img_output.xi:
                    //    img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .parallel(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    output
                        .compute_at(img_output, y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    break;

                case 2:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.y:
                    //  allocate output
                    //  for output.y:
                    //   for output.c:
                    //    for output.xo:
                    //     vectorize output.xi:
                    //      output(x, y, c)
                    //  for output.y:
                    //   for kernel.y:
                    //    for output.xo:
                    //      for kernel.x:
                    //       allocate weights
                    //       for weights.j:
                    //        for weights.i:
                    //         for weights.y:
                    //          for weights.xo:
                    //           vectorize xi:
                    //            weights(x, y, i, j)
                    //       unroll output.c:
                    //        vectorize output.xi:
                    //          update output(x, y, c)
                    //  for img_output.xo:
                    //   for img_output.c:
                    //    vectorize img_output.xi:
                    //     img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .parallel(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    output
                        .compute_at(img_output, y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    break;

                case 3:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.y:
                    //  allocate output
                    //  for output.y:
                    //   for output.c:
                    //    for output.xo:
                    //     vectorize output.xi:
                    //      output(x, y, c)
                    //  for output.y:
                    //   for kernel.y:
                    //    for output.xo:
                    //     allocate weights
                    //     for weights.j:
                    //      for weights.i:
                    //       for weights.y:
                    //        for weights.x:
                    //         weights(x, y, i, j)
                    //     for kernel.x:
                    //      unroll output.c:
                    //       vectorize output.xi:
                    //        update output(x, y, c)
                    //  for img_output.xo:
                    //   for img_output.c:
                    //    vectorize img_output.xi:
                    //     img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .parallel(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    output
                        .compute_at(img_output, y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, xo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    break;

                case 4:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.y:
                    //  allocate output
                    //  for output.y:
                    //   for output.c:
                    //    for output.xo:
                    //     vectorize output.xi:
                    //      output(x, y, c)
                    //  for output.y:
                    //   allocate input_bound
                    //   for input_bound.c:
                    //    for input_bound.y:
                    //     for input_bound.x:
                    //      input_bound(x, y, c)
                    //   for kernel.y:
                    //    for output.xo:
                    //     for kernel.x:
                    //      allocate weights
                    //      for weights.j:
                    //       for weights.i:
                    //        for weights.y:
                    //         for weights.xo:
                    //          vectorize xi:
                    //           weights(x, y, i, j)
                    //      unroll output.c:
                    //       vectorize output.xi:
                    //        update output(x, y, c)
                    //  for img_output.xo:
                    //   for img_output.c:
                    //    vectorize img_output.xi:
                    //     img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .parallel(y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, y)
                    ;
                    output
                        .compute_at(img_output, y)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y);
                    break;

                case 5:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.yo:
                    //  allocate input_bound
                    //  for img_output.yi:
                    //   allocate output
                    //   for output.y:
                    //    for output.c:
                    //     for output.xo:
                    //      vectorize xi:
                    //       output(x, y, c)
                    //   for output.y:
                    //    for input_bound.c:
                    //     for input_bound.y:
                    //      for input_bound.x:
                    //       input_bound(x, y, c)
                    //    for kernel.y:
                    //     for output.xo:
                    //      for kernel.x:
                    //       allocate weights
                    //       for weights.j:
                    //        for weights.i:
                    //         for weights.y:
                    //          for weights.xo:
                    //           vectorize weights.xi:
                    //            weights(x, y, i, j)
                    //       unroll output.c:
                    //        vectorize output.xi:
                    //          update output(x, y, c)
                    //   for img_output.xo:
                    //    for img_output.c:
                    //     vectorize img_output.xi:
                    //      img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 4).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, yi)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 6:
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 8).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, yi)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 7:
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 16).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, yi)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 8:
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 32).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, yi)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 9:
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, yi)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 10:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.yo:
                    //  allocate input_bound
                    //  for img_output.yi:
                    //   for img_output.xo:
                    //    allocate output
                    //    for output.y:
                    //     for output.c:
                    //      for output.xo:
                    //       vectorize xi:
                    //        output(x, y, c)
                    //    for output.y:
                    //     for input_bound.c:
                    //      for input_bound.y:
                    //       for input_bound.x:
                    //        input_bound(x, y, c)
                    //     for kernel.y:
                    //      for output.xo:
                    //       for kernel.x:
                    //        allocate weights
                    //        for weights.j:
                    //         for weights.i:
                    //          for weights.y:
                    //           for weights.xo:
                    //            vectorize weights.xi:
                    //             weights(x, y, i, j)
                    //        unroll output.c:
                    //         vectorize output.xi:
                    //          update output(x, y, c)
                    //    for img_output.c:
                    //     vectorize img_output.xi:
                    //      img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 16).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, xo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    weights
                        .compute_at(output, kernel.x)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                case 11:
                    // for i:
                    //  weights_spatial(i)
                    // for i:
                    //  weights_range(i)
                    // parallel img_output.yo:
                    //  allocate input_bound
                    //  for img_output.yi:
                    //   for img_output.xo:
                    //    allocate output
                    //    for output.y:
                    //     for output.c:
                    //      for output.xo:
                    //       vectorize xi:
                    //        output(x, y, c)
                    //    for output.y:
                    //     for input_bound.c:
                    //      for input_bound.y:
                    //       for input_bound.x:
                    //        input_bound(x, y, c)
                    //     for kernel.y:
                    //      for output.xo:
                    //       for kernel.x:
                    //        unroll output.c:
                    //         vectorize output.xi:
                    //          update output(x, y, c)
                    //    for img_output.c:
                    //     vectorize img_output.xi:
                    //      img_output(x, y, c)
                    weights_spatial
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    weights_range
                        .compute_root()
                        .split(i, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_output
                        .compute_root()
                        .bound(c, 0, 3)
                        .split(y, yo, yi, 16).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo)
                    ;
                    output
                        .compute_at(img_output, xo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    output.update()
                        .unroll(c)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .reorder(xi, c, kernel.x, xo, kernel.y, y)
                    ;
                    input_bound.compute_at(output, y).store_at(img_output, yo);
                    break;

                default:
                    break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"}, i{"i"}, j{"j"};
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"};
        RDom kernel;

        Func input_bound{"input_bound"}, diff{"diff"}, norm{"norm"};
        Func weights_spatial{"weights_spatial"}, weights_range{"weights_range"};
        Func weights{"weights"}, sum_weights{"sum_weights"}, weights_norm{"weights_norm"};
        Func output{"output"};

    // Exercício:
    // Explorar outras opções de schedulers
    // Sugestão 1: fazer reorder com weights.compute_at(output, kernel.x) no case 2
    // Sugestão 2: explorar diferentes input_bound.compute_at

};
HALIDE_REGISTER_GENERATOR(HalideDenoise, denoise);
