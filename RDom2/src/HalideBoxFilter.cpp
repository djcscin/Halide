/* run all versions using
for m in f sf i si; do for s in $(seq 1 4); do make METHOD=$m SCHEDULER=$s; done; done
*/
#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

enum BoxFilterMethod {
    FILTER = 0,
    SEPARABLE_FILTER,
    INTEGRAL_IMAGE,
    SEPARABLE_INTEGRAL_IMAGE,
};

class HalideBoxFilter : public Generator<HalideBoxFilter> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<uint> window_size{"window_size"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<uint> scheduler{"scheduler", 0};
        GeneratorParam<enum BoxFilterMethod> method = {"method", SEPARABLE_FILTER,
            {
                {"f", FILTER},
                {"sf", SEPARABLE_FILTER},
                {"i", INTEGRAL_IMAGE},
                {"si", SEPARABLE_INTEGRAL_IMAGE},
            }
        };

        void generate() {
            switch (method) {
                case FILTER:
                    // redução de domínio
                    // window_size = 2
                    // 2 dimensões - 2 RVar
                    // kernel.x  kernel.y
                    // kernel[0] kernel[1]
                    //        0        0
                    //        0        1
                    //        1        0
                    //        1        1
                    kernel = RDom(0, i32(window_size), 0, i32(window_size));
                    sum_kernel(x, y, c) += u32(img_input(x + kernel.x, y + kernel.y, c));
                    break;

                case SEPARABLE_FILTER:
                    kernel = RDom(0, i32(window_size));
                    sum1(x, y, c) += u32(img_input(x, y + kernel, c));
                    sum_kernel(x, y, c) += sum1(x + kernel, y, c);
                    break;

                case INTEGRAL_IMAGE:
                    kernel = RDom(img_input);
                    sum_i(x, y, c) = undef<uint64_t>();
                    sum_i(-1, y, c) = u64(0);
                    sum_i(x, -1, c) = u64(0);
                    sum_i(kernel.x, kernel.y, c) =
                          img_input(kernel.x, kernel.y,     c)
                        + sum_i(kernel.x - 1, kernel.y    , c)
                        + sum_i(kernel.x    , kernel.y - 1, c)
                        - sum_i(kernel.x - 1, kernel.y - 1, c)
                    ;
                    break;

                case SEPARABLE_INTEGRAL_IMAGE:
                    kernel1 = RDom(0, img_input.height());
                    sum1(x, y, c) = undef<uint32_t>();
                    sum1(x, -1, c) = u32(0);
                    sum1(x, kernel1, c) = img_input(x, kernel1, c) + sum1(x, kernel1 - 1, c);

                    kernel = RDom(0, img_input.width());
                    sum_i(x, y, c) = undef<uint64_t>();
                    sum_i(-1, y, c) = u64(0);
                    sum_i(kernel, y, c) = sum1(kernel, y, c) + sum_i(kernel - 1, y, c);
                    break;
            }

            switch (method) {
                case INTEGRAL_IMAGE:
                case SEPARABLE_INTEGRAL_IMAGE:
                    sum_kernel(x, y, c) = u32(
                          sum_i(x - 1              , y - 1              , c)
                        - sum_i(x - 1 + window_size, y - 1              , c)
                        - sum_i(x - 1              , y - 1 + window_size, c)
                        + sum_i(x - 1 + window_size, y - 1 + window_size, c)
                    );
                    break;
            }

            img_output(x, y, c) = u8(sum_kernel(x, y, c)/(window_size * window_size));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                window_size.set_estimate(10);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<uint32_t>();
                switch (method) {
                    case FILTER:
                        switch(scheduler) {
                            // diferenças estão em sum_kernel.compute_at e sum_kernel.update().reorder
                            default:
                            case 1:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel.x, kernel.y, xo, y, c)
                                ;
                                break;
                            case 2:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel.x, xo, kernel.y, y, c)
                                ;
                                break;
                            case 3:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel.x, kernel.y, xo, y, c)
                                ;
                                break;
                            case 4:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel.x, xo, kernel.y, y, c)
                                ;
                                break;
                        }
                        break;

                    case SEPARABLE_FILTER:
                        switch(scheduler) {
                            // diferenças estão em sum_kernel.compute_at e sum1.compute_at.store_at
                            case 1:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                sum1
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                break;
                            case 2:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                sum1
                                    .compute_at(img_output, xo)
                                    .store_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                break;
                            case 3:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                sum1
                                    .compute_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                break;
                            default:
                            case 4:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel.update()
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                sum1
                                    .compute_at(img_output, yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1.update()
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .reorder(xi, kernel, xo, y, c)
                                ;
                                break;
                        }
                        break;

                    case INTEGRAL_IMAGE:
                        switch(scheduler) {
                            // diferenças estão em sum_i.compute_at, sum_i.update(0) e sum_kernel.compute_at
                            case 1:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_i
                                    .compute_root()
                                ;
                                sum_i.update(0)
                                    .split(y, yo, yi, 64).fuse(yo, c, yc).parallel(yc)
                                ;
                                sum_i.update(1)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum_i.update(2)
                                    .parallel(c)
                                ;
                                break;
                            default:
                            case 2:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_i
                                    .compute_root()
                                ;
                                sum_i.update(0)
                                    .split(y, yo, yi, vector_size).vectorize(yi)
                                    .split(yo, yo_o, yo_i, 8).fuse(yo_o, c, yc).parallel(yc)
                                ;
                                sum_i.update(1)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum_i.update(2)
                                    .parallel(c)
                                ;
                                break;
                            case 3:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_i
                                    .compute_root()
                                ;
                                sum_i.update(0)
                                    .split(y, yo, yi, vector_size).vectorize(yi)
                                    .split(yo, yo_o, yo_i, 8).fuse(yo_o, c, yc).parallel(yc)
                                ;
                                sum_i.update(1)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum_i.update(2)
                                    .parallel(c)
                                ;
                                break;
                            case 4:
                                img_output
                                    .compute_root()
                                    .parallel(c)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_i
                                    .compute_at(img_output, c)
                                ;
                                sum_i.update(0)
                                    .split(y, yo, yi, vector_size).vectorize(yi)
                                ;
                                sum_i.update(1)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                break;
                        }
                        break;

                    case SEPARABLE_INTEGRAL_IMAGE:
                        switch(scheduler) {
                            // diferenças estão em sum_i.compute_at e sum_kernel.compute_at
                            case 1:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1
                                    .compute_root()
                                ;
                                sum1.update(0)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum1.update(1)
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                    .reorder(xi, kernel1, xo_i, xc)
                                ;
                                sum_i
                                    .compute_root()
                                ;
                                sum_i.update(0)
                                    .split(y, yo, yi, vector_size).vectorize(yi)
                                    .split(yo, yo_o, yo_i, 8).fuse(yo_o, c, yc).parallel(yc)
                                ;
                                sum_i.update(1)
                                    .split(y, yo, yi, vector_size).vectorize(yi)
                                    .split(yo, yo_o, yo_i, 8).fuse(yo_o, c, yc).parallel(yc)
                                    .reorder(yi, kernel, yo_i, yc)
                                ;
                                break;
                            case 2:
                                img_output
                                    .compute_root()
                                    .fuse(y, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1
                                    .compute_root()
                                ;
                                sum1.update(0)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum1.update(1)
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                    .reorder(xi, kernel1, xo_i, xc)
                                ;
                                sum_i
                                    .compute_at(img_output, yc)
                                ;
                                break;
                            default:
                            case 3:
                                img_output
                                    .compute_root()
                                    .split(y, yo, yi, 128).fuse(yo, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1
                                    .compute_root()
                                ;
                                sum1.update(0)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum1.update(1)
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                    .reorder(xi, kernel1, xo_i, xc)
                                ;
                                sum_i
                                    .compute_at(img_output, yi)
                                    .store_at(img_output, yc)
                                ;
                                break;
                            case 4:
                                img_output
                                    .compute_root()
                                    .split(y, yo, yi, 128).fuse(yo, c, yc).parallel(yc)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum_kernel
                                    .compute_at(img_output, xo)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                ;
                                sum1
                                    .compute_root()
                                ;
                                sum1.update(0)
                                    .split(x, xo, xi, vector_size).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                ;
                                sum1.update(1)
                                    .split(x, xo, xi, vector_size, TailStrategy::GuardWithIf).vectorize(xi)
                                    .split(xo, xo_o, xo_i, 8).fuse(xo_o, c, xc).parallel(xc)
                                    .reorder(xi, kernel1, xo_i, xc)
                                ;
                                sum_i
                                    .compute_at(img_output, yi)
                                    .store_at(img_output, yc)
                                ;
                                break;
                        }
                        break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"};
        Var yc{"yc"}, xc{"xc"};
        Var xo_o{"xo_o"}, xo_i{"xo_i"};
        Var yo_o{"yo_o"}, yo_i{"yo_i"};
        RDom kernel1, kernel;

        Func sum1{"sum1"}, sum_i{"sum_i"}, sum_kernel{"sum_kernel"};

    // Exercício:
    // Explorar outras opções de schedulers

};
HALIDE_REGISTER_GENERATOR(HalideBoxFilter, boxfilter);
