/*
for i in $(seq 0 4); do make SCHEDULER=$i; done
*/
#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideCD : public Generator<HalideCD> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<float> sigma{"sigma"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<uint32_t> scheduler{"scheduler", 3};
        GeneratorParam<int> split_size{"split_size", 32};

        void generate() {
            Expr gaussian_width = i32(3*sigma + 0.5f);
            Expr kernel_size = 2*gaussian_width + 1;
            kernel = RDom(-gaussian_width, kernel_size, "kernel");
            weights(x) = exp(-(x*x)/(2.f*sigma*sigma));
            sum_weights() += weights(kernel);
            w(x) = weights(x)/sum_weights();

            input_f32(x, y, c) = f32(BoundaryConditions::repeat_edge(img_input)(x, y, c));

            yCbCr_input = rgb_to_YCbCr(input_f32);

            yCbCr_blurred_intm(x, y, c) += w(kernel) * yCbCr_input(x, y + kernel, c);
            yCbCr_blurred(x, y, c) += w(kernel) * yCbCr_blurred_intm(x + kernel, y, c);

            yCbCr_output(x, y, c) = mux(c, {yCbCr_input(x, y, 0), yCbCr_blurred(x, y, 1), yCbCr_blurred(x, y, 2)});

            output_f32 = YCbCr_to_rgb(yCbCr_output);

            img_output(x, y, c) = u8_sat(output_f32(x, y, c));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                sigma.set_estimate(1.5);
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();
                switch (scheduler) {
                    default:
                    case 0:
                        weights.compute_root();
                        sum_weights.compute_root();
                        w.compute_root();
                        img_output
                            .compute_root()
                            .bound(c, 0, 3)
                            .unroll(c) // eliminar o mux/select do output_f32
                            .split(y, yo, yi, split_size).parallel(yo)
                            .vectorize(x, vector_size)
                            .reorder(x, c, yi, yo) // paralelizar a variável mais externa
                        ;
                        yCbCr_output // não testei, mas talvez ela inline seria melhor
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo) // não testei, mas talvez não precise do store_at
                            .unroll(c) // eliminar o mux/select
                            .vectorize(x, vector_size) //.split(x, x, xi, vector_size).vectorize(xi)
                            .reorder(x, c, y) // talvez seja uma boa alternativa de teste de diferente schedule colocar o c mais interno
                            //.split(c, x, y) <-> .split(xi, c, x, y) -> não testei
                        ;
                        yCbCr_blurred_intm
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred_intm.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c) // talvez seja uma boa alternativa colocar o kernel depois xo
                            //.reorder(xi, xo, kernel, y, c) -> não testei
                        ;
                        yCbCr_blurred
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo) // não testei, mas talvez não precise do store_at
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_input
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c) // eliminar o mux/select
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        break;

                    case 1:
                        weights.compute_root();
                        sum_weights.compute_root();
                        w.compute_root();
                        img_output
                            .compute_root()
                            .bound(c, 0, 3)
                            .unroll(c)
                            .split(y, yo, yi, split_size).parallel(yo)
                            .vectorize(x, vector_size)
                            .reorder(x, c, yi, yo)
                        ;
                        yCbCr_output
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_blurred_intm
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred_intm.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_blurred
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        // yCbCr_input usado em yCbCr_blurred_intm e em yCbCr_output apresentam domínios diferentes
                        yCbCr_input.in(yCbCr_blurred_intm)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_input.in(yCbCr_output)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                        ;
                        break;

                    case 2:
                        weights.compute_root();
                        sum_weights.compute_root();
                        w.compute_root();
                        img_output
                            .compute_root()
                            .bound(c, 0, 3)
                            .unroll(c)
                            .split(y, yo, yi, split_size).parallel(yo)
                            .vectorize(x, vector_size)
                            .reorder(x, c, yi, yo)
                        ;
                        yCbCr_output
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_blurred_intm
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred_intm.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_blurred
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        // a mesma coisa do scheduler 1, mas usando clone_in
                        yCbCr_input
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_input.clone_in(yCbCr_output) //o unroll, o vectorize e o reorder são copiados
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                        ;
                        break;

                    case 3:
                        weights.compute_root();
                        sum_weights.compute_root();
                        w.compute_root();
                        img_output
                            .compute_root()
                            .bound(c, 0, 3)
                            .unroll(c)
                            .split(y, yo, yi, split_size).parallel(yo)
                            .vectorize(x, vector_size)
                            .reorder(x, c, yi, yo)
                        ;
                        yCbCr_output
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        // É basicamente como o compute_with, mas não podemos fazer entre updates
                        // Contra: a cópia
                        // BEFORE
                        // for c:
                        //     for y:
                        //         for xo:
                        //             vectorize xi:
                        //                 compute yCbCr_blurred_intm
                        // for c:
                        //     for y:
                        //         for xo:
                        //             for kernel:
                        //                 vectorize xi:
                        //                     update yCbCr_blurred_intm
                        // AFTER
                        // for c:
                        //     for y:
                        //         for xo:
                        //             vectorize xi:
                        //                 compute yCbCr_blurred_intm
                        //             for kernel:
                        //                 vectorize xi:
                        //                     update yCbCr_blurred_intm
                        //             vectorize xi:
                        //                 copy yCbCr_burred_intm to global_wrapper
                        yCbCr_blurred_intm.in()
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .split(x, xo, xi, vector_size).vectorize(xi)
                        ;
                        yCbCr_blurred_intm
                            .compute_at(yCbCr_blurred_intm.in(), xo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred_intm.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_blurred.in()
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .split(x, xo, xi, vector_size).vectorize(xi)
                        ;
                        yCbCr_blurred
                            .compute_at(yCbCr_blurred.in(), xo)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_input.in(yCbCr_blurred_intm)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_input.in(yCbCr_output)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                        ;
                        break;

                    case 4:
                        weights.compute_root();
                        sum_weights.compute_root();
                        w.compute_root();
                        img_output
                            .compute_root()
                            .bound(c, 0, 3)
                            .unroll(c)
                            .split(y, yo, yi, split_size).parallel(yo)
                            .vectorize(x, vector_size)
                            .reorder(x, c, yi, yo)
                        ;
                        yCbCr_output
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        // BEFORE
                        // for c:
                        //     for y:
                        //         for xo:
                        //             vectorize xi:
                        //                 compute yCbCr_blurred_intm
                        // for c:
                        //     for y:
                        //         for xo:
                        //             for kernel:
                        //                 vectorize xi:
                        //                     update yCbCr_blurred_intm
                        // AFTER
                        // for c:
                        //     for y:
                        //         for xo:
                        //             vectorize xi:
                        //                 compute yCbCr_blurred_intm
                        //         for xo:
                        //             for kernel:
                        //                 vectorize xi:
                        //                     update yCbCr_blurred_intm
                        //         for xo:
                        //             vectorize xi:
                        //                 copy yCbCr_burred_intm to global_wrapper
                        yCbCr_blurred_intm.in()
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .split(x, xo, xi, vector_size).vectorize(xi)
                        ;
                        yCbCr_blurred_intm
                            .compute_at(yCbCr_blurred_intm.in(), y)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred_intm.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_blurred.in()
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .split(x, xo, xi, vector_size).vectorize(xi)
                        ;
                        yCbCr_blurred
                            .compute_at(yCbCr_blurred.in(), y)
                            .vectorize(x, vector_size)
                        ;
                        yCbCr_blurred.update()
                            .split(x, xo, xi, vector_size).vectorize(xi)
                            .reorder(xi, kernel, xo, y, c)
                        ;
                        yCbCr_input.in(yCbCr_blurred_intm)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                            .reorder(x, c, y)
                        ;
                        yCbCr_input.in(yCbCr_output)
                            .compute_at(img_output, yi)
                            .store_at(img_output, yo)
                            .unroll(c)
                            .vectorize(x, vector_size)
                        ;
                        break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var yo{"yo"}, yi{"yi"}, xo{"xo"}, xi{"xi"};
        RDom kernel;

        Func weights{"weights"}, sum_weights{"sum_weights"}, w{"w"};
        Func input_f32{"input_f32"}, output_f32{"output_f32"};
        Func yCbCr_input;
        Func yCbCr_blurred_intm{"yCbCr_blurred_intm"};
        Func yCbCr_blurred{"yCbCr_blurred"};
        Func yCbCr_output{"yCbCr_output"};

        Func rgb_to_YCbCr(Func input) {
            Func output("yCbCr_input");

            Expr r = input(x, y, 0);
            Expr g = input(x, y, 1);
            Expr b = input(x, y, 2);

            Expr yy = 0.299f*r + 0.587f*g + 0.114f*b;
            Expr cb = 128.f - 0.168736f*r - 0.331264f*g + 0.5f*b;
            Expr cr = 128.f + 0.5f*r - 0.418688f*g - 0.081312f*b;

            output(x, y, c) = mux(c, {yy, cb, cr});

            return output;
        }

        Func YCbCr_to_rgb(Func input) {
            Func output("rgb_output");

            Expr yy = input(x, y, 0);
            Expr cb = input(x, y, 1);
            Expr cr = input(x, y, 2);

            Expr r = yy + 1.402f * (cr-128.f);
            Expr g = yy - 0.344136f * (cb-128.f) - 0.714136f * (cr-128.f);
            Expr b = yy + 1.772f * (cb-128.f);

            output(x, y, c) = mux(c, {r, g, b});

            return output;
        }
};
HALIDE_REGISTER_GENERATOR(HalideCD, cd);
