/* run all schedulers using
for i in $(seq 0 11); do make SCHEDULER=$i; done
*/
/* test different split size for schedulers 3 to 10 using
for i in $(seq 4 11); do for ss in 8 16 32 64 128 256 512; do make SCHEDULER=$i SPLIT_SIZE=$ss; done; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideMatMul : public Generator<HalideMatMul> {
    public:
        Input<Buffer<float>> input1{"input1", 2};
        Input<Buffer<float>> input2{"input2", 2};

        Output<Buffer<float>> output{"output", 2};

        GeneratorParam<uint32_t> scheduler{"scheduler", 10};
        GeneratorParam<int> split_size{"split_size", 32};

        void generate() {
            r = RDom(0, input2.width(), "r");

            output(x, y) += input1(x, r) * input2(r, y);

            add_requirement(input1.height() == input2.width(), "matrixes do not multiply");
            add_requirement(input1.width() == output.width(), "output width is wrong");
            add_requirement(input2.height() == output.height(), "output height is wrong");
        }

        void schedule() {
            if (auto_schedule) {
                input1.set_estimates({{0, 3000}, {0, 3000}});
                input2.set_estimates({{0, 3000}, {0, 3000}});
                output.set_estimates({{0, 3000}, {0, 3000}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();
                switch (scheduler)
                {
                case 0:
                    // parallel y:
                    //     for x:
                    //         vectorize x:
                    //             output(x, y)
                    // parallel y:
                    //     for x:
                    //         for r:
                    //             update output(x, y)
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    // vectorize(x, vector_size) -> split(x, x, xi, vector_size).vectorize(xi) -> xi anônimo
                    // unroll(c, 4) -> split(c, c, ci, 4).unroll(ci) -> ci anônimo
                    // parallel(y, split_size) -> split(y, y, yi, vector_size).parallel(y) -> yi anônimo
                    // não aconselho o uso principalmente com o parallel pois perdemos a referência para o loop mais interno
                    output.update()
                        .parallel(y)
                    ;
                    break;

                case 1:
                    // parallel y:
                    //     for x:
                    //         vectorize x:
                    //             output(x, y)
                    // parallel y:
                    //     for r:
                    //         for x:
                    //             update output(x, y)
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                        .reorder(x, r, y)
                    ;
                    break;

                case 2:
                    // input1_in(_0, _1) = input1(_1, _0); // similarmente a uma transposição
                    // com input1.in(), todos os acessos de input1 será através de input.in()/input_in
                    // output(x, y) += input1_in(x, r) * input2(r, y);
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        // _0 -> primeira variável (geralmente usamos x),
                        // _1 -> segunda variável (geralmente usamos y)
                        .parallel(_1)
                    ;
                    break;

                case 3:
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        .reorder(_1, _0)
                        .parallel(_0)
                    ;
                    break;

                case 4:
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        .split(_0, xo, xi, split_size).split(_1, yo, yi, split_size).reorder(xi, yi, xo, yo)
                        // .tile(x, y, xo, yo, xi, yi, split_size_x, split_size_y) ->
                        //   .split(x, xo, xi, split_size_x)
                        //   .split(y, yo, yi, split_size_y)
                        //   .reorder(xi, yi, xo, yo)
                        // .tile(x, y, xi, yi, split_size_x, split_size_y) ->
                        //    .tile(x, y, x, y, xi, yi, split_size_x, splite_size_y)
                        .parallel(yo)
                    ;
                    break;

                case 5: //fuse
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        .split(_0, xo, xi, split_size).split(_1, yo, yi, split_size).reorder(xi, yi, xo, yo)
                        .fuse(xo, yo, xy).parallel(xy)
                    ;
                    break;

                case 6:
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        .split(_0, xo, xi, split_size).split(_1, yo, yi, split_size).reorder(yi, xi, yo, xo)
                        .parallel(xo)
                    ;
                    break;

                case 7: //fuse
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .parallel(y)
                    ;
                    input1.in()
                        .compute_root()
                        .reorder_storage(_1, _0)
                        .split(_0, xo, xi, split_size).split(_1, yo, yi, split_size).reorder(yi, xi, yo, xo)
                        .fuse(yo, xo, yx).parallel(yx)
                    ;
                    break;

                case 8:
                    // parallel y:
                    //     for x:
                    //         vectorize x:
                    //             output(x, y)
                    // parallel yo:
                    //     for xo:
                    //         for ro:
                    //             for yi:
                    //                 for xi:
                    //                     for ri:
                    //                         update output(x, y)
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .split(x, xo, xi, split_size, TailStrategy::GuardWithIf)
                        .split(y, yo, yi, split_size, TailStrategy::GuardWithIf)
                        .split(r, ro, ri, split_size, TailStrategy::GuardWithIf)
                        .reorder(ri, xi, yi, ro, xo, yo)
                        .parallel(yo)
                    ;
                    break;

                case 9: //fuse
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .split(x, xo, xi, split_size, TailStrategy::GuardWithIf)
                        .split(y, yo, yi, split_size, TailStrategy::GuardWithIf)
                        .split(r, ro, ri, split_size, TailStrategy::GuardWithIf)
                        .fuse(xo, yo, xy)
                        .reorder(ri, xi, yi, ro, xy)
                        .parallel(xy)
                    ;
                    break;

                default:
                case 10:
                    // parallel y:
                    //     for x:
                    //         vectorize x:
                    //             output(x, y)
                    // parallel yo:
                    //     for xo:
                    //         for ro:
                    //             for yi:
                    //                 for ri:
                    //                     for xi:
                    //                         update output(x, y)
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .split(x, xo, xi, split_size, TailStrategy::GuardWithIf)
                        .split(y, yo, yi, split_size, TailStrategy::GuardWithIf)
                        .split(r, ro, ri, split_size, TailStrategy::GuardWithIf)
                        .reorder(xi, ri, yi, ro, xo, yo)
                        .parallel(yo)
                    ;
                    break;

                case 11: //fuse
                    output
                        .compute_root()
                        .parallel(y)
                        .vectorize(x, vector_size)
                    ;
                    output.update()
                        .split(x, xo, xi, split_size, TailStrategy::GuardWithIf)
                        .split(y, yo, yi, split_size, TailStrategy::GuardWithIf)
                        .split(r, ro, ri, split_size, TailStrategy::GuardWithIf)
                        .fuse(xo, yo, xy)
                        .reorder(xi, ri, yi, ro, xy)
                        .parallel(xy)
                    ;
                    break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"};
        Var xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"}, xy{"xy"}, yx{"yx"};

        RDom r;
        RVar ri{"ri"}, ro{"ro"};

    // Exercício:
    // Explorar outras opções de schedulers
    // Sugestão: reorder(xi, ri, yi, xo, ro, yo)
};
HALIDE_REGISTER_GENERATOR(HalideMatMul, mat_mul);
