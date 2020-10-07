/* run all schedulers using
for i in $(seq 0 34); do make SCHEDULER=$i; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideGradient : public Generator<HalideGradient> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<uint> threshold{"threshold"};

        Output<Buffer<uint8_t>> img_output{"img_output", 2};

        GeneratorParam<uint> scheduler{"scheduler", 0};

        void generate() {
            // img_gray seria a média dos canais, mas
            // em vez de dividir o img_gray por 3, multiplico o threshold por 9 (3*3)
            img_gray(x, y) = i32(img_input(x, y, 0)) + img_input(x, y, 1) + img_input(x, y, 2);

            // Sobel
            // em vez de dividir por 4, multiplico o threshold por 16 (4*4)
            //  1  2  1          1
            //  0  0  0 * 1/4 =  0 * 1 2 1 * 1/4
            // -1 -2 -1         -1
            int_gradient_y(x, y) = img_gray(x, y) - img_gray(x, y + 2);
            // int_gradient_y(0, 0) = img_gray(0, 0) - img_gray(0, 2);
            // int_gradient_y(0, 1) = img_gray(0, 1) - img_gray(0, 3);
            // int_gradient_y(0, 2) = img_gray(0, 2) - img_gray(0, 4);
            // int_gradient_y(0, 3) = img_gray(0, 3) - img_gray(0, 5);
            // na computação do int_gradient_y, de cada img_gray(x, y) precisa duas vezes
            gradient_y(x, y) = int_gradient_y(x, y) + 2*int_gradient_y(x + 1, y) + int_gradient_y(x + 2, y);
            // gradient_y(0, 0) = int_gradient_y(0, 0) + 2*int_gradient_y(1, 0) + int_gradient_y(2, 0);
            // gradient_y(1, 0) = int_gradient_y(1, 0) + 2*int_gradient_y(2, 0) + int_gradient_y(3, 0);
            // gradient_y(2, 0) = int_gradient_y(2, 0) + 2*int_gradient_y(3, 0) + int_gradient_y(4, 0);
            // na computação do gradient_y, de cada int_gradient_y(x, y) precisa três vezes
            // na computação do gradient_y, de cada img_gray(x, y) vai ser calculado seis vezes

            // 1 0 -1          1
            // 2 0 -2 * 1/4 =  2 * 1 0 -1 * 1/4
            // 1 0 -1          1
            int_gradient_x(x, y) = img_gray(x, y) + 2*img_gray(x, y + 1) + img_gray(x, y + 2);
            gradient_x(x, y) = int_gradient_x(x, y) - int_gradient_x(x + 2, y);

            // em vez de tirar a raiz quadrada da magnitude do gradiente, elevo o threshold ao quadrado
            Expr gy = gradient_y(x, y);
            Expr gx = gradient_x(x, y);
            gradient_mag(x, y) = gy*gy + gx*gx;

            thrshd() = i32(threshold)*threshold*9*16;

            //img_output(x, y) = select(gradient_mag(x, y) < thrshd(), 255, 0);
            img_output(x, y) = 255*u8(gradient_mag(x, y) < thrshd());
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                threshold.set_estimate(50);
                img_output.set_estimates({{0, 4000}, {0, 3000}});
            } else {
                int vector_size = get_target().natural_vector_size(Int(32));

                // a(x) = b(x) + c(x)
                // a(0 até 15) = b(0 até 15) + c(0 até 15) -> 16 somas com uma instrução só -> vetorização
                // 16, 32, 64, 128 bytes
                // 32 bytes -> soma com um vetor de 8 elementos de 4 bytes

                switch (scheduler)
                // evitar redundância de computação das funções
                // usar localidade de referência temporal
                // usar localidade de referência espacial
                // usar paralelização
                {
                case 0:
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    break;

                case 1:
                    // thrsh
                    // for x:
                    //  for y:
                    //    img_output(x, y)
                    img_output.compute_root().reorder(y, x);
                    // a ordem das variáveis sempre vai ser do mais interno pra o mais externo,
                    // exceto pro split
                    thrshd.compute_root();
                    break;

                // A variável mais interna tem que ser o for mais interno

                case 2:
                    // thrsh
                    // parallel y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root().parallel(y);
                    thrshd.compute_root();
                    break;

                case 3:
                    // thrsh
                    // for y:
                    //  parallel x:
                    //    img_output(x, y)
                    img_output.compute_root().parallel(x);
                    thrshd.compute_root();
                    break;

                // false sharing - a mesma linha de cache está sendo usada por threads diferentes
                // perdendo a noção de localidade de referência espacial
                // não podemos paralelizar a variável mais interna

                case 4:
                    // thrsh
                    // for y:
                    //  parallel xo:
                    //   for xi from 0 to 64:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 64).parallel(xo);
                    thrshd.compute_root();
                    break;

                case 5:
                    // thrsh
                    // for y:
                    //  parallel xo:
                    //   for xi from 0 to 128:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 128).parallel(xo);
                    thrshd.compute_root();
                    break;

                case 6:
                    // thrsh
                    // for y:
                    //  parallel xo:
                    //   for xi from 0 to 256:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 256).parallel(xo);
                    thrshd.compute_root();
                    break;

                case 7:
                    // thrsh
                    // for y:
                    //  parallel xo:
                    //   for xi from 0 to 512:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 512).parallel(xo);
                    thrshd.compute_root();
                    break;

                case 8:
                    // thrsh
                    // parallel y:
                    //  parallel xo:
                    //   for xi from 0 to 512:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 512).parallel(xo).parallel(y);
                    thrshd.compute_root();
                    break;

                case 9:
                    // thrsh
                    // parallel y,xo -> xy:
                    //   for xi from 0 to 512:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, 512).fuse(xo, y, xy).parallel(xy);
                    thrshd.compute_root();
                    break;

                case 10:
                    // thrsh
                    // for y:
                    //  for xo:
                    //   vectorize xi with vector_size:
                    //    img_output(x, y)
                    img_output.compute_root().split(x, xo, xi, vector_size).vectorize(xi);
                    thrshd.compute_root();
                    break;

                case 11:
                    // thrsh
                    // for yo:
                    //  for x:
                    //   vectorize yi with vector_size:
                    //    img_output(x, y)
                    img_output.compute_root().split(y, yo, yi, vector_size).vectorize(yi);
                    thrshd.compute_root();
                    break;

                // a vetorização é na variável mais interna

                case 12:
                    // se não colocamos store_at ou store_root, o local de alocação de memória será
                    // o mesmo do local de computação (compute_at, compute_root, compute_inline)
                    // thrsh
                    // for y:
                    //  allocate img_gray
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_at(img_output, y);
                    break;

                case 13:
                    // thrsh
                    // allocate img_gray
                    // for y:
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_at(img_output, y).store_root(); // fold_storage(y, 4)
                    break;

                case 14:
                    // thrsh
                    // for y:
                    //  allocate int_gradient_x
                    //  allocate int_gradient_y
                    //  for y:
                    //   for x:
                    //    int_gradient_x(x, y)
                    //    int_gradient_y(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x);
                    img_gray.compute_inline();
                    break;

                case 15:
                    // thrsh
                    // allocate int_gradient_x
                    // allocate int_gradient_y
                    // for y:
                    //  for y:
                    //   for x:
                    //    int_gradient_x(x, y)
                    //    int_gradient_y(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_at(img_output, y).store_root();
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x).store_root();
                    img_gray.compute_inline();
                    break;

                case 16:
                    // thrsh
                    // for y:
                    //  allocate img_gray
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  allocate int_gradient_x
                    //  allocate int_gradient_y
                    //  for y:
                    //   for x:
                    //    int_gradient_x(x, y)
                    //    int_gradient_y(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x);
                    img_gray.compute_at(img_output, y);
                    break;

                case 17:
                    // thrsh
                    // allocate img_gray
                    // allocate int_gradient_x
                    // allocate int_gradient_y
                    // for y:
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  for y:
                    //   for x:
                    //    int_gradient_x(x, y)
                    //    int_gradient_y(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    int_gradient_x.compute_at(img_output, y).store_root();
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x).store_root();
                    img_gray.compute_at(img_output, y).store_root();
                    break;

                case 18:
                    // thrsh
                    // allocate img_gray
                    // for y:
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  for x:
                    //    img_output(x, y)
                    // ----------------
                    // thrsh
                    // parallel yo:
                    //  allocate img_gray
                    //  for yi:
                    //   for y:
                    //    for x:
                    //     img_gray(x, y)
                    //   for x:
                    //    img_output(x, y)
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 4).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 19:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 8).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 20:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 16).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 21:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 32).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 22:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 23:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 128).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 24:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 256).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 25:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 512).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 26:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 1024).parallel(yo)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                    ;
                    break;

                case 27:
                    // thrsh
                    // parallel yo:
                    //  allocate img_gray
                    //  for yi:
                    //   for y:
                    //    for xo:
                    //     vectorize xi:
                    //      img_gray(x, y)
                    //   for xo:
                    //    vectorize xi:
                    //     img_output(x, y)
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                    ;
                    break;

                case 28:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                    ;
                    break;

                case 29:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    break;

                case 30:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, 2*vector_size).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, 2*vector_size).vectorize(xi)
                    ;
                    break;

                case 31:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, 4*vector_size).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, 4*vector_size).vectorize(xi)
                    ;
                    break;

                case 32:
                    // 0 1 2 3 4 5 6 7 8 9
                    // RoundUp -> 0 1 2 3, 4 5 6 7, 8 9 10 11
                    // GuardWithIf -> 0 1 2 3, 4 5 6 7, 8 9
                    // ShiftInwards -> 0 1 2 3, 4 5 6 7, 6 7 8 9
                    // RoundUp dá erro de execução e a gente deveria usar BoundaryConditions
                    // Não pode usar o RoundUp com acesso aos Buffers de entrada e de saída
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/2, TailStrategy::RoundUp).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                    ;
                    break;

                case 33:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/2, TailStrategy::GuardWithIf).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                    ;
                    break;

                case 34:
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/2, TailStrategy::ShiftInwards).vectorize(xi)
                    ;
                    thrshd.compute_root();
                    img_gray
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/2).vectorize(xi)
                    ;
                    break;

                default:
                    break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, xy{"xy"}, xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"};
        Func img_gray{"img_gray"};
        Func int_gradient_y{"int_gradient_y"}, gradient_y{"gradient_y"};
        Func int_gradient_x{"int_gradient_x"}, gradient_x{"gradient_x"};
        Func gradient_mag{"gradient_mag"};
        Func thrshd{"thrshd"};

    // Exercícios:
    // Fazer o scheduler do DoG do Basics2
};
HALIDE_REGISTER_GENERATOR(HalideGradient, gradient);
