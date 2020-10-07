/* run all schedulers using
for i in $(seq 0 17); do make SCHEDULER=$i; done
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
                switch (scheduler)
                {
                case 0:
                    // scheduler padrão é todas as funções compute_inline e output compute_root
                    // int_gradient_y(x, y) = img_gray(x, y) - img_gray(x, y + 2);
                    // gradient_y(x, y) = int_gradient_y(x, y) + 2*int_gradient_y(x + 1, y) + int_gradient_y(x + 2, y);
                    // gradient_y.compute_root();
                    // int_gradient_y.compute_inline(); <-- default, não preciso colocar
                    // gradient_y(x, y) = img_gray(x, y) - img_gray(x, y + 2)
                        // + 2*img_gray(x + 1, y) - 2*img_gray(x + 1, y + 2)
                        // + img_gray(x + 2, y) - img_gray(x + 2, y + 2)

                    // for y:
                    //   for x:
                    //      img_output(x, y)

                    // vantagem: calcula uma função, ele usa o resultado diretamente -> ótima localidade de referência
                    // desvantagem: redundância de computação de algumas funções

                    break;

                case 1:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // for y:
                    //  for x:
                    //    gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    gradient_x(x, y)
                    // for y:
                    //  for x:
                    //    gradient_mag(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)

                    // vantagem: não tem redudância na computação
                    // desvantagem: péssima localidade de referência, só vou usar um valor de uma função depois de ter calculado todos os valores

                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_root();
                    gradient_x.compute_root();
                    gradient_y.compute_root();
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 2:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // for y:
                    //  for x:
                    //    gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_root();
                    gradient_y.compute_root();
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 3:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // for y:
                    //  for x:
                    //    gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //   for y:
                    //     for x:
                    //      gradient_mag(x, y)
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_at(img_output, x);
                    gradient_x.compute_root();
                    gradient_y.compute_root();
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 4:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // for y:
                    //  for x:
                    //    gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //      gradient_mag(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_at(img_output, y);
                    gradient_x.compute_root();
                    gradient_y.compute_root();
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 5:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 6:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //   for y:
                    //    for x:
                    //     gradient_y(x, y)
                    //   for y:
                    //    for x:
                    //     gradient_x(x, y)
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_at(img_output, x);
                    gradient_y.compute_at(img_output, x);
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 7:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_y(x, y)
                    // for y:
                    //  for x:
                    //    int_gradient_x(x, y)
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //     gradient_y(x, y)
                    //  for y:
                    //   for x:
                    //     gradient_x(x, y)
                    //  for x:
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_at(img_output, y);
                    gradient_y.compute_at(img_output, y);
                    int_gradient_x.compute_root();
                    int_gradient_y.compute_root();
                    img_gray.compute_root();
                    break;

                case 8:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_root();
                    break;

                case 9:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //   for y:
                    //    for x:
                    //     int_gradient_y(x, y)
                    //   for y:
                    //    for x:
                    //     int_gradient_x(x, y)
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, x);
                    int_gradient_y.compute_at(img_output, x);
                    img_gray.compute_root();
                    break;

                case 10:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //     int_gradient_y(x, y)
                    //  for y:
                    //   for x:
                    //     int_gradient_x(x, y)
                    //  for x:
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y);
                    img_gray.compute_root();
                    break;

                case 11:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // thrsh
                    // for y:
                    //  for x:
                    //   for y:
                    //    for x:
                    //     int_gradient_x(x, y)
                    //     int_gradient_y(x, y)
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, x);
                    int_gradient_y.compute_at(img_output, x).compute_with(int_gradient_x, x);
                    img_gray.compute_root();
                    break;

                case 12:
                    // for y:
                    //  for x:
                    //    img_gray(x, y)
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //     int_gradient_x(x, y)
                    //     int_gradient_y(x, y)
                    //  for x:
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x);
                    img_gray.compute_root();
                    break;

                case 13:
                    // thrsh
                    // for y:
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_inline();
                    break;

                case 14:
                    // thrsh
                    // for y:
                    //  for x:
                    //   for y:
                    //    for x:
                    //     img_gray(x, y)
                    //   img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_at(img_output, x);
                    break;

                case 15:
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //    img_gray(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_inline();
                    int_gradient_y.compute_inline();
                    img_gray.compute_at(img_output, y);
                    break;

                case 16:
                    // thrsh
                    // for y:
                    //  for y:
                    //   for x:
                    //    int_gradient_x(x, y)
                    //    int_gradient_y(x, y)
                    //  for x:
                    //    img_output(x, y)
                    img_output.compute_root();
                    thrshd.compute_root();
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x);
                    img_gray.compute_inline();
                    break;

                case 17:
                    // thrsh
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
                    gradient_mag.compute_inline();
                    gradient_x.compute_inline();
                    gradient_y.compute_inline();
                    int_gradient_x.compute_at(img_output, y);
                    int_gradient_y.compute_at(img_output, y).compute_with(int_gradient_x, x);
                    img_gray.compute_at(img_output, y);
                    break;

                default:
                    break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"};
        Func img_gray{"img_gray"};
        Func int_gradient_y{"int_gradient_y"}, gradient_y{"gradient_y"};
        Func int_gradient_x{"int_gradient_x"}, gradient_x{"gradient_x"};
        Func gradient_mag{"gradient_mag"};
        Func thrshd{"thrshd"};
};
HALIDE_REGISTER_GENERATOR(HalideGradient, gradient);
