#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

enum Filter {
    LAPLACIAN_0 = 0,
    LAPLACIAN_1,
    LAPLACIAN_2,
    UNSHARP_GAUSSIAN,
    DoG,
    LoG,
    GAUSSIAN, //apenas para testar o sharpening com uma imagem borrada
};

// img_input -(filtro passa alta)-> unsharp
// img_input + strength * unsharp

// filtro passa alta:
// Laplaciana
// img_input - Gaussiana da img_input
// DoG = Gaussiana da img_input com sigma s - Gaussiana da img_input com um sigma maior que s
// LoG = Laplaciana no resultado da Gaussiana da img_input

class HalideSharp : public Generator<HalideSharp> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<float> strength{"strength"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        // GeneratoParam você passa o nome de debug e o valor default
        // lista {string do enum, enum}
        GeneratorParam<enum Filter> filter{"filter", LAPLACIAN_1,
            {
                {"laplacian0", LAPLACIAN_0},
                {"laplacian1", LAPLACIAN_1},
                {"laplacian2", LAPLACIAN_2},
                {"unsharp_gauss", UNSHARP_GAUSSIAN},
                {"dog", DoG},
                {"log", LoG},
                {"gaussian", GAUSSIAN}
            }
        };

        void generate() {
            // https://halide-lang.org/docs/_boundary_conditions_8h.html
            // https://open.gl/textures
            // GL_CLAMP_TO_BORDER -> constant_exterior
            // GL_CLAMP_TO_EDGE -> repeat_edge
            // GL_REPEAT -> repeat_image
            // GL_MIRRORED_REPEAT -> mirror_image
            // 0 1 2 -mirror_image-> 0 1 2 2 1 0
            // 0 1 2 -mirror_interior-> 0 1 2 1 0
            input_bound(x, y, c) = i32(BoundaryConditions::mirror_interior(img_input)(x, y, c));
            // além do mirror_interior para tratar as fronteiras da imagem,
            // foi feita o cast de u8 pra i32 pra evitar overflow
            // Func input_boundc = BoundaryConditions::mirror_interior(img_input);
            // input_bound(x, y, c) = i32(input_boundc(x, y, c));

            switch (filter)
            {
            case LAPLACIAN_0:
                output = laplacian_0(input_bound, int_x, int_y);
                break;

            case LAPLACIAN_1:
                output = laplacian_1(input_bound, int_x, int_y);
                break;

            case LAPLACIAN_2:
                output = laplacian_2(input_bound, int_x, int_y);
                break;

            case UNSHARP_GAUSSIAN:
                output = unsharp_gaussian(input_bound, int_x, int_y);
                break;

            case DoG:
                output = dog(input_bound, int_x, int_y, int2_x, int2_y);
                break;

            case GAUSSIAN:
                output = gaussian5(input_bound, int_x, int_y);

            default:
                break;
            }

            img_output(x, y, c) = u8_sat(output(x, y, c));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                strength.set_estimate(0.2);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Func input_bound{"input_bound"};
        Func int_x{"int_x"}, int_y{"int_y"};
        Func int2_x{"int2_x"}, int2_y{"int2_y"};
        Func output{"output"};

        //  0 -1  0   -1  0 -1   -1 -1 -1
        // -1  4 -1 ,  0  4  0 , -1  8 -1
        //  0 -1  0   -1  0 -1   -1 -1 -1

        // d²f(x,y)/dx² + d²f(x,y)/dy²
        // df(x,y)/dx = f(x+1, y) - f(x, y)
        // d²f(x,y)/dx² = f(x+1, y) - 2*f(x, y) + f(x-1, y)

        //  0 -1  0    0  0  0    0 -1  0
        // -1  4 -1 = -1  2 -1 +  0  2  0
        //  0 -1  0    0  0  0    0 -1  0
        Func laplacian_0(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) = - input(x, y-1, c) + 2 * input(x, y, c) - input(x, y+1, c);
            int_x(x, y, c) = - input(x-1, y, c) + 2 * input(x, y, c) - input(x+1, y, c);

            Expr laplacian = int_x(x, y, c) + int_y(x, y, c);
            output(x, y, c) = input(x, y, c) + strength * laplacian;

            return output;
        }

        //  1             1 -2  1    -1              1 -2  1
        // -2 * 1 -2 1 = -2  4 -2 ou  2 * -1 2 -1 = -2  4 -2
        //  1             1 -2  1    -1              1 -2  1
        Func laplacian_1(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) = input(x, y-1, c) - 2 * input(x, y, c) + input(x, y+1, c);
            int_x(x, y, c) = int_y(x-1, y, c) - 2 * int_y(x, y, c) + int_y(x+1, y, c);

            Expr laplacian = int_x(x, y, c);
            output(x, y, c) = input(x, y, c) + strength * laplacian;

            return output;
        }

        // -1 -1 -1   0 0 0   1 1 1
        // -1  8 -1 = 0 9 0 - 1 1 1
        // -1 -1 -1   0 0 0   1 1 1
        // 1           1 1 1
        // 1 * 1 1 1 = 1 1 1
        // 1           1 1 1
        Func laplacian_2(Func input, Func & int_x, Func & int_y) {
            Func output;

            //blur
            int_y(x, y, c) = input(x, y-1, c) + input(x, y, c) + input(x, y+1, c);
            int_x(x, y, c) = int_y(x-1, y, c) + int_y(x, y, c) + int_y(x+1, y, c);

            // Expr laplacian = 9 * input(x, y, c) - int_x(x, y, c));
            // output(x, y, c) = input(x, y, c) + strength * laplacian;
            // output(x, y, c) = input(x, y, c) + strength * (9 * input(x, y, c) - int_x(x, y, c));
            output(x, y, c) = (1 + strength * 9) * input(x, y, c) - strength * int_x(x, y, c);

            return output;
        }

        //       1            sum = 2^0
        //      1 1           sum = 2^1
        //     1 2 1          sum = 2^2
        //    1 3 3 1         sum = 2^3
        //   1 4 6 4 1        sum = 2^4
        //  1 5 10 10 5 1     sum = 2^5
        // 1 6 15 20 15 6 1   sum = 2^6
        // Dividir por 2^n é menos custoso, pois é convertido num shift
        Func unsharp_gaussian(Func input, Func & int_x, Func & int_y) {
            Func output;

            Expr interm = gaussian7(input, int_x, int_y)(x, y, c);

            // Expr unsharp = input(x, y, c) - interm;
            // output(x, y, c) = input(x, y, c) + strength * unsharp;
            // output(x, y, c) = input(x, y, c) + strength * (input(x, y, c) - interm);
            output(x, y, c) = (1 + strength) * input(x, y, c) - strength * interm;

            return output;
        }

        Func gaussian7(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) =
                       input(x, y-3, c)
                +  6 * input(x, y-2, c)
                + 15 * input(x, y-1, c)
                + 20 * input(x, y, c)
                + 15 * input(x, y+1, c)
                +  6 * input(x, y+2, c)
                +      input(x, y+3, c);
            int_x(x, y, c) =
                       int_y(x-3, y, c)
                +  6 * int_y(x-2, y, c)
                + 15 * int_y(x-1, y, c)
                + 20 * int_y(x,   y, c)
                + 15 * int_y(x+1, y, c)
                +  6 * int_y(x+2, y, c)
                +      int_y(x+3, y, c);

            output(x, y, c) = int_x(x, y, c) / 4096; // 4096 -> 64*64

            return output;
        }

        Func gaussian5(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) =
                       input(x, y-2, c)
                +  4 * input(x, y-1, c)
                +  6 * input(x, y, c)
                +  4 * input(x, y+1, c)
                +      input(x, y+2, c);
            int_x(x, y, c) =
                       int_y(x-2, y, c)
                +  4 * int_y(x-1, y, c)
                +  6 * int_y(x,   y, c)
                +  4 * int_y(x+1, y, c)
                +      int_y(x+2, y, c);

            output(x, y, c) = int_x(x, y, c) / 256; // 256 -> 16*16

            return output;
        }

        Func dog(Func input, Func & int_x, Func & int_y, Func & int2_x, Func & int2_y) {
            Func output;

            Func gaussian1 = gaussian5(input, int_x, int_y);
            Func gaussian2 = gaussian7(input, int2_x, int2_y);

            Expr d_o_g = gaussian1(x, y, c) - gaussian2(x, y, c);
            output(x, y, c) = input(x, y, c) + strength * d_o_g;

            return output;
        }

        // Exercício:
        // Implementar o LoG (Evite replicação de código)

};
HALIDE_REGISTER_GENERATOR(HalideSharp, sharp);
