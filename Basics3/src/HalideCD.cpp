#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideCD : public Generator<HalideCD> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<bool> update{"update", false}; //update ou select
        GeneratorParam<bool> meta{"meta", false}; //computar a gaussiana em tempo de compilação ou execução
        GeneratorParam<float> sigma{"sigma", 1.5f}; //sigma da gaussiana

        void generate() {
            // compilação -> static int, static float, ...
            // execução -> Expr, Func
            static const int gaussian_width = int(3*sigma + 0.5f);
            if(meta) { // tempo de compilação
                static float * weights = new float[gaussian_width + 1];
                for(int i = 0; i <= gaussian_width; ++i) {
                    weights[i] = exp(-(i*i)/(2.f*sigma*sigma));
                }
                static float sum = weights[0];
                for(int i = 1; i <= gaussian_width; ++i) {
                    sum += 2*weights[i]; //weights[i] + weights[-i]
                }
                for(int i = 0; i <= gaussian_width; ++i) {
                    weights[i] /= sum; //sum(weights)=1
                }
                w(x) = Buffer<float>(weights, gaussian_width + 1)(x);
            } else { // tempo de execução
                Func weights;
                weights(x) = exp(-(x*x)/(2.f*sigma*sigma));
                Expr sum = weights(0);
                for(int i = 1; i <= gaussian_width; ++i) {
                    sum += 2*weights(i);
                } // se gaussian_width for 3 sum = weigths(0) + 2*weight(1) + 2*weights(2) + 2*weights(3)
                w(x) = weights(x)/sum; //w(x) = weights(x)/(weigths(0) + 2*weight(1) + 2*weights(2) + 2*weights(3))
            }

            input_f32(x, y, c) = f32(img_input(x, y, c));

            yCbCr_input = rgb_to_YCbCr(input_f32);

            yCbCr_bound = BoundaryConditions::mirror_interior(yCbCr_input, {{0, img_input.width()}, {0, img_input.height()}});

            yCbCr_blurred = gaussian(yCbCr_bound, w, gaussian_width, yCbCr_blurred_intermediate);

            if(update) {
                yCbCr_output(x, y, c) = undef(Float(32));
                yCbCr_output(x, y, 0) = yCbCr_input(x, y, 0);
                yCbCr_output(x, y, 1) = yCbCr_blurred(x, y, 1);
                yCbCr_output(x, y, 2) = yCbCr_blurred(x, y, 2);
            } else {
                yCbCr_output(x, y, c) = select(c == 0, yCbCr_input(x, y, 0), yCbCr_blurred(x, y, c));
            }

            output_f32 = YCbCr_to_rgb(yCbCr_output);

            img_output(x, y, c) = u8_sat(output_f32(x, y, c));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        static int gaussian_width;
        Func w{"w"}, input_f32{"input_f32"}, output_f32{"output_f32"};
        Func yCbCr_input{"yCbCr_input"}, yCbCr_bound{"yCbCr_bound"};
        Func yCbCr_blurred_intermediate{"yCbCr_blurred_intermediate"};
        Func yCbCr_blurred{"yCbCr_blurred"}, yCbCr_output{"yCbCr_output"};

        Func rgb_to_YCbCr(Func input) {
            Func output;

            Expr r = input(x, y, 0);
            Expr g = input(x, y, 1);
            Expr b = input(x, y, 2);

            Expr yy = 0.299f*r + 0.587f*g + 0.114f*b;
            Expr cb = 128.f - 0.168736f*r - 0.331264f*g + 0.5f*b;
            Expr cr = 128.f + 0.5f*r - 0.418688f*g - 0.081312f*b;

            if(update) {
                // undef serve pra dizer os tipos sem dizer os valores
                // os valores tem que ser ditos nos updates seguintes
                output(x, y, c) = undef(Float(32)); //update(0) ou update() - tem que usar todas as variáveis
                output(x, y, 0) = yy; //update(1)
                output(x, y, 1) = cb; //update(2)
                output(x, y, 2) = cr; //update(3)
                // não existe uma forma de substituir uma expressão na outra
                // os schedules são feitos para cada update, mas tem o mesmo momento quando a função é computada
            } else{
                output(x, y, c) = mux(c, {yy, cb, cr});
                // mux(c, {a, b, c}) -> select(c == 0, a, c == 1, b, c)
                // executa todos os valores das expressões
                // é uma boa estratégia eliminar esse select
            }

            return output;
        }

        Func YCbCr_to_rgb(Func input) {
            Func output;

            Expr yy = input(x, y, 0);
            Expr cb = input(x, y, 1);
            Expr cr = input(x, y, 2);

            Expr r = yy + 1.402f * (cr-128.f);
            Expr g = yy - 0.344136f * (cb-128.f) - 0.714136f * (cr-128.f);
            Expr b = yy + 1.772f * (cb-128.f);

            if(update) {
                output(x, y, c) = undef(Float(32));
                output(x, y, 0) = r;
                output(x, y, 1) = g;
                output(x, y, 2) = b;
            } else{
                output(x, y, c) = mux(c, {r, g, b});
            }

            return output;
        }

        Func gaussian(Func input, Func w, int gaussian_width, Func & intermediate) {
            Func output;

            Expr g_y = w(0)*input(x, y, c);
            for(int i = 1; i <= gaussian_width; ++i) {
                g_y += w(i)*(input(x, y-i, c) + input(x, y+i, c));
            } // se gaussian_width for 3
                            // g_y = w(0)*input(x, y, c)
                            //     + w(1)*(input(x, y-1, c) + input(x, y+1, c)) // w(-1) = w(1)
                            //     + w(2)*(input(x, y-2, c) + input(x, y+2, c)) // w(-2) = w(2)
                            //     + w(3)*(input(x, y-3, c) + input(x, y+2, c)) // w(-3) = w(3)
            intermediate(x, y, c) = g_y;

            Expr g_x = w(0)*intermediate(x, y, c);
            for(int i = 1; i <= gaussian_width; ++i) {
                g_x += w(i)*(intermediate(x-i, y, c) + intermediate(x+i, y, c));
            }
            output(x, y, c) = g_x;

            return output;
        }

};
HALIDE_REGISTER_GENERATOR(HalideCD, cd);
