#include "Halide.h"

using namespace Halide; // For using Var, Func, Expr, Generator, ... instead of
                        // Halide::Var, Halide::Func, Halide::Expr, Halide::Generator, ...
using namespace Halide::ConciseCasts; // For using {u,i}{8,16,32,64}, {u,i}{8,16,32,64}_sat, f{32,64}

//img_output(_) = a1 * img_input1(_)^gamma1 + a2 * img_input2(_)^gamma2
class HalideGenGamma : public Generator<HalideGenGamma> {
    public:
        // entradas - Input<>
        // entre os chaves temos o nome para debug da variável e o número de dimensões
        Input<Buffer<uint8_t>> img_input1{"img_input1", 3};
        Input<float> a1{"a1"};
        Input<float> gamma1{"gamma1"};

        Input<Buffer<uint8_t>> img_input2{"img_input2", 3};
        Input<float> a2{"a2"};
        Input<float> gamma2{"gamma2"};

        // saídas - Output<>
        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        // variáveis de tempo de compilação
        // conseguimos usar na metaprogramação com if, for, while
        // vem no comando de execução do binário do generator
        GeneratorParam<uint32_t> version{"version", 0};

        // duas funções: o generate com o algoritmo e
        // o schedule com o schedule ou as definições para auto_schedule
        void generate() {
            if (version == 0) {
                // img_output(_) =
                //    a1 * img_input1(_)^gamma1 +
                //    a2 * img_input2(_)^gamma2
                // Para fazer reuso, usamos as funções da alta ordem
                img_gamma1 = gamma_v0(img_input1, a1, gamma1, input1_float);
                // Poderia tá assim
                // img_gamma1 (x, y, c) = gamma_v0(img_input1, a1, gamma1, input1_float)(x, y, c);
                // mas não tem necessidade nesse caso e não vamos fazer isso
                // se fizer isso, img_gamma1 e output de gamma_v0 seriam funções diferentes
                // como queremos ter acesso a todas as funções, output na verdade deveria ser
                // uma função intermediária
                img_gamma2 = gamma_v0(img_input2, a2, gamma2, input2_float);

                sum(x, y, c) = img_gamma1(x, y, c) + img_gamma2(x, y, c);

                img_output(x, y, c) = u8_sat(sum(x, y, c));
            } else if (version == 1) {
                Func a1_f {"a1_f"}, gamma1_f{"gamma1_f"};
                Func a2_f {"a2_f"}, gamma2_f{"gamma2_f"};

                // Como é um escalar, não temos variáveis na função
                a1_f() = a1;
                gamma1_f() = gamma1;
                img_gamma1 = gamma_v1(img_input1, a1_f, gamma1_f, input1_float);

                a2_f() = a2;
                gamma2_f() = gamma2;
                img_gamma2 = gamma_v1(img_input2, a2_f, gamma2_f, input2_float);

                sum(x, y, c) = img_gamma1(x, y, c) + img_gamma2(x, y, c);

                img_output(x, y, c) = u8_sat(sum(x, y, c));
            }
        }

        void schedule() {
            if (auto_schedule) { // O schedule é feito automaticamente pelo compilador
                // Para isso, você precisa estimar as dimensões dos Buffers e os valores dos escalares
                // Usa o set_estimates para os Buffers
                // e o set_estimate para os escalares
                // O argumento do set_estimates é uma lista
                // {min, extent} da dimensão
                img_input1.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                a1.set_estimate(0.5);
                gamma1.set_estimate(2.2);

                img_input2.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                a2.set_estimate(0.5);
                gamma2.set_estimate(2.2);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            }
        }

    private:
        // Declarar todas as variáveis como private
        Var x{"x"}, y{"y"}, c{"c"};

        // Declarar todas as funções intermediárias
        Func input1_float{"input1_float"}, img_gamma1{"img_gamma1"};
        Func input2_float{"input2_float"}, img_gamma2{"img_gamma2"};
        Func sum{"sum"};

        // Isso é para usar as variáveis e as funções intermediárias na função schedule

        // Temos dois parâmetros passados por referência que são as funções intermediárias
        Func gamma_v0(Func input, Expr a, Expr gamma, Func & input_float) {
            Func output;

            input_float(x, y, c) = f32(input(x, y, c));

            Expr input_unit = input_float(x, y, c)/255.0f;
            Expr output_unit = a * pow(input_unit, gamma);
            output(x, y, c) = output_unit * 255.0f;

            return output;
        }

        Func gamma_v1(Func input, Func a, Func gamma, Func & input_float) {
            Func output;

            input_float(x, y, c) = f32(input(x, y, c));

            Expr input_unit = input_float(x, y, c)/255.0f;
            Expr output_unit = a() * pow(input_unit, gamma());
            output(x, y, c) = output_unit * 255.0f;

            return output;
        }

        // Exercícios:
        // Fazer uma versão version=2 que chama gamma_v2 com
        // a, gamma, input_float, input_unit e output_unit as Expr

        // Fazer uma versão version=3 que chama gamma_v3 com
        // a, gamma, input_float, input_unit e output_unit as Func

};
HALIDE_REGISTER_GENERATOR(HalideGenGamma, gamma);
