/*
Implemente schedulers para esse código Halide
Faça schedulers que eliminem os três selects usando unroll e specialize
Onde devemos usar unroll e specialize?
Compare os tempos de execução com schedulers sem usar o specialize e diga o porquê a melhora não foi significativa.
RESPOSTA: Devemos usar unroll e specialize para eliminar instruções de select. Nesse código específico, o unroll é usado nas
funções output e yuv_in enquanto que specialize na função gaussian. Caso a função gaussian for inline, o specialize passa a
ser usado na função onde ela é substituída, no caso, output. A melhora com o specialize não foi significativa, especialmente
quando usada no gaussian, pois o select só tem duas opções de resultado com operações simples e a CPU é capaz de prever o
resultado da operação sem problemas, uma vez que a condição não muda para cada um dos pixels. Além disso, o Halide de certa
forma consegue eliminar a execução do gaussian2 sem a necessidade do specialize.

Observe que yuv_in tem domínios diferentes para gaussian1_int e output
Faça schedulers que calculem yuv_in de formas diferentes para cada uma dessas funções
Por que ao levar em consideração essas diferenças houve uma grande diminuição do tempo de execução?
RESPOSTA: Os canais de crominância são usados no filtro gaussiano e precisam que para cada linha de gaussian1_int 11 linhas
de yuv_in serem computadas, enquanto que o valor do canal de luminância é usado apenas para os cálculos dos canais de mesma
coordenada
*/
/* run all schedulers using
for i in $(seq 1 12); do echo SCHEDULER $i; make clean && make SCHEDULER=$i; done
or
for i in $(seq 1 12); do echo SCHEDULER $i; make clean && make test_desktop DESKTOP=true SCHEDULER=$i; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideGenerator : public Halide::Generator<HalideGenerator> {
private:
    Var x{"x"}, y{"y"}, c{"c"};
    Var xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"}, yc{"yc"};
    Var i{"i"};
    RDom r_gaussian;

    Func w_gaussian{"w_gaussian"}, w_gaussian_norm{"w_gaussian_norm"}, sum_w_gaussian{"sum_w_gaussian"};
    Func input_f{"input_f"}, yuv_in{"yuv_in"}, gaussian{"gaussian"};
    Func gaussian1_int{"gaussian1_int"}, gaussian1{"gaussian1"};
    Func gaussian2_int{"gaussian2_int"}, gaussian2{"gaussian2"};

public:
    GeneratorParam<int> scheduler{"scheduler", 0};
    GeneratorParam<int> parallel_size{"parallel_size", 128};

    Input<Buffer<uint8_t>> input{"input", 3};
    Input<float> sigma{"sigma"};
    Input<bool> run_twice{"run_twice"};
    Output<Buffer<uint8_t>> output{"output", 3};

    void filter(const Func & in, const Func & w, RDom r, Func & f_y, Func & out) {
        f_y(x, y, c) = 0.0f;
        f_y(x, y, c) += in(x, y + r, c) * w(r);

        out(x, y, c) = 0.0f;
        out(x, y, c) += f_y(x + r, y, c) * w(r);
    }

    void generate() {
        r_gaussian = RDom(-5, 11, "r_gaussian");
        w_gaussian(i) = exp( (i*i) / (-2.0f*(sigma*sigma)) );
        sum_w_gaussian() = 0.0f;
        sum_w_gaussian() += w_gaussian(r_gaussian);
        w_gaussian_norm(i) = w_gaussian(i) / sum_w_gaussian();

        input_f(x, y, c) = f32(BoundaryConditions::mirror_image(input)(x, y, c));

        Expr _r = input_f(x, y, 0);
        Expr _g = input_f(x, y, 1);
        Expr _b = input_f(x, y, 2);
        yuv_in(x, y, c) = select(
            c == 0,  0.299000f * _r + 0.587000f * _g + 0.114000f * _b,
            c == 1, -0.168935f * _r - 0.331655f * _g + 0.500590f * _b,
                     0.499813f * _r - 0.418531f * _g - 0.081282f * _b
        );

        filter(yuv_in, w_gaussian_norm, r_gaussian, gaussian1_int, gaussian1);
        filter(gaussian1, w_gaussian_norm, r_gaussian, gaussian2_int, gaussian2);

        gaussian(x, y, c) = select(
            run_twice, gaussian2(x, y, c),
                       gaussian1(x, y, c)
        );

        Expr _y = yuv_in(x, y, 0);
        Expr _u = gaussian(x, y, 1);
        Expr _v = gaussian(x, y, 2);
        output(x, y, c) = select(
            c == 0, u8_sat(_y               + 1.403f * _v),
            c == 1, u8_sat(_y - 0.344f * _u - 0.714f * _v),
                    u8_sat(_y + 1.770f * _u              )
        );
    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
        sigma.set_estimate(1.5);
        output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});

        if (!auto_schedule) {
            w_gaussian.compute_root();
            sum_w_gaussian.compute_root();
            w_gaussian_norm.compute_root();

            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3)
                .unroll(c)
                .reorder(xi, c, xo, yi, yo)
            ;
            gaussian1
                .compute_at(output, yi).store_at(output, yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian1.update()
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian1_int
                .compute_at(gaussian1, y)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian1_int.update()
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian2
                .compute_at(output, yi)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian2.update()
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian2_int
                .compute_at(gaussian2, y)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            gaussian2_int.update()
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;

            switch (scheduler) {
            case 1:
            case 5:
            case 9:
                gaussian.specialize(run_twice);
            case 2:
            case 6:
            case 10:
                gaussian
                    .compute_at(output, yi)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
                break;

            case 3:
            case 7:
            case 11:
                output.specialize(run_twice);
            case 4:
            case 8:
            case 12:
                break;

            default:
                break;
            }

            switch (scheduler){
            case 1:
            case 2:
            case 3:
            case 4:
                yuv_in
                    .compute_at(output, yi).store_at(output, yo)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .unroll(c)
                    .reorder(xi, c, xo, y)
                ;
                break;

            case 5:
            case 6:
            case 7:
            case 8:
                yuv_in.in(output)
                    .compute_at(output, yi).store_at(output, yo)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .unroll(c)
                    .reorder(xi, c, xo, y)
                ;
            case 9:
            case 10:
            case 11:
            case 12:
                yuv_in.in(gaussian1_int)
                    .compute_at(output, yi).store_at(output, yo)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .unroll(c)
                    .reorder(xi, c, xo, y)
                ;
                break;

            default:
                break;
            }
        }
    }
};
HALIDE_REGISTER_GENERATOR(HalideGenerator, halide_func)
