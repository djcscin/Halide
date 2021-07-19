/*
Implemente schedulers para esse código Halide
Faça schedulers que eliminem os três selects usando unroll e specialize
Onde devemos usar unroll e specialize?
Compare os tempos de execução com schedulers sem usar o specialize e diga o porquê a melhora não foi significativa.
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

        //Responda aqui

    }
};
HALIDE_REGISTER_GENERATOR(HalideGenerator, halide_func)
