/* run all versions using
for lut in false true; do make LUT=$lut; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

//img_output(_) = a1 * img_input1(_)^gamma1 + a2 * img_input2(_)^gamma2
class HalideGenGamma : public Generator<HalideGenGamma> {
    public:
        Input<Buffer<uint8_t>> img_input1{"img_input1", 3};
        Input<float> a1{"a1"};
        Input<float> gamma1{"gamma1"};

        Input<Buffer<uint8_t>> img_input2{"img_input2", 3};
        Input<float> a2{"a2"};
        Input<float> gamma2{"gamma2"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<bool> lut{"lut", true};

        void generate() {
            img_gamma1 = gamma_v0(img_input1, a1, gamma1, lut_gamma1);
            img_gamma2 = gamma_v0(img_input2, a2, gamma2, lut_gamma2);

            sum(x, y, c) = img_gamma1(x, y, c) + img_gamma2(x, y, c);

            img_output(x, y, c) = u8_sat(sum(x, y, c));
        }

        void schedule() {
            if (auto_schedule) {
                img_input1.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                a1.set_estimate(0.5);
                gamma1.set_estimate(2.2);

                img_input2.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                a2.set_estimate(0.5);
                gamma2.set_estimate(2.2);

                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size<float>();

                if(lut) {
                    lut_gamma1
                        .compute_root()
                        .split(value, xo, xi, vector_size).vectorize(xi)
                    ;
                    lut_gamma2
                        .compute_root()
                        .split(value, xo, xi, vector_size).vectorize(xi)
                    ;
                }
                img_output
                    .compute_root()
                    .fuse(y, c, yc).parallel(yc)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                ;
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xi{"xi"}, xo{"xo"}, yc{"yc"};
        Var value{"value"};

        Func input1_float{"input1_float"}, img_gamma1{"img_gamma1"}, lut_gamma1{"lut_gamma1"};
        Func input2_float{"input2_float"}, img_gamma2{"img_gamma2"}, lut_gamma2{"lut_gamma2"};
        Func sum{"sum"};

        Func gamma_v0(Func input, Expr a, Expr gamma, Func & lut_gamma) {
            Func output;

            if(lut) {
                Expr input_float = f32(value);
                Expr input_unit = input_float/255.0f;
                Expr output_unit = a * pow(input_unit, gamma);
                lut_gamma(value) = output_unit * 255.0f;

                output(x, y, c) = lut_gamma(input(x, y, c));
            } else {
                Expr input_float = f32(input(x, y, c));
                Expr input_unit = input_float/255.0f;
                Expr output_unit = a * pow(input_unit, gamma);
                output(x, y, c) = output_unit * 255.0f;
            }

            return output;
        }
};
HALIDE_REGISTER_GENERATOR(HalideGenGamma, gamma);
