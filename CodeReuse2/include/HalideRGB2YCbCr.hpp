#include "Halide.h"
#include "Convert.hpp"

using namespace Halide;

class HalideRGB2YCbCr : public Generator<HalideRGB2YCbCr> {
    public:
        Input<Func> img_input{"img_input", 3};

        Output<Func> img_output{"img_r2y_output", 3};

        GeneratorParam<Type> img_output_type{"img_output_type", UInt(16)};
        GeneratorParam<bool> define_schedule{"define_schedule", true};

        void generate() {
            assert(img_input.type() == img_output_type);

            img_input_f32(x, y, c) = f32(img_input(x, y, c));

            Expr r = img_input_f32(x, y, 0);
            Expr g = img_input_f32(x, y, 1);
            Expr b = img_input_f32(x, y, 2);

            float k = k_ycbcr(img_input.type());
            Expr yy = 0.299f*r + 0.587f*g + 0.114f*b;
            Expr cb = k - 0.168736f*r - 0.331264f*g + 0.5f*b;
            Expr cr = k + 0.5f*r - 0.418688f*g - 0.081312f*b;

            img_output(x, y, c) = cast(img_output_type, mux(c, {yy, cb, cr}));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if (define_schedule) {
                int vector_size = get_target().natural_vector_size<float>();

                img_output
                    .compute_root()
                    .bound(c, 0, 3)
                    .unroll(c)
                    .parallel(y)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .reorder(xi, c, xo, y)
                ;
                img_input_f32
                    .compute_at(img_output, y)
                    .vectorize(x, vector_size)
                ;
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var xo{"xo"}, xi{"xi"};
        Func img_input_f32{"img_input_f32"};
};
