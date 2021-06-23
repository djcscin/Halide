#include "Halide.h"
#include "Convert.hpp"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideYCbCr2RGB : public Generator<HalideYCbCr2RGB> {
    public:
        Input<Func> img_input{"img_input", 3};

        Output<Func> img_output{"img_y2r_output", 3};

        GeneratorParam<Type> img_output_type{"img_output_type", UInt(8)};

        GeneratorParam<bool> define_schedule{"define_schedule", true};

        void generate() {
            img_input_f32(x, y, c) = f32(img_input(x, y, c));

            Expr yy = img_input_f32(x, y, 0);
            Expr cb = img_input_f32(x, y, 1);
            Expr cr = img_input_f32(x, y, 2);

            float k = k_ycbcr(img_input.type());
            Expr r = yy + 1.402f * (cr-k);
            Expr g = yy - 0.344136f * (cb-k) - 0.714136f * (cr-k);
            Expr b = yy + 1.772f * (cb-k);

            Expr out = mux(c, {r, g, b});

            if (img_input.type() != img_output_type) {
                if (img_output_type == UInt(8)) {
                    if (img_input.type() == UInt(16)) {
                        img_output(x, y, c) = u16_to_u8(out);
                    }
                }
            } else {
                img_output(x, y, c) = cast(img_output_type, out);
            }
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
