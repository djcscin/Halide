#include "Median.hpp"

class HalideMedian_Y : public Generator<HalideMedian_Y>, public Median {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        GeneratorParam<bool> run_only_on_luma{"run_only_on_luma", "true"};

        void generate() {
            if(run_only_on_luma) {
                img_input_f32(x, y, c) = f32(img_input(x, y, c));

                YCbCr_input = rgb_to_YCbCr(img_input_f32);

                Y(x, y) = YCbCr_input(x, y, 0);

                evaluate_median(Y, img_input.width(), img_input.height());

                YCbCr_output(x, y, c) = select(c == 0, median_x(x, y), YCbCr_input(x, y, c));

                img_output_f32 = YCbCr_to_rgb(YCbCr_output);

                img_output(x, y, c) = u8_sat(img_output_f32(x, y, c));
            } else {
                evaluate_median(img_input, img_input.width(), img_input.height());

                img_output = median_x;
            }
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else {
                int vector_size = get_target().natural_vector_size(UInt(8));
                if(run_only_on_luma) {
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                        .reorder(xi, c, xo, yi, yo).unroll(c).bound(c, 0, 3)
                    ;
                    img_input_bound
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(x, xo, xi, vector_size/4).vectorize(xi)
                    ;
                } else {
                    Var c = img_output.args()[2];
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).fuse(yo, c, yc).parallel(yc)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_input_bound
                        .compute_at(img_output, yi)
                        .store_at(img_output, yc)
                        .split(_0, xo, xi, vector_size).vectorize(xi)
                    ;
                }
            }
        }

    private:
        Var c{"c"}, xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"}, yc{"yc"};
        Func img_input_f32{"img_input_f32"}, img_output_f32{"img_output_f32"};
        Func YCbCr_input{"YCbCr_input"}, Y{"Y"}, YCbCr_output{"YCbCr_output"};

        Func rgb_to_YCbCr(Func input) {
            Func output;

            Expr r = input(x, y, 0);
            Expr g = input(x, y, 1);
            Expr b = input(x, y, 2);

            Expr yy = 0.299f*r + 0.587f*g + 0.114f*b;
            Expr cb = 128.f - 0.168736f*r - 0.331264f*g + 0.5f*b;
            Expr cr = 128.f + 0.5f*r - 0.418688f*g - 0.081312f*b;

            output(x, y, c) = mux(c, {yy, cb, cr});

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

            output(x, y, c) = mux(c, {r, g, b});

            return output;
        }

};
HALIDE_REGISTER_GENERATOR(HalideMedian_Y, median_y);
