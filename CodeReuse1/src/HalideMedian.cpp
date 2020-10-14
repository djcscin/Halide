#include "Median.hpp"

class HalideMedian : public Generator<HalideMedian>, public Median {
    public:
        Input<Buffer<>> img_input{"img_input"};
        Output<Buffer<>> img_output{"img_output"};

        void generate() {
            evaluate_median(img_input, img_input.width(), img_input.height());

            img_output = median_x;
        }

        void schedule() {
            if (auto_schedule) {
                if(img_input.dimensions() == 2) {
                    img_input.set_estimates({{0, 4000}, {0, 3000}});
                    img_output.set_estimates({{0, 4000}, {0, 3000}});
                } else if(img_input.dimensions() == 3) {
                    img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                    img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                }
            } else {
                int vector_size = get_target().natural_vector_size(img_input.type());
                if(img_input.dimensions() == 2) {
                    img_output
                        .compute_root()
                        .split(y, yo, yi, 64).parallel(yo)
                        .split(x, xo, xi, vector_size).vectorize(xi)
                    ;
                    img_input_bound
                        .compute_at(img_output, yi)
                        .store_at(img_output, yo)
                        .split(_0, xo, xi, vector_size).vectorize(xi)
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
        Var xo{"xo"}, xi{"xi"}, yo{"yo"}, yi{"yi"}, yc{"yc"};
};
HALIDE_REGISTER_GENERATOR(HalideMedian, median);
