#include "Halide.h"

using namespace Halide;

class HalideMix : public Generator<HalideMix> {
    public:
        Input<Func> luma_input{"luma_input", 3};
        Input<Func> chroma_input{"chroma_input", 3};

        Output<Func> img_output{"img_mix_output", 3};

        GeneratorParam<Type> img_output_type{"img_output_type", UInt(16)};
        GeneratorParam<bool> define_schedule{"define_schedule", true};

        void generate() {
            assert(luma_input.type() == img_output_type);
            assert(chroma_input.type() == img_output_type);

            img_output(x, y, c) = mux(c, {luma_input(x, y, 0), chroma_input(x, y, 1), chroma_input(x, y, 2)});
            // select( c == 0, luma_input(x, y, 0),
            //         c == 1, chroma_input(x, y, 1),
            //               , chroma_input(x, y, 2)) //c == 2
        }

        void schedule() {
            if(auto_schedule) {
                luma_input.set_estimates({{0, 4000}, {0, 3000}, {0, 1}});
                chroma_input.set_estimates({{0, 4000}, {0, 3000}, {1, 2}});
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if (define_schedule) {
                img_output
                    .compute_root()
                    .bound(c, 0, 3)
                    .unroll(c)
                    .parallel(y)
                    .reorder(x, c, y)
                ;
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
};
