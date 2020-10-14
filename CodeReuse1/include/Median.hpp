#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;
using Halide::_; // representa 0 a inf variáveis

class Median {
    protected:
        Var x{"x"}, y{"y"};
        Func img_input_bound{"img_input_bound"}, median_y{"median_y"}, median_x{"median_x"};

        // O filtro da mediana em si não é separável, mas o filtro de mediana 3x3 pode ser feito
        // separável com uma boa aproximação
        // - - ?
        // - . +
        // ? + +
        void evaluate_median(Func input, Expr width, Expr height) {
            img_input_bound = BoundaryConditions::mirror_image(input, {{0, width}, {0, height}});

            median_y(x, y, _) = median(img_input_bound(x, y-1, _), img_input_bound(x, y, _), img_input_bound(x, y+1, _));
            median_x(x, y, _) = median(median_y(x-1, y, _), median_y(x, y, _), median_y(x+1, y, _));
        }

    private:
        Expr median(Expr a, Expr b, Expr c) {
            return min(max(min(a, b),c), max(a, b));
            // min(max(a, c), b) -> a < b, a 1o ou 2o e b 2o ou 3o
        }
};