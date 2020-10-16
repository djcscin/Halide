#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideFibonacci : public Generator<HalideFibonacci> {
    public:
        Input<uint8_t> input{"input"};
        Output<uint64_t> output{"output"};

        void generate() {
            i = RDom(2, input - 1);

            fibonacci(x) = undef<uint64_t>();
            fibonacci(0) = u64(0);
            fibonacci(1) = u64(1);
            fibonacci(i) = fibonacci(i - 2) + fibonacci(i - 1);

            output() = fibonacci(input);
        }

    private:
        Var x{"x"};
        RDom i;
        Func fibonacci{"fibonacci"};

};
HALIDE_REGISTER_GENERATOR(HalideFibonacci, fibonacci);

// Exercício:
// Calcular o número de combinações de n elementos tomados p a p.
// Use a recursão do triângulo de Pascal
