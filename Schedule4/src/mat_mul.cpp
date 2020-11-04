
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "mat_mul.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 2) {
        puts("Usage: ./mat_mul size");
        return 1;
    }
    const uint32_t size = atoi(argv[1]);

    // C = A * B
    Buffer<float> A(size, size), B(size, size), C(size, size);

    A.for_each_value([](float &v) { v = (rand() % 1024) / 1023.0f; });
    B.for_each_value([](float &v) { v = (rand() % 1024) / 1023.0f; });

    printf("benchmark %u: %.2f ms\n", size,
        1e3*benchmark(5, 1, [&] {
            mat_mul(A, B, C);
        })
    );

    return 0;
}
