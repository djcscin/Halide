
#include "HalideBuffer.h"
#include "halide_benchmark.h"
#include "sort.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 2) {
        puts("Usage: ./sort size");
        return 1;
    }
    const uint32_t size = atoi(argv[1]);

    Buffer<int> input(size), output(size);

    input.for_each_value([](int &v) { v = rand() & 0x7fffffff; });

    printf("benchmark %u: %.2f ms\n", size,
        1e3*benchmark(5, 20, [&] {
            sort(input, output);
        })
    );

    return 0;
}
