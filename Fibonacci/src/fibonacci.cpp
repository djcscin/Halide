
#include <string>

#include "HalideBuffer.h"
#include "fibonacci.h"

using namespace Halide::Runtime;

int main(int argc, char ** argv) {

    if(argc < 2) {
        puts("Usage: ./fibonacci number");
        return 1;
    }
    const uint number = atoi(argv[1]);

    Buffer<uint64_t> output = Buffer<uint64_t>::make_scalar();

    fibonacci(number, output);

    printf("Fibonacci de %u é %lu\n", number, output());

    return 0;
}
