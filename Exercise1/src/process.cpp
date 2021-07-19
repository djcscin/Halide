#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "halide_func.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {

    if (argc < 6) {
        printf("Usage: ./process in.png out.png max_difference iterations samples\n");
        return 0;
    }
    const char * input_filename = argv[1];
    const char * output_filename = argv[2];
    const int max_difference = atoi(argv[3]);
    const int iterations = atoi(argv[4]);
    const int samples = atoi(argv[5]);

    Buffer<uint8_t> input = load_image(input_filename);
    Buffer<uint8_t> output(input.width(), input.height(), 3);

    if(iterations > 0) {
        double best = benchmark(samples, iterations, [&]() {
            halide_func(input, max_difference, output);
        });
        printf("execution time: %lf ms\n", best * 1e3);
    } else {
        halide_func(input, max_difference, output);
    }

    convert_and_save_image(output, output_filename);

    return 0;
}