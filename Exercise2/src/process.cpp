#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "halide_func.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {

    if (argc < 7) {
        printf("Usage: ./process in.png out.png sigma run_twice iterations samples\n");
        return 0;
    }
    const char * input_filename = argv[1];
    const char * output_filename = argv[2];
    const float sigma = atof(argv[3]);
    const bool run_twice = atoi(argv[4]) != 0;
    const int iterations = atoi(argv[5]);
    const int samples = atoi(argv[6]);

    Buffer<uint8_t> input = load_image(input_filename);
    Buffer<uint8_t> output(input.width(), input.height(), 3);

    if(iterations > 0) {
        double best = benchmark(samples, iterations, [&]() {
            halide_func(input, sigma, run_twice, output);
        });
        printf("execution time: %lf ms\n", best * 1e3);
    } else {
        halide_func(input, sigma, run_twice, output);
    }

    convert_and_save_image(output, output_filename);

    return 0;
}