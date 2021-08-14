#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "centroid.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char **argv) {

    if (argc < 2) {
        printf("Usage: ./process in.mat\n");
        return 0;
    }
    const char * input_filename = argv[1];

    Buffer<uint8_t> input = load_image(input_filename);
    Buffer<float> output_x = Buffer<float>::make_scalar();
    Buffer<float> output_y = Buffer<float>::make_scalar();

    BenchmarkResult time = benchmark([&]() {
        centroid(input, output_x, output_y);
    });
    printf("centroid: (%.3f,%.3f) ", output_x(), output_y());
    printf("execution time: %lf ms\n", time * 1e3);

    return 0;
}
