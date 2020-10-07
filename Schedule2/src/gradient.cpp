
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "gradient.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 4) {
        puts("Usage: ./gradient path_input_image threshold path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const uint threshold = atoi(argv[2]);
    const char * path_output = argv[3];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>(input.width() - 2, input.height() - 2);

    printf("benchmark: %.2f ms\n",
        1e3*benchmark(3, 20, [&] {
            gradient(input, threshold, output);
        })
    );

    save_image(output, path_output);

    return 0;
}
