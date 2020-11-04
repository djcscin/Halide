
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"

#include "cd.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 4) {
        puts("Usage: ./cd path_input_image sigma path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const float sigma = atof(argv[2]);
    const char * path_output = argv[3];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>::make_with_shape_of(input);

    printf("benchmark %.2f: %.2f ms\n", sigma,
        1e3*benchmark(5, 1, [&] {
            cd(input, sigma, output);
        })
    );

    save_image(output, path_output);

    return 0;
}
