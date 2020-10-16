
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "denoise.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 5) {
        puts("Usage: ./denoise path_input_image sigma_spatial sigma_range path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const float sigma_spatial = atof(argv[2]);
    const float sigma_range = atof(argv[3]);
    const char * path_output = argv[4];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>::make_with_shape_of(input);

    printf("benchmark: %.2f ms\n",
        1e3*benchmark(3, 1, [&] {
            denoise(input, sigma_spatial, sigma_range, output);
        })
    );

    save_image(output, path_output);

    return 0;
}
