
#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "hist_eq.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 3) {
        puts("Usage: ./hist_eq path_input_image path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const char * path_output = argv[2];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>::make_with_shape_of(input);

    printf("benchmark: %.2f ms\n",
        1e3*benchmark(5, 10, [&] {
            hist_eq(input, output);
        })
    );

    save_image(output, path_output);

    return 0;
}
