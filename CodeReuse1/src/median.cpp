
#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "median_2.h"
#include "median_3.h"
#include "median_y.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 4) {
        puts("Usage: ./median path_input_image run_only_on_luma path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const bool run_only_on_luma = std::string(argv[2]) == "true";
    const char * path_output = argv[3];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>::make_with_shape_of(input);

    if(input.dimensions() == 2) {
        median_2(input, output);
    } else if(input.dimensions() == 3) {
        if(run_only_on_luma) {
            median_y(input, output);
        } else {
            median_3(input, output);
        }
    }

    save_image(output, path_output);

    return 0;
}
