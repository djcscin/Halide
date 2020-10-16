
#include "HalideBuffer.h"
#include "dng_io.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

#include "demosaic.h"

int main(int argc, char ** argv) {

    if(argc < 6) {
        puts("Usage: ./demosaic path_input_image white_balance_red white_balance_green white_balance_blue path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    Buffer<float> white_balance(3);
    white_balance(0) = atof(argv[2]);
    white_balance(1) = atof(argv[3]);
    white_balance(2) = atof(argv[4]);
    const char * path_output = argv[5];

    Raw<uint16_t> input = load_dng<uint16_t>(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>(input.buffer.width(), input.buffer.height(), 3);

    printf("benchmark: %.2f us per pixel\n",
        1e9*benchmark(3, 5, [&] {
            demosaic(input.buffer, white_balance, input.black_level, input.white_level, input.cfa_pattern, output);
        }) / (input.buffer.number_of_elements())
    );

    save_image(output, path_output);

    return 0;
}
