
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "boxfilter.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 4) {
        puts("Usage: ./boxfilter path_input_image window_size path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const uint window_size = atof(argv[2]);
    const char * path_output = argv[3];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>(input.width() - window_size + 1, input.height() - window_size + 1, 3);

    printf("benchmark %d: %.2f ms\n", window_size, 1e3*benchmark(5, 1, [&] {
            boxfilter(input, window_size, output);
        })
    );
    
    save_image(output, path_output);

    return 0;
}
