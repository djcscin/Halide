
#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"

#include "gamma.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 8) {
        puts("Usage: ./gamma0 path_input_image1 a1 gamma1 path_input_image2 a2 gamma2 path_output_image");
        return 1;
    }
    const char * path_input1 = argv[1];
    const float a1 = atof(argv[2]);
    const float gamma1 = atof(argv[3]);

    const char * path_input2 = argv[4];
    const float a2 = atof(argv[5]);
    const float gamma2 = atof(argv[6]);

    const char * path_output = argv[7];

    Buffer<uint8_t> input1 = load_image(path_input1);
    Buffer<uint8_t> input2 = load_image(path_input2);
    Buffer<uint8_t> output = Buffer<uint8_t>(input1.width(), input1.height(), input1.channels());

    printf("benchmark: %.2f ms\n",
        1e3*benchmark(3, 10, [&] {
            gamma(input1, a1, gamma1, input2, a2, gamma2, output);
        })
    );

    save_image(output, path_output);

    return 0;
}
