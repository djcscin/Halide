
#include <string>

#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "laplacian0.h"
#include "laplacian1.h"
#include "laplacian2.h"
#include "unsharp_gauss.h"
#include "dog.h"
#include "gaussian.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 5) {
        puts("Usage: ./sharp path_input_image strength filter path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const float strength = atof(argv[2]);
    const char * filter = argv[3];
    const char * path_output = argv[4];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output = Buffer<uint8_t>::make_with_shape_of(input);

    if(std::string(filter) == "laplacian0") {
        laplacian0(input, strength, output);
    } else if(std::string(filter) == "laplacian1") {
        laplacian1(input, strength, output);
    } else if(std::string(filter) == "laplacian2") {
        laplacian2(input, strength, output);
    } else if(std::string(filter) == "unsharp_gauss") {
        unsharp_gauss(input, strength, output);
    } else if(std::string(filter) == "dog") {
        dog(input, strength, output);
    }  else if(std::string(filter) == "gaussian") {
        gaussian(input, strength, output);
    }

    save_image(output, path_output);

    return 0;
}
