#include "buffer.hpp"
#include "dng_io.h"
#include "read_metadata.hpp"
#include "transform_wb.hpp"
#include "halide_image_io.h"
#include "run_benchmark.hpp"

using namespace Halide::Runtime;
using namespace Halide::Tools;

#include "isp.h"

int main(int argc, char ** argv) {

    if(argc < 8) {
        puts("Usage: ./test_isp path_input path_lsc_map path_input_metadata gamma sigma_spatial sigma_range path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const char * path_lsc_map = argv[2];
    const char * path_input_metadata = argv[3];
    const float gamma = atof(argv[4]);
    const float sigma_spatial = atof(argv[5]);
    const float sigma_range = atof(argv[6]);
    const char * path_output = argv[7];

    Raw<uint16_t> input = load_dng<uint16_t>(path_input);
    const int width = input.buffer.width();
    const int height = input.buffer.height();
    const int numel = input.buffer.number_of_elements();
    Buffer<float> lsc_map = load_image(path_lsc_map);
    Buffer<float> wb_rgb(3);
    Buffer<float> wb4(4);
    Buffer<float> ccm(3,3);
    read_metadata(path_input_metadata, wb_rgb, ccm);
    transform_wb(input.cfa_pattern, wb_rgb, wb4);

    Buffer<uint16_t> output(width, height, 3);

    const struct halide_device_interface_t * texture_interface = INTERFACE_TEX;

    run_benchmark(numel, [&]() {
        copy_to_gpu(input.buffer, texture_interface);
        copy_to_gpu(lsc_map, texture_interface);
        copy_to_gpu(wb4, texture_interface);
        copy_to_gpu(ccm, texture_interface);
        isp(input.buffer, lsc_map, wb4, ccm, input.black_level, input.white_level, input.cfa_pattern, gamma, sigma_spatial, sigma_range, output);
        copy_to_cpu(output);
    });

    save_image(output, path_output);

    return 0;
}
