#include "HalideBuffer.h"
#include "dng_io.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

#include "lsc_map.h"
#include "demosaic.h"
#include "demosaic16.h"
#include "denoise.h"
#include "rgb_to_ycbcr.h"
#include "ycbcr_to_rgb.h"
#include "denoise16.h"
#include "mix.h"
#include "denoise16_chroma.h"
#include "denoise16_luma.h"
#include "isp.h"
#include "isp_luma.h"
#include "isp_debug.h"

int main(int argc, char ** argv) {

    if(argc < 19) {
        puts("Usage: ./isp path_input_image path_input_lsc_map white_balance_red white_balance_green white_balance_blue \
sigma_spatial sigma_range sigma_spatial_luma sigma_range_luma path_output_image(demosaiced) \
path_output1_image(demosaiced and denoised rgb) path_output2_image(demosaiced and denoised ycbcr) \
path_output3_image(demosaiced and denoised chroma) path_output4_image(demosaiced and denoise chroma & luma) \
path_output5_image(demosaiced and denoised chroma, but composable) \
path_output6_image(demosaiced and denoise chroma & luma, but composable) \
path_output7_image(demosaiced and denoised chroma, but composable and debug) \
path_output8_image(demosaiced, but composable and debug)");
        return 1;
    }
    const char * path_input = argv[1];
    const char * path_lsc_map = argv[2];
    Buffer<float> white_balance(3);
    white_balance(0) = atof(argv[3]);
    white_balance(1) = atof(argv[4]);
    white_balance(2) = atof(argv[5]);
    const float sigma_spatial = atof(argv[6]);
    const float sigma_range = atof(argv[7]);
    const float sigma_spatial_luma = atof(argv[8]);
    const float sigma_range_luma = atof(argv[9]);
    const char * path_output0 = argv[10];
    const char * path_output1 = argv[11];
    const char * path_output2 = argv[12];
    const char * path_output3 = argv[13];
    const char * path_output4 = argv[14];
    const char * path_output5 = argv[15];
    const char * path_output6 = argv[16];
    const char * path_output7 = argv[17];
    const char * path_output8 = argv[18];

    Raw<uint16_t> input = load_dng<uint16_t>(path_input);
    Buffer<float> lsc_map_input = load_and_convert_image(path_lsc_map);
    const int width = input.buffer.width();
    const int height = input.buffer.height();
    Buffer<uint8_t> output(width, height, 3);

    printf("benchmark:");

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            Buffer<float> lsc_map_resized = Buffer<float>::make_with_shape_of(input.buffer);
            lsc_map(lsc_map_input, width, height, lsc_map_resized);
            demosaic(input.buffer, lsc_map_resized, white_balance, input.black_level, input.white_level, input.cfa_pattern, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output0);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            Buffer<float> lsc_map_resized = Buffer<float>::make_with_shape_of(input.buffer);
            Buffer<uint16_t> output_dms = Buffer<uint16_t>(width, height, 3);
            lsc_map(lsc_map_input, width, height, lsc_map_resized);
            demosaic16(input.buffer, lsc_map_resized, white_balance, input.black_level, input.white_level, input.cfa_pattern, output_dms);
            denoise(output_dms, width, height, sigma_spatial, sigma_range, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output1);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            Buffer<float> lsc_map_resized = Buffer<float>::make_with_shape_of(input.buffer);
            Buffer<uint16_t> output_dms = Buffer<uint16_t>(width, height, 3);
            Buffer<uint16_t> ycbcr = Buffer<uint16_t>::make_with_shape_of(output_dms);
            Buffer<uint16_t> output_dns = Buffer<uint16_t>::make_with_shape_of(output_dms);
            lsc_map(lsc_map_input, width, height, lsc_map_resized);
            demosaic16(input.buffer, lsc_map_resized, white_balance, input.black_level, input.white_level, input.cfa_pattern, output_dms);
            rgb_to_ycbcr(output_dms, ycbcr);
            denoise16(ycbcr, width, height, sigma_spatial, sigma_range, output_dns);
            ycbcr_to_rgb(output_dns, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output2);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            Buffer<float> lsc_map_resized = Buffer<float>::make_with_shape_of(input.buffer);
            Buffer<uint16_t> output_dms = Buffer<uint16_t>(width, height, 3);
            Buffer<uint16_t> ycbcr = Buffer<uint16_t>::make_with_shape_of(output_dms);
            Buffer<uint16_t> chroma_dns = Buffer<uint16_t>(width, height, 2);
            Buffer<uint16_t> output_dns = Buffer<uint16_t>::make_with_shape_of(output_dms);
            lsc_map(lsc_map_input, width, height, lsc_map_resized);
            demosaic16(input.buffer, lsc_map_resized, white_balance, input.black_level, input.white_level, input.cfa_pattern, output_dms);
            rgb_to_ycbcr(output_dms, ycbcr);
            chroma_dns.set_min({0, 0, 1});
            denoise16_chroma(ycbcr, width, height, sigma_spatial, sigma_range, chroma_dns);
            mix(ycbcr, chroma_dns, output_dns);
            ycbcr_to_rgb(output_dns, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output3);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            Buffer<float> lsc_map_resized = Buffer<float>::make_with_shape_of(input.buffer);
            Buffer<uint16_t> output_dms = Buffer<uint16_t>(width, height, 3);
            Buffer<uint16_t> ycbcr = Buffer<uint16_t>::make_with_shape_of(output_dms);
            Buffer<uint16_t> chroma_dns = Buffer<uint16_t>(width, height, 2);
            Buffer<uint16_t> luma_dns = Buffer<uint16_t>(width, height, 1);
            Buffer<uint16_t> output_dns = Buffer<uint16_t>::make_with_shape_of(output_dms);
            lsc_map(lsc_map_input, width, height, lsc_map_resized);
            demosaic16(input.buffer, lsc_map_resized, white_balance, input.black_level, input.white_level, input.cfa_pattern, output_dms);
            rgb_to_ycbcr(output_dms, ycbcr);
            chroma_dns.set_min({0, 0, 1});
            denoise16_chroma(ycbcr, width, height, sigma_spatial, sigma_range, chroma_dns);
            denoise16_luma(ycbcr, width, height, sigma_spatial_luma, sigma_range_luma, luma_dns);
            mix(luma_dns, chroma_dns, output_dns);
            ycbcr_to_rgb(output_dns, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output4);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            isp(input.buffer, lsc_map_input, white_balance, input.black_level, input.white_level, input.cfa_pattern,
                sigma_spatial, sigma_range, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output5);

    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            isp_luma(input.buffer, lsc_map_input, white_balance, input.black_level, input.white_level, input.cfa_pattern,
                sigma_spatial, sigma_range, sigma_spatial_luma, sigma_range_luma, output);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output6);

    Buffer<uint8_t> output2 = Buffer<uint8_t>::make_with_shape_of(output);
    printf(" %.2f",
        1e9*benchmark(3, 1, [&] {
            isp_debug(input.buffer, lsc_map_input, white_balance, input.black_level, input.white_level, input.cfa_pattern,
                sigma_spatial, sigma_range, output, output2);
        }) / (input.buffer.number_of_elements())
    );
    save_image(output, path_output7);
    save_image(output2, path_output8);

    printf(" us per pixel\n");

    return 0;
}
