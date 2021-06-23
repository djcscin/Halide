#include "HalideBuffer.h"
#include "halide_image_io.h"
#include "halide_benchmark.h"
#include "sharp.h"

using namespace Halide::Runtime;
using namespace Halide::Tools;

int main(int argc, char ** argv) {

    if(argc < 4) {
        puts("Usage: ./sharp path_input_image strength path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const int strength = atoi(argv[2]);
    const char * path_output = argv[3];

    Buffer<uint8_t> input = load_image(path_input);
    Buffer<uint8_t> output(input.width(), input.height(), input.channels());

#ifdef HALIDE_RUNTIME_HEXAGON
    halide_hexagon_set_performance_mode(nullptr, halide_hexagon_power_turbo);
    halide_hexagon_power_hvx_on(nullptr);
#endif

    printf("%.2f\n",
        1e3*benchmark(3, 1, [&] {
#if defined(HALIDE_RUNTIME_HEXAGON) | defined(HALIDE_RUNTIME_OPENCL) | defined(HALIDE_RUNTIME_CUDA)
            input.set_host_dirty(); // Isso significa que o buffer no host está modificado,
                                    // então o conteúdo precisa ser copiado para o device
            input.copy_to_device( // Implícito pelo set_host_dirty
#if defined(HALIDE_RUNTIME_HEXAGON)
                halide_hexagon_device_interface()
#elif defined(HALIDE_RUNTIME_OPENCL)
                halide_opencl_device_interface()
#else
                halide_cuda_device_interface()
#endif
            );
#endif
            sharp(input, strength, output);
#if defined(HALIDE_RUNTIME_HEXAGON) | defined(HALIDE_RUNTIME_OPENCL) | defined(HALIDE_RUNTIME_CUDA)
            output.copy_to_host();
#endif
        })
    );

#ifdef HALIDE_RUNTIME_HEXAGON
    halide_hexagon_power_hvx_off(nullptr);
    halide_hexagon_set_performance_mode(nullptr, halide_hexagon_power_default);
#endif

    save_image(output, path_output);

    return 0;
}
