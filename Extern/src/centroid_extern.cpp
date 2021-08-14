#include "HalideBuffer.h"

#ifdef _WIN32
#define DLLEXPORT __declspec(dllexport)
#else
#define DLLEXPORT
#endif

extern "C" {

DLLEXPORT int evaluate_run(halide_buffer_t * input, int width, halide_buffer_t * output) {
    if(input->is_bounds_query()) {
        input->dim[0].min = 0;
        input->dim[0].extent = width;
        input->dim[1].min = output->dim[1].min;
        input->dim[1].extent = output->dim[1].extent;
    } else {
        const int y_min = output->dim[1].min;
        const int y_max = y_min + output->dim[1].extent;
        // ponteiros com o início da linha a ser analizada
        // (x, y) = (x - 0.min) * 0.stride + (y - 1.min) * 1.stride
        uint8_t * in = ((uint8_t *)input->host) + (y_min - input->dim[1].min) * input->dim[1].stride;
        int32_t * out = (int32_t *)output->host;
        for(int j = y_min; j < y_max; ++j) {
            int num_runs = 0;
            uint8_t x0 = 0; // o pixel anterior
            for(int i = 0; i < width; ++i) {
                out[num_runs] = i;
                num_runs += x0 ^ in[i]; // se ele é diferente, vou começar a procurar a próxima corrida
                x0 = in[i]; // atualizando o pixel anterior
            }
            out[num_runs] = width;
            num_runs += x0 ^ 0;
            if(num_runs < width)
                out[num_runs] = width;

            // atualização dos ponteiros com o início da linha a ser analizada
            in += input->dim[1].stride;
            out += output->dim[1].stride;
        }
    }
    return 0;
}

DLLEXPORT int evaluate_momentum(halide_buffer_t * run, int width,
        halide_buffer_t * m00, halide_buffer_t * m10, halide_buffer_t * m01) {
    if(run->is_bounds_query()) {
        run->dim[0].min = 0;
        run->dim[0].extent = width;
        run->dim[1].min = m00->dim[0].min;
        run->dim[1].extent = m00->dim[0].extent;
    } else {
        const int y_min = m00->dim[0].min;
        const int y_max = y_min + m00->dim[0].extent;
        int32_t * in = ((int32_t *)run->host) + (y_min - run->dim[1].min) * run->dim[1].stride;
        int32_t * const p_m00 = (int32_t *)m00->host - y_min * m00->dim[0].stride;
        int32_t * const p_m10 = (int32_t *)m10->host - y_min * m10->dim[0].stride;
        int32_t * const p_m01 = (int32_t *)m01->host - y_min * m01->dim[0].stride;
        for(int j = y_min; j < y_max; ++j) {
            int _m00 = 0, _m10 = 0, _m01 = 0;
            for(int i = 0; i < width; i+=2) {
                if(in[i] == width)
                    break;
                const int size = in[i+1] - in[i];
                const int sum_x = in[i] + in[i+1] - 1;
                _m00 += size;
                _m10 += size * sum_x / 2;
                _m01 += size * j;
            }
            p_m00[j] = _m00;
            p_m10[j] = _m10;
            p_m01[j] = _m01;

            in += run->dim[1].stride;
        }
    }
    return 0;
}

}
