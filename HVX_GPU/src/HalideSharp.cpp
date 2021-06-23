#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideSharp : public Generator<HalideSharp> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 3};
        Input<uint8_t> strength{"strength"};

        Output<Buffer<uint8_t>> img_output{"img_output", 3};

        void generate() {
            input_bound = BoundaryConditions::repeat_edge(img_input);
            // repeat_image, mirror_image and mirror_interior do not work on Hexagon
            input_bound_32(x, y, c) = i32(input_bound(x, y, c));
            output = dog(input_bound_32, int_x, int_y, int2_x, int2_y);
            img_output(x, y, c) = u8_sat(output(x, y, c));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                strength.set_estimate(4);
                img_output.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
            } else if(get_target().has_gpu_feature()) {
                // GPU é um multi-processador com vários processadores e cada processador tem vários cores
                // Em Cuda as atividades são enviadas para blocos (processadores) que serão divididos em threads (que são executadas em conjunto nos cores)
                // Warp é um conjunto de threads que executam a mesma instrução nos cores, geralmente 16, 32 ou 64
                // Quando tem alguma iterrupção, troca-se de warp
                // OpenCL chama os blocos de work-group, as threads de work-item e warp de wavefront
                // OpenCL chama a memória local às threads de "per-element private memory" e a memória compartilhada entre as threads de "local memory"
                // Halide usa a nomenclatura de Cuda, exceto que a memória local às threads é MemoryType::Stack e a memória global é MemoryType::Heap
                // A memória compartilhada entre as threads é MemoryType::GPUShared e registradores é MemoryType::Register
                img_output
                    .compute_root()
                    .split(x, vo, vi, 122) // o número de threads será 128 devido à computação de int2_y (sempre usar um múltiplo do warp)
                    .split(y, yo, yi, 8).unroll(yi)
                    .reorder(yi, vi, vo, yo, c)
                    .gpu_threads(vi)
                    .gpu_blocks(vo, yo, c)
                    //.fuse(vo, yo, vo).fuse(vo, c, vo).gpu_blocks(vo)
                ;
                int_y
                    // .compute_at(img_output, vo) <- todos os valores que o bloco precisa no thread 0 ou em threads se usado gpu_threads
                    // .compute_at(img_output, vi) <- todos os valores que a thread precisa
                    .compute_at(img_output, vo)
                    .store_in(MemoryType::GPUShared)
                    .unroll(y).unroll(c)
                    .reorder(y, c, x)
                    .gpu_threads(x)
                ;
                int2_y
                    .compute_at(img_output, vo)
                    .store_in(MemoryType::GPUShared)
                    .unroll(y).unroll(c)
                    .reorder(y, c, x)
                    .gpu_threads(x)
                    .compute_with(int_y, x, {{x, LoopAlignStrategy::AlignStart}}) //AlignStart, AlignEnd ou NoAlign
                ;
            } else if(get_target().has_feature(Target::HVX_128)) {
                // HVX é uma especialização do Hexagon para vetorização de 128 bytes (HVX_128) e 64 bytes (HVX_64)
                // Halide removeu a implementação para HVX_64
                // HVX não tem memória L1 (Hexagon tem, mas HVX não), então recomenda-se usar prefetch
                // HVX tem problemas com conversão para aumentar o tamanho (scatter / gather)
                // HVX tem uma memória especial VTCM que é rápida para 8 bits, mas não tem para 32 bits
                // HVX não trabalha com ponto flutuante, só trabalha com inteiros
                // HVX trabalha melhor com memórias alinhadas do que com desalinhadas (pode ser uma afirmação antiga)
                //  por isso align_storage
                // Download Qualcomm Package Manager em https://developer.qualcomm.com/software/hexagon-dsp-sdk/tools
                // Faça o login
                //  * qpm-cli --login
                // Instale o Hexagon SDK
                //  * sudo mkdir /local
                //  * sudo chown <username> /local
                //  * qpm-cli --install hexagonsdk4.x
                // testsig-(SERIAL).so
                //  - passos em Qualcomm/Hexagon_SDK/4.3.0.0/docs/tools/sign.html
                //    * source Qualcomm/Hexagon_SDK/4.3.0.0/setup_sdk_env.source
                //    * Qualcomm/Hexagon_SDK/4.3.0.0/utils/scripts/signer.py sign -s <devices' serial number>
                //      (Retrive the serial number using `adb devices -l`)
                // libhalide_hexagon_host.so e libhalide_hexagon_remote_skel.so:
                //  - https://github.com/halide/Halide/tree/master/src/runtime/hexagon_remote/bin
                //    (baixar a pasta com https://minhaskamal.github.io/DownGit)
                //  - Qualcomm/Hexagon_SDK/4.3.0.0/tools/HALIDE_Tools/2.3.03/Halide/lib/
                //   * adb push <path>/v<xx>/libhalide_hexagon_remote_skel.so /vendor/lib/rfsa/adsp
                //   * adb push <path>/arm-32-android/libhalide_hexagon_host.so /vendor/lib/
                //   * adb push <path>/arm-64-android/libhalide_hexagon_host.so /vendor/lib64/
                const int vector_size = 128; // seria 64 para Target::HVX_64 (removido na versão 11 do Halide)
                img_output
                    .compute_root()
                    .fuse(y, c, vo).split(vo, vo, vi, 8).parallel(vo)
                    .split(x, xo, xi, vector_size).vectorize(xi)
                    .hexagon(vo) // <- Todos os loops internos a vo estão no Hexagon
                ;
                int_y
                    .compute_at(img_output, vi)
                    .align_storage(x, vector_size)
                    .split(x, xo, xi, vector_size, TailStrategy::RoundUp).vectorize(xi)
                ;
                int2_y
                    .compute_at(img_output, vi)
                    .align_storage(x, vector_size)
                    .split(x, xo, xi, vector_size, TailStrategy::RoundUp).vectorize(xi)
                    .compute_with(int_y, y)
                ;
                if(get_target().has_feature(Target::HVX_v65)) { // VTCM apenas no Hexagon v65
                    input_bound
                        .compute_at(img_output, vi).store_at(img_output, vo).fold_storage(_1, 8)
                        .store_in(MemoryType::VTCM)
                        .align_storage(_0, vector_size)
                        .split(_0, xo, xi, vector_size, TailStrategy::RoundUp).vectorize(xi)
                        .prefetch(img_input, y, 1)
                    ;
                }
            } else {
                const int vector_size = get_target().natural_vector_size<int8_t>()/2;
                img_output
                    .compute_root()
                    .fuse(y, c, vo).parallel(vo)
                    .vectorize(x, vector_size)
                ;
                int_y
                    .compute_at(img_output, vo)
                    .vectorize(x, vector_size)
                ;
                int2_y
                    .compute_at(img_output, vo)
                    .vectorize(x, vector_size)
                    .compute_with(int_y, y)
                ;
            }
        }

    private:
        Var x{"x"}, y{"y"}, c{"c"};
        Var vo{"vo"}, vi{"vi"};
        Var xo{"xo"}, xi{"xi"};
        Var yo{"yo"}, yi{"yi"};
        Func input_bound{"input_bound"};
        Func input_bound_32{"input_bound_32"};
        Func int_x{"int_x"}, int_y{"int_y"};
        Func int2_x{"int2_x"}, int2_y{"int2_y"};
        Func output{"output"};

        Func gaussian7(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) =
                       input(x, y-3, c)
                +  6 * input(x, y-2, c)
                + 15 * input(x, y-1, c)
                + 20 * input(x, y,   c)
                + 15 * input(x, y+1, c)
                +  6 * input(x, y+2, c)
                +      input(x, y+3, c);
            int_x(x, y, c) =
                       int_y(x-3, y, c)
                +  6 * int_y(x-2, y, c)
                + 15 * int_y(x-1, y, c)
                + 20 * int_y(x,   y, c)
                + 15 * int_y(x+1, y, c)
                +  6 * int_y(x+2, y, c)
                +      int_y(x+3, y, c);

            output(x, y, c) = int_x(x, y, c) / 4096; // 4096 -> 64*64

            return output;
        }

        Func gaussian5(Func input, Func & int_x, Func & int_y) {
            Func output;

            int_y(x, y, c) =
                       input(x, y-2, c)
                +  4 * input(x, y-1, c)
                +  6 * input(x, y,   c)
                +  4 * input(x, y+1, c)
                +      input(x, y+2, c);
            int_x(x, y, c) =
                       int_y(x-2, y, c)
                +  4 * int_y(x-1, y, c)
                +  6 * int_y(x,   y, c)
                +  4 * int_y(x+1, y, c)
                +      int_y(x+2, y, c);

            output(x, y, c) = int_x(x, y, c) / 256; // 256 -> 16*16

            return output;
        }

        Func dog(Func input, Func & int_x, Func & int_y, Func & int2_x, Func & int2_y) {
            Func output;

            Func gaussian1 = gaussian5(input, int_x, int_y);
            Func gaussian2 = gaussian7(input, int2_x, int2_y);

            Expr d_o_g = gaussian1(x, y, c) - gaussian2(x, y, c);
            output(x, y, c) = input(x, y, c) + i32(strength) * d_o_g;

            return output;
        }

        // Exercício testar a conversão para i16 em vez de i32
        // Tomar cuidado com overflow nas funções de gaussian5 e gaussian7
};
HALIDE_REGISTER_GENERATOR(HalideSharp, sharp)
