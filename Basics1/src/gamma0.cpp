#include "Halide.h"
#include "halide_image_io.h"

using namespace Halide;
using namespace Halide::Tools;
using namespace Halide::ConciseCasts;

// building:
/* g++ src/gamma0.cpp -fno-rtti -std=c++11 \
    -I $HALIDE_ROOT/include/ -I $HALIDE_ROOT/tools \
    -L $HALIDE_ROOT/lib/ -lHalide -lz -ldl -lpthread -ljpeg \
    `libpng-config --cflags --ldflags` -o bin/gamma0
*/

// running:
// bin/gamma0 path_input_image a gamma path_output_image
// output_image(_) = a * input_image(_) ^ gamma

int main(int argc, char ** argv) {

    if(argc < 5) {
        puts("Usage: ./gamma0 path_input_image a gamma path_output_image");
        return 1;
    }
    const char * path_input = argv[1];
    const float a = atof(argv[2]);
    const float gamma = atof(argv[3]);
    const char * path_output = argv[4];

    //include halide_image_io.h
    //Buffer<uint8_t> input = Halide::Tools::load_image(path_input);
    Buffer<uint8_t> input = load_image(path_input);

    // input(x, y, c)
    // a distância entre input(x, y, c) e input(x + 1, y, c) é o input.stride(0) - 1
    // a distância entre input(x, y, c) e input(x, y + 1, c) é o input.stride(1) - at least width
    // 1 2 3
    // 4 5 6
    // 1 2 3 _ 4 5 6 _
    // stride(0) = 1
    // stride(1) = 4
    // Ver sobre Buffer em https://halide-lang.org/docs/class_halide_1_1_runtime_1_1_buffer.html

    // HALIDE - BEGIN
    // Os algoritmos são implementados de forma declarativa através de funções paradigma funcional
    // Paradigma funcional - Variáveis e funções
    // Não temos for, while, if - instruções de controle
    // Sem efeito colateral
    Var x, y, c;

    Func input_float, gamma_float, gamma_int;

    // Halide::ConciseCasts
    // f32(f) é a mesma coisa que cast<float>(f) que é a mesma coisa de cast(Float(32), f)
    input_float(x, y, c) = f32(input(x, y, c));

    // Expr é usado pra melhorar legibilidade
    // gamma_float(x, y, c) = a * pow(input_float(x, y, c)/255.0f, gamma) * 255.0f;
    // 255.0f é 32 bits
    // 255.0 é 64 bits
    // Expr só pode literais dos tipos Int(32) e Float(32)
    // Para usar outros tipos, você tem que converter esses literais
    // Pra usar Float(64) teria que ser f64(255.0f)
    // Pra usar UInt(8) teria que ser u8(255)
    Expr input_unit = input_float(x, y, c)/255.0f;
    Expr output_unit = a * pow(input_unit, gamma);
    gamma_float(x, y, c) = output_unit * 255.0f;

    // u8(clamp(f, 0.0f, 255.0f)) é a mesma coisa de u8_sat(f)
    gamma_int(x, y, c) = u8_sat(gamma_float(x, y, c));
    // HALIDE - END

    // Ele computa as funções usando na dim(0) entre 0 e input.width
    // na dim(1) entre 0 e input.height() e
    // na dim(2) entre 0 e input.channels()
    Buffer<uint8_t> output = gamma_int.realize(input.width(), input.height(), input.channels());

    //include halide_image_io.h
    //Halide::Tools::save_image(output, path_output);
    save_image(output, path_output);

    return 0;
}
