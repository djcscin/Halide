/* run all schedulers using
for i in $(seq 1 45); do echo SCHEDULER $i; make clean && make SCHEDULER=$i PARALLEL_SIZE=32; done
or
for i in $(seq 1 45); do echo SCHEDULER $i; make clean && make test_desktop DESKTOP=true SCHEDULER=$i; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideGenerator : public Halide::Generator<HalideGenerator> {
private:
    Var x{"x"}, y{"y"}, c{"c"};
    Var xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"}, yc{"yc"};
    Func input16{"input16"}, n_min{"min"}, n_max{"max"};

public:
    GeneratorParam<int> scheduler{"scheduler", 0};
    GeneratorParam<int> parallel_size{"parallel_size", 64};

    Input<Buffer<uint8_t>> input{"input", 3};
    Input<uint8_t> max_difference{"max_difference"};
    Output<Buffer<uint8_t>> output{"output", 3};

    void generate() {
        input16(x, y, c) = i16(BoundaryConditions::mirror_image(input)(x, y, c));
        n_min(x, y, c) = min(
            min(input16(x, y - 1, c), input16(x - 1, y, c)),
            min(input16(x + 1, y, c), input16(x, y + 1, c))
        ) - i16(max_difference);
        n_max(x, y, c) = max(
            max(input16(x, y - 1, c), input16(x - 1, y, c)),
            max(input16(x + 1, y, c), input16(x, y + 1, c))
        ) + i16(max_difference);

        output(x, y, c) = u8(clamp(input16(x, y, c), n_min(x, y, c), n_max(x, y, c)));
    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        input.set_estimates({{0, 4000},{0, 3000},{0, 3}});
        max_difference.set_estimate(30);
        output.set_estimates({{0, 4000},{0, 3000},{0, 3}});

        switch (scheduler) {
//Schedules 1 a 6 exercitam a ordem dos loops. Dica: usará reorder e parallel.
//Qual deve ser a ordem de loops? Qual deve ser a variável a do loop mais interno? Por quê?
//Qual variável não deve ser a do loop mais externo? Por quê?
//RESPOSTA: A variável x deve ser a mais interna, pois assim o acesso à memória seria sequencial na mesma thread.
        case 1:
            output
                .compute_root()
                .parallel(c)
            ;
            break;
// produce output:
//   parallel c:
//     for y:
//       for x:
//         output(...) = ...
        case 2:
            output
                .compute_root()
                .reorder(y, x, c)
                .parallel(c)
            ;
            break;
// produce output:
//   parallel c:
//     for x:
//       for y:
//         output(...) = ...
        case 3:
            output
                .compute_root()
                .reorder(x, c, y)
                .parallel(y)
            ;
            break;
// produce output:
//   parallel y:
//     for c:
//       for x:
//         output(...) = ...
        case 4:
            output
                .compute_root()
                .reorder(c, x, y)
                .parallel(y)
            ;
            break;
// produce output:
//   parallel y:
//     for x:
//       for c:
//         output(...) = ...
        case 5:
            output
                .compute_root()
                .reorder(y, c, x)
                .parallel(x)
            ;
            break;
// produce output:
//   parallel x:
//     for c:
//       for y:
//         output(...) = ...
        case 6:
            output
                .compute_root()
                .reorder(c, y, x)
                .parallel(x)
            ;
            break;
// produce output:
//   parallel x:
//     for y:
//       for c:
//         output(...) = ...

//Os schedules 1 a 6 já exercitaram execuções em CPU diferentes com parallel,
//mas em qual loop devemos usar parallel? Mais externo? Mais interno? Schedules 7 e 8 exercitam isso.
//Devemos usar loops paralelos encadeados? Por quê? Schedules 9 e 10 exercitam isso.
//Como dar mais paralelismo sem usar loops paralelos encadeados? Schedules 11 a 16 exercitam isso.
//Por que os schedules 15 e 16 apresentam um tempo de execução menores?
//O valor do split para ser usado no parallel é parallel_size. No meu computador, o melhor foi 128; no celular, 32.
//Exercite trocar o parallel_size. Para isso, use PARALLEL_SIZE={1,2,4,8,...} no make.
//for p in 1 2 4 8 16 32 64 128 256; do echo PARALLEL_SIZE $p; make clean && make SCHEDULER=15 PARALLEL_SIZE=$p; done
//Por que até um certo valor há uma melhora de tempo, mas depois piora?
//Observe "average threads used" no profile.
//Dica: use split e fuse para alguns dos schedules
//Dica: y.yc significa que a variável mais interna do fuse é y.
//RESPOSTA: Os loops paralelos devem ser mais externos possível e não devem haver loops paralelos encadeados,
//pois existe um custo para criar e destruir uma thread.
//A partir de um certo valor de parallel_size o tempo piora por dois motivos: o número de threads usados diminui
//e a memória usada passa a ser maior do que a cache disponível.
        case 7:
            output
                .compute_root()
                .parallel(y)
            ;
            break;
// produce output:
//   for c:
//     parallel y:
//       for x:
//         output(...) = ...
        case 8:
            output
                .compute_root()
                .parallel(x)
            ;
            break;
// produce output:
//   for c:
//     for y:
//       parallel x:
//         output(...) = ...
        case 9:
            output
                .compute_root()
                .parallel(y)
                .parallel(c)
            ;
            break;
// produce output:
//   parallel c:
//     parallel y:
//       for x:
//         output(...) = ...
        case 10:
            output
                .compute_root()
                .parallel(y)
                .parallel(c)
                .reorder(x, c, y)
            ;
            break;
// produce output:
//   parallel y:
//     parallel c:
//       for x:
//         output(...) = ...
        case 11:
            output
                .compute_root()
                .fuse(y, c, yc).parallel(yc)
            ;
            break;
// produce output:
//   parallel y.yc:
//     for x:
//       output(...) = ...
        case 12:
            output
                .compute_root()
                .fuse(c, y, yc).parallel(yc)
            ;
            break;
// produce output:
//   parallel c.yc:
//     for x:
//       output(...) = ...
        case 13:
            output
                .compute_root()
                .parallel(c)
                .split(y, yo, yi, parallel_size).parallel(yo)
                .reorder(x, yi, yo, c)
            ;
            break;
// produce output:
//   parallel c:
//     parallel y.yo:
//       for y.yi in [0, 127]:
//         for x:
//           output(...) = ...
        case 14:
            output
                .compute_root()
                .parallel(c)
                .split(y, yo, yi, parallel_size).parallel(yo)
                .reorder(x, yi, c, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     parallel c:
//       for y.yi in [0, 127]:
//         for x:
//           output(...) = ...
        case 15:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).fuse(yo, c, yc).parallel(yc)
                .reorder(x, yi, yc)
            ;
            break;
// produce output:
//   parallel y.yo.yc:
//     for y.yi in [0, 127]:
//       for x:
//         output(...) = ...
        case 16:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).fuse(c, yo, yc).parallel(yc)
                .reorder(x, yi, yc)
            ;
            break;
// produce output:
//   parallel c.yc:
//     for y.yi in [0, 127]:
//       for x:
//         output(...) = ...

//Schedules 17 a 20 exercitam vectorize. Qual é a recomendação para o uso do vectorize?
//Loop mais interno ou mais externo? Algum paralelo com parallel?
//O split para usar no vectorize usa vector_size a partir de natural_vector_size(UInt(8)).
//Esse valor será diferente de acordo com a CPU. Nos exemplos, esse valor é 32.
//RESPOSTA:O loop a ser vectorizado deve ser o mais interno, o oposto ao loop a ser paralelizado.
        case 17:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).fuse(yo, c, yc).parallel(yc)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(xi, xo, yi, yc)
            ;
            break;
// produce output:
//   parallel y.yo.yc:
//     for y.yi in [0, 127]:
//       for x.xo:
//         vectorized x.xi in [0, 31]:
//           output(...) = ...
        case 18:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).fuse(c, yo, yc).parallel(yc)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(xi, xo, yi, yc)
            ;
            break;
// produce output:
//   parallel c.yc:
//     for y.yi in [0, 127]:
//       for x.xo:
//         vectorized x.xi in [0, 31]:
//           output(...) = ...
        case 19:
            output
                .compute_root()
                .parallel(c)
                .split(y, yo, yi, vector_size).vectorize(yi)
            ;
            break;
// produce output:
//   parallel c:
//     for y.yo:
//       vectorized y.yi in [0, 31]:
//         for x:
//           output(...) = ...
        case 20:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            break;
// produce output:
//   parallel c:
//     for y:
//       for x.xo:
//         vectorized x.xi in [0, 31]:
//           output(...) = ...

//Os schedules de 21 a 30 exercitam compute_root/compute_at e store_at.
//Existe alguma lógica no tempo de execução? Qual?
//Olhe o stmt.
//Observe onde está o "store" no print loop nest.
//RESPOSTA: O store_at quanto mais externo, menor o tempo de execução, uma vez que diminui a quantidade de alocações
//e aumenta a probabilidade de reuso da memória sem a necessidade de uma nova computação.
//O compute_at quanto mais interno, melhor localidade produtor-consumidor, mas também reduz o acesso sequencial de memória.
//Logo, geralmente a melhor posição do compute_at não é nem mais interno, nem mais externo (compute_root).
        case 21:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_root()
            ;
            break;
// produce input16:
//   for c:
//     for y:
//       for x:
//         input16(...) = ...
// consume input16:
//   produce output:
//     parallel c:
//       for y:
//         for x:
//           output(...) = ...
        case 22:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, c).store_at(output, c)
            ;
            break;
// produce output:
//   parallel c:
//     produce input16:
//       for c:
//         for y:
//           for x:
//             input16(...) = ...
//     consume input16:
//       for y:
//         for x:
//           output(...) = ...
        case 23:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       for y:
//         produce input16:
//           for c:
//             for y:
//               for x:
//                 input16(...) = ...
//         consume input16:
//           for x:
//             output(...) = ...
        case 24:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, y).store_at(output, y)
            ;
            break;
// produce output:
//   parallel c:
//     for y:
//       produce input16:
//         for c:
//           for y:
//             for x:
//               input16(...) = ...
//       consume input16:
//         for x:
//           output(...) = ...
        case 25:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, x).store_at(output, c)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       for y:
//         for x:
//           produce input16:
//             for c:
//               for y:
//                 for x:
//                   input16(...) = ...
//           consume input16:
//             output(...) = ...
        case 26:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, x).store_at(output, y)
            ;
            break;
// produce output:
//   parallel c:
//     for y:
//       store input16:
//         for x:
//           produce input16:
//             for c:
//               for y:
//                 for x:
//                   input16(...) = ...
//           consume input16:
//             output(...) = ...
        case 27:
            output
                .compute_root()
                .parallel(c)
            ;
            input16
                .compute_at(output, x).store_at(output, x)
            ;
            break;
// produce output:
//   parallel c:
//     for y:
//       for x:
//         produce input16:
//           for c:
//             for y:
//               for x:
//                 input16(...) = ...
//         consume input16:
//           output(...) = ...
        case 28:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       for y:
//         produce input16:
//           for c:
//             for y:
//               for x:
//                 input16(...) = ...
//         consume input16:
//           for x.xo:
//             vectorized x.xi in [0, 31]:
//               output(...) = ...
        case 29:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, xo).store_at(output, c)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       for y:
//         for x.xo:
//           produce input16:
//             for c:
//               for y:
//                 for x:
//                   input16(...) = ...
//           consume input16:
//             vectorized x.xi in [0, 31]:
//               output(...) = ...
        case 30:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       for y:
//         produce input16:
//           for c:
//             for y:
//               for x.xo:
//                 vectorized x.xi in [0, 31]:
//                   input16(...) = ...
//         consume input16:
//           for x.xo:
//             vectorized x.xi in [0, 31]:
//               output(...) = ...

//Schedules 31 a 35 exercitam compute_with. Quando usar?
//RESPOSTA: Você usa o compute_with quando duas funções não tem dependência. Pelo tempo de execução, você percebe que
//o compute_with só melhora o tempo de execução quando está no loop mais interno e ambas as funções aproveitam
//a localidade de referência. Observe que os schedules 34 e 35 podem produzir o mesmo binário visto que o loop vectorize
//não é realmente um loop.
        case 31:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_min
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_max
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       store max:
//         store min:
//           for y:
//             produce input16:
//               for c:
//                 for y:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       input16(...) = ...
//             consume input16:
//               produce max:
//                 for c:
//                   for y:
//                     for x.xo:
//                       vectorized x.xi in [0, 31]:
//                         max(...) = ...
//               consume max:
//                 produce min:
//                   for c:
//                     for y:
//                       for x.xo:
//                         vectorized x.xi in [0, 31]:
//                           min(...) = ...
//                 consume min:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       output(...) = ...
        case 32:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_min
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_max
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .compute_with(n_min, c)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       store min:
//         store max:
//           for y:
//             produce input16:
//               for c:
//                 for y:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       input16(...) = ...
//             consume input16:
//               produce min:
//                 produce max:
//                   for fused.c:
//                     for y:
//                       for x.xo:
//                         vectorized x.xi in [0, 31]:
//                           min(...) = ...
//                     for fused.c:
//                       for y:
//                         for x.xo:
//                           vectorized x.xi in [0, 31]:
//                             max(...) = ...
//               consume min:
//                 consume max:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       output(...) = ...
        case 33:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_min
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_max
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .compute_with(n_min, y)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       store min:
//         store max:
//           for y:
//             produce input16:
//               for c:
//                 for y:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       input16(...) = ...
//             consume input16:
//               produce min:
//                 produce max:
//                   for fused.c:
//                     for fused.y:
//                       for x.xo:
//                         vectorized x.xi in [0, 31]:
//                           min(...) = ...
//                       for fused.c:
//                         for fused.y:
//                           for x.xo:
//                             vectorized x.xi in [0, 31]:
//                               max(...) = ...
//               consume min:
//                 consume max:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       output(...) = ...
        case 34:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_min
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_max
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .compute_with(n_min, xo)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       store min:
//         store max:
//           for y:
//             produce input16:
//               for c:
//                 for y:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       input16(...) = ...
//             consume input16:
//               produce min:
//                 produce max:
//                   for fused.c:
//                     for fused.y:
//                       for x.fused.xo:
//                         vectorized x.xi in [0, 31]:
//                           min(...) = ...
//                         for fused.c:
//                           for fused.y:
//                             for x.fused.xo:
//                               vectorized x.xi in [0, 31]:
//                                 max(...) = ...
//               consume min:
//                 consume max:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       output(...) = ...
        case 35:
            output
                .compute_root()
                .parallel(c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            input16
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_min
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            n_max
                .compute_at(output, y).store_at(output, c)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .compute_with(n_min, xi)
            ;
            break;
// produce output:
//   parallel c:
//     store input16:
//       store min:
//         store max:
//           for y:
//             produce input16:
//               for c:
//                 for y:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       input16(...) = ...
//             consume input16:
//               produce min:
//                 produce max:
//                   for fused.c:
//                     for fused.y:
//                       for x.fused.xo:
//                         vectorized x.fused.xi:
//                           min(...) = ...
//                           for fused.c:
//                             for fused.y:
//                               for x.fused.xo:
//                                 for x.fused.xi:
//                                   max(...) = ...
//               consume min:
//                 consume max:
//                   for x.xo:
//                     vectorized x.xi in [0, 31]:
//                       output(...) = ...

//Schedules de 36 a 45 exercitam o bound e unroll.
//Por que essas estratégias não traduzem em melhora significativa quando não reduzem significativamente o número de operações?
//RESPOSTA: O uso do unroll neste caso aumenta o código sem diminuir significativamente o número de operações, por isso,
//o tempo pode até ser maior quando ele é usado.
        case 36:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
            ;
            break;
// produce output:
//   for c:
//     parallel y.yo:
//       for y.yi in [0, 127]:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 37:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3).unroll(c)
            ;
            break;
// produce output:
//   unrolled c in [0, 2]:
//     parallel y.yo:
//       for y.yi in [0, 127]:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 38:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(xi, xo, yi, c, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for c:
//       for y.yi in [0, 127]:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 39:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3).unroll(c)
                .reorder(xi, xo, yi, c, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     unrolled c in [0, 2]:
//       for y.yi in [0, 127]:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 40:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(xi, xo, c, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       for c:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 41:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3).unroll(c)
                .reorder(xi, xo, c, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       unrolled c in [0, 2]:
//         for x.xo:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 42:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(xi, c, xo, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       for x.xo:
//         for c:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 43:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3).unroll(c)
                .reorder(xi, c, xo, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       for x.xo:
//         unrolled c in [0, 2]:
//           vectorized x.xi in [0, 15]:
//             output(...) = ...
        case 44:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .reorder(c, xi, xo, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       for x.xo:
//         vectorized x.xi in [0, 15]:
//           for c:
//             output(...) = ...
        case 45:
            output
                .compute_root()
                .split(y, yo, yi, parallel_size).parallel(yo)
                .split(x, xo, xi, vector_size).vectorize(xi)
                .bound(c, 0, 3).unroll(c)
                .reorder(c, xi, xo, yi, yo)
            ;
            break;
// produce output:
//   parallel y.yo:
//     for y.yi in [0, 127]:
//       for x.xo:
//         vectorized x.xi in [0, 15]:
//           unrolled c in [0, 2]:
//             output(...) = ...

        default:
            break;
        }

        if (!auto_schedule)
            output.print_loop_nest();
    }
};
HALIDE_REGISTER_GENERATOR(HalideGenerator, halide_func)
