/* run all schedulers using
for i in $(seq 0 2); do make SCHEDULER=$i; done
*/
/* test different split size for scheduler 0 using
for ss in 8 16 32 64 128 256 512; do make SCHEDULER=0 SPLIT_SIZE=$ss; done
*/
/* test different level values for schedulers 0 using
for l in $(seq 0 15); do make SCHEDULER=0 LEVEL=$l; done
*/
/* test different level values for schedulers 1 using
for l in $(seq 0 14); do make SCHEDULER=1 LEVEL=$l; done
*/
/* test different level values for schedulers 2 using
for l in $(seq 0 14); do make SCHEDULER=2 LEVEL=$l; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideSort : public Generator<HalideSort> {
    public:
        Input<Buffer<int>> input{"input", 1};

        Output<Buffer<int>> output{"output", 1};

        GeneratorParam<int> input_size{"input_size", 0x10000}; // tamanho máximo dos vetores de entrada e saída

        GeneratorParam<uint32_t> scheduler{"scheduler", 2};
        GeneratorParam<int> split_size{"split_size", 128};
        GeneratorParam<int> level{"level", 9}; // o nível da árvore do mergesort

        void generate() {
            // Se a gente quiser ordenar 100 valores, na verdade vai estar ordenando input_size valores
            // Pra 100, o ideal seria input_size=100, que ordenaria na verdade 128 valores
            // Ele sempre ordena n valores, sendo n potência de 2
            // std::numeric_limits<int>::max() é o maior valor inteiro
            // Então a gente teria, 100 valores do input e 28 valores igual ao std::numeric_limits<int>::max()
            input_bound = BoundaryConditions::constant_exterior(input, std::numeric_limits<int>::max());

            Expr a = input_bound(2*x); // par
            Expr b = input_bound(2*x + 1); // ímpar
            min_max(x) = tuple_select( // é a mesma coisa do select, mas ele retorna uma tupla
                a < b,
                Tuple(a, b), // uma tupla é um conjunto de n valores
                Tuple(b, a)
            ); // faz sentido usar tupla quando temos a atualização de mais de um valor por função
            // [0] é o primeiro elemento da tupla, no caso, o mínimo
            // [1] é o segundo elemento da tupla, no caso, o máximo
            result2(x, y) = select(x == 0, min_max(y)[0], min_max(y)[1]);
            // a Var y representa os chunks
            // a Var x representa os elementos no chunk
            // ordenando o vetor a cada dois elementos (último nível da árvore do mergesort)

            previous_result = result2;
            for(int chunk_size = 2; chunk_size < input_size; chunk_size *= 2) {
                r = RDom(0, min(input.width(), 2*chunk_size), "r");

                Func sorted("sorted" + std::to_string(2*chunk_size));
                // função recursiva com 3 valores:
                // 1. indice do primeiro chunk
                // 2. indice do segundo chunk
                // 3. valor ordenado (resultado)
                sorted(x, y) = Tuple(undef<int>(), undef<int>(), undef<int>());
                // inicialização os índices com 0 e o resultado com 0
                sorted(-1, y) = Tuple(0, 0, 0);

                Expr index_a = sorted(r - 1, y)[0];
                Expr index_b = sorted(r - 1, y)[1];
                // o clamp é necessário para informar o compilador do Halide que os valores do
                // acesso ao resultado da iteração anterior esteja entre 0 e chunk_size - 1
                Expr value_a = previous_result(clamp(index_a, 0, chunk_size - 1), 2*y);
                Expr value_b = previous_result(clamp(index_b, 0, chunk_size - 1), 2*y + 1);
                sorted(r, y) = tuple_select(
                // (index_b == chunk_size) quer dizer que ele processou todos os valores do segundo chunk
                // senão ele pergunta quem é o menor dos valores
                    (index_b == chunk_size) || ((index_a < chunk_size) && (value_a < value_b)),
                    Tuple(index_a + 1, index_b,     value_a),
                    Tuple(index_a,     index_b + 1, value_b)
                );

                // a atualização do resultado
                Func result("result" + std::to_string(2*chunk_size));
                result(x, y) = sorted(x, y)[2];
                previous_result = result;

                // adicionando num vetor pra poder fazer os schedulers
                sorteds.push_back(sorted);
                results.push_back(result);
            }

            output(x) = previous_result(x, 0); // no último nível só há um chunk
        }

        void schedule() {
            if (auto_schedule) {
                input.set_estimates({{0, input_size}});
                output.set_estimates({{0, input_size}});
            } else {
                int vector_size = get_target().natural_vector_size<int>();
                switch (scheduler) {
                    case 0:
						sorteds.back().compute_root();
                        for (int i = sorteds.size() - 2; i >= level; --i) {
                            sorteds[i].compute_root();
                            sorteds[i].update(1).parallel(y);
                        }
                        for (int i = level - 1; i >= 0; --i) {
                            sorteds[i].compute_root();
                            sorteds[i].update(1).split(y, yo, yi, split_size).parallel(yo);
                        }
                        result2
                            .compute_root()
                            .split(y, yo, yi, split_size).parallel(yo)
                            .bound(x, 0, 2).unroll(x) // pra eliminar o select
                        ;
                        break;

                    case 1:
						sorteds.back().compute_root();
                        for (int i = sorteds.size() - 2; i >= level; --i) {
                            sorteds[i].compute_root();
                            sorteds[i].update(1).parallel(y);
                        }
                        for (int i = level - 1; i >= 0; --i) {
                            sorteds[i].compute_at(sorteds[i+1], y);
                        }
                        result2
                            .compute_at(sorteds[0], y)
                            .bound(x, 0, 2).unroll(x)
                        ;
                        break;

                    default:
                    case 2:
						sorteds.back().compute_root();
                        for (int i = sorteds.size() - 2; i >= level; --i) {
                            sorteds[i].compute_root();
                            sorteds[i].update(1).parallel(y);
                        }
                        for (int i = level - 1; i >= 0; --i) {
                            sorteds[i].compute_at(sorteds[level], y);
                        }
                        result2
                            .compute_at(sorteds[level], y)
                            .bound(x, 0, 2).unroll(x)
                        ;
                        break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"};
        Var xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"};
        RDom r;
        RVar ri{"ri"}, ro{"ro"};

        Func input_bound{"input_bound"};
        Func min_max{"min_max"};
        Func result2{"result2"};
        Func previous_result{"previous_result"};
        std::vector<Func> sorteds, results;

    // Exercício:
    // Explorar outras opções de schedulers
};
HALIDE_REGISTER_GENERATOR(HalideSort, sort);
