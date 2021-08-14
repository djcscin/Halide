/* run all schedulers and versions using
for v in $(seq 0 2); do for i in $(seq 0 2); do echo VERSION $v SCHEDULER $i; make clean && make VERSION=$v SCHEDULER=$i; done; done
or
for v in $(seq 0 2); do for i in $(seq 0 2); do echo VERSION $v SCHEDULER $i; make clean && make VERSION=$v SCHEDULER=$i test_desktop DESKTOP=true; done; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideCentroid : public Halide::Generator<HalideCentroid> {
private:
    Var x{"x"}, y{"y"};
    Var ry{"ry"};
    RDom r_idx, r_run;
    Func idx{"idx"}, run{"run"};
    Func m00{"m00"}, m10{"m10"}, m01{"m01"}, m{"m"};
    RDom r_m;
    Func mm00{"mm00"}, mm10{"mm10"}, mm01{"mm01"};

public:
    GeneratorParam<int> version{"version", 0};
    GeneratorParam<int> scheduler{"scheduler", 0};

    Input<Buffer<uint8_t>> input{"input", 2};
    Output<float[2]> output{"output"};

    void generate() {
        // faça que os min sejam 0
        input.dim(0).set_min(0);
        input.dim(1).set_min(0);

        Expr w = input.dim(0).extent();
        Expr h = input.dim(1).extent();

        switch (version)
        {
        case 0:
            {
                // momentum nm = soma (image(x, y) * x^n * y^m);
                // x = momentum 10 / momentum 00
                // y = momentum 01 / momentum 00
                m00(x, y) = i32(input(x, y));
                m10(x, y) = i32(input(x, y) * x);
                m01(x, y) = i32(input(x, y) * y);

                r_m = RDom(0, w, 0, h, "r_m"); // r_m = RDom(input); supondo os min igual à 0
                mm00() += m00(r_m.x, r_m.y);
                mm10() += m10(r_m.x, r_m.y);
                mm01() += m01(r_m.x, r_m.y);
            }
            break;

        case 1:
            {
                r_idx = RDom(1, w - 1, "r_idx");
                idx(x, y) = undef<int>();
                idx(0, y) = i32(input(0, y)) - 1;
                idx(r_idx, y) = idx(r_idx-1, y) + (input(r_idx-1, y) ^ input(r_idx, y)); // (a != b) == (a ^ b) porque a e b são binários
                // representação de pixel
                //  0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21
                //  0  0  1  1  0  1  1  1  1  1  1  0  0  0  1  1  1  1  1  1  0  0
                // -1 -1  0  0  1  2  2  2  2  2  2  3  3  3  4  4  4  4  4  4  5  5
                // representação de corrida
                // 0  1  2  3  4  5  6
                // 2  4  5 11 14 20 22

                r_run = RDom(1, w, "r_output");
                run(x, y) = w; // completo a linha da representação de corrida com a largura
                run(x, y) = select(x == idx(w-r_run, y), w-r_run, run(x, y)); // varrer do maior pro menor, porque pego o menor índice
                // run(x, y) = select(x == idx(r_run, y), min(run(x, y), r_run), run(x, y)); // varrer do menor pro maior, fazendo o mínimo

                Expr size = run(2*x + 1, y) - run(2*x, y);
                Expr sum_x = run(2*x, y) + run(2*x + 1, y) - 1;
                m00(x, y) = size;
                m10(x, y) = size * sum_x / 2; // soma da progressão arimética
                m01(x, y) = size * y;

                r_m = RDom(0, w/2, 0, h, "r_m");
                mm00() += m00(r_m.x, r_m.y);
                mm10() += m10(r_m.x, r_m.y);
                mm01() += m01(r_m.x, r_m.y);
            }
            break;

        case 2:
            {
                run.define_extern("evaluate_run", {Func(input), w}, Int(32), {x, y});
                run.function().extern_definition_proxy_expr() = input(0, y) + input(w-1, y);

                m.define_extern("evaluate_momentum", {run, w}, {Int(32), Int(32), Int(32)}, {y});
                Expr m_proxy_expr = run(0, y) + run(w-1, y);
                m.function().extern_definition_proxy_expr() = (m_proxy_expr, m_proxy_expr, m_proxy_expr);

                m00(y) = m(y)[0];
                m10(y) = m(y)[1];
                m01(y) = m(y)[2];

                r_m = RDom(0, h, "r_m");
                mm00() += m00(r_m);
                mm10() += m10(r_m);
                mm01() += m01(r_m);
            }
            break;

        }

        output[0]() = mm10()/f32(mm00());
        output[1]() = mm01()/f32(mm00());

    }

    void schedule() {
        int vector_size = natural_vector_size(UInt(8));

        if(auto_schedule) {
            input.set_estimates({{0, 4000},{0, 3000}});
        } else {
            switch(version) {
            case 0:
                {
                    mm00.compute_root();
                    mm10.compute_root();
                    mm01.compute_root();
                    switch(scheduler)
                    {
                    case 1:
                        break;

                    default:
                        Func int00 = mm00.update().rfactor(r_m.y, ry);
                        Func int10 = mm10.update().rfactor(r_m.y, ry);
                        Func int01 = mm01.update().rfactor(r_m.y, ry);
                        int00.in().compute_root().parallel(ry);
                        int10.in().compute_root().parallel(ry).compute_with(int00.in(), ry);
                        int01.in().compute_root().parallel(ry).compute_with(int00.in(), ry);
                        int00.compute_at(int00.in(), ry);
                        int10.compute_at(int00.in(), ry);
                        int01.compute_at(int00.in(), ry);
                        break;
                    }
                }
                break;

            case 1:
                {
                    mm00.compute_root();
                    mm10.compute_root();
                    mm01.compute_root();
                    Func int00 = mm00.update().rfactor(r_m.y, ry);
                    Func int10 = mm10.update().rfactor(r_m.y, ry);
                    Func int01 = mm01.update().rfactor(r_m.y, ry);
                    int00.in().compute_root().parallel(ry);
                    int10.in().compute_root().parallel(ry).compute_with(int00.in(), ry);
                    int01.in().compute_root().parallel(ry).compute_with(int00.in(), ry);
                    int00.compute_at(int00.in(), ry);
                    int10.compute_at(int00.in(), ry);
                    int01.compute_at(int00.in(), ry);
                    m00.compute_at(int00.in(), ry);
                    m10.compute_at(int00.in(), ry).compute_with(m00, x);
                    m01.compute_at(int00.in(), ry).compute_with(m00, x);
                    switch(scheduler) {
                    case 1:
                        run.compute_root().parallel(y);
                        run.update().reorder(x, r_run, y).parallel(y);
                        break;

                    default:
                        run.compute_at(int00.in(), ry);
                        run.update().reorder(x, r_run, y);
                        break;
                    }
                    idx.compute_at(run, y);
                }
                break;

            case 2:
                mm00.compute_root();
                mm10.compute_root();
                mm01.compute_root();
                m.compute_root().parallel(y);
                switch(scheduler)
                {
                case 1:
                    run.compute_root().parallel(y);
                    Func(input).compute_at(run, y);
                    break;

                default:
                    run.compute_at(m, y);
                    Func(input).compute_at(m, y);
                    break;
                }
                break;

            }
            Var yi, yo;
        }

    }

};
HALIDE_REGISTER_GENERATOR(HalideCentroid, centroid)
