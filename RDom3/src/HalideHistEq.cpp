/* run all schedulers using
for i in $(seq 0 7); do make SCHEDULER=$i; done
*/

#include "Halide.h"

using namespace Halide;
using namespace Halide::ConciseCasts;

class HalideHistEq : public Generator<HalideHistEq> {
    public:
        Input<Buffer<uint8_t>> img_input{"img_input", 2};

        Output<Buffer<uint8_t>> img_output{"img_output", 2};

        GeneratorParam<uint> scheduler{"scheduler", 0};

        void generate() {
            r_hist = RDom(img_input);
            hist(i) = 0;
            hist(img_input(r_hist.x, r_hist.y)) += 1;

            r_cum_hist = RDom(1, 255);
            cum_hist(i) = undef<int32_t>();
            cum_hist(0) = hist(0);
            cum_hist(r_cum_hist) = cum_hist(r_cum_hist - 1) + hist(r_cum_hist);

            lut(i) = u8(cum_hist(i)*255.0f/cum_hist(255));

            img_output(x, y) = lut(img_input(x, y));
        }

        void schedule() {
            if (auto_schedule) {
                img_input.set_estimates({{0, 4000}, {0, 3000}, {0, 3}});
                img_output.set_estimates({{0, 4000}, {0, 3000}});
            } else {
                int vector_size = get_target().natural_vector_size<int32_t>();
                switch (scheduler)
                {
                default:
                case 0:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    break;

                case 1:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().rfactor(r_hist.y, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 2:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 4).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 3:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 8).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 4:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 16).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 5:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 32).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 6:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 64).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;

                case 7:
                    img_output
                        .compute_root()
                        .split(x, xo, xi, vector_size).vectorize(xi)
                        .parallel(y)
                    ;
                    lut
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    cum_hist.compute_root();
                    hist
                        .compute_root()
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm = hist.update().split(r_hist.y, ryo, ryi, 128).rfactor(ryo, ry);
                    intm
                        .compute_root()
                        .parallel(ry)
                        .split(i, io, ii, vector_size).vectorize(ii)
                    ;
                    intm.update()
                        .parallel(ry)
                    ;
                    hist.update()
                        .parallel(i)
                    ;
                    break;
                }
            }
        }

    private:
        Var x{"x"}, y{"y"}, i{"i"};
        Var ii{"ii"}, io{"io"}, xi{"xi"}, xo{"xo"}, yi{"yi"}, yo{"yo"};

        Func hist{"hist"}, cum_hist{"cum_hist"};
        Func lut{"lut"};

        RDom r_hist, r_cum_hist;
        RVar rxi{"rxi"}, rxo{"rxo"}, ryi{"ryi"}, ryo{"ryo"};
        Var rx{"rx"}, ry{"ry"};
        Func intm{"intm"};

};
HALIDE_REGISTER_GENERATOR(HalideHistEq, hist_eq);
