#ifndef __RUN_BENCHMARK__
#define __RUN_BENCHMARK__

#include "halide_benchmark.h"

namespace {
    using namespace Halide::Tools;

    void run_benchmark(int numel, const std::function<void()> &op) {
        BenchmarkResult time = benchmark(op);
        printf("execution time: %lf ms %lf ns per pixel\n", time * 1e3, time * 1e9 / numel);
    }
}

#endif
