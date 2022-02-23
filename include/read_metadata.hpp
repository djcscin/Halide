#ifndef __READ_METADATA__
#define __READ_METADATA__

#include "HalideBuffer.h"

namespace {
    using namespace Halide::Runtime;

    void read_metadata(const char * path, Buffer<float> & wb, Buffer<float> & ccm) {
        int ret;
        FILE *f = fopen(path, "r");
        ret = fscanf(f, "%f %f %f", &wb(0), &wb(1), &wb(2));
        for(int i = 0; i < 3; ++i)
            ret = fscanf(f, "%f %f %f", &ccm(0,i), &ccm(1,i), &ccm(2,i));
        fclose(f);
    }
};

#endif
