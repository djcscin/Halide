#ifndef _DNG_IO_
#define _DNG_IO_

#include <cstdio>
#include <cstdlib>
#include <iostream>

#define TINY_DNG_LOADER_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#define TINY_DNG_NO_EXCEPTION
#include "tiny_dng_loader.h"

#include "HalideBuffer.h"
#include "CFA.hpp"

using namespace Halide::Runtime;

template<typename T>
struct Raw {
    Buffer<T> buffer;
    Buffer<T> black_level;
    T white_level;
    uint8_t cfa_pattern;
};

template<typename T>
Raw<T> load_dng(const char * filename) {
    Raw<T> ret;

    std::string warn, err;
    std::vector<tinydng::DNGImage> images;
    std::vector<tinydng::FieldInfo> custom_fields;

    bool read = tinydng::LoadDNG(filename, custom_fields, &images, &warn, &err);

    if(read && (images.size() > 0)) {
        const tinydng::DNGImage &image = images[0];

        ret.white_level = std::max(
            std::max(image.white_level[0], image.white_level[1]),
            std::max(image.white_level[2], image.white_level[3])
        );

        const char first = image.cfa_plane_color[image.cfa_pattern[0][0]];
        if(first == 0) {
            ret.cfa_pattern = RGGB;
        } else if(first == 1) {
            const char second = image.cfa_plane_color[image.cfa_pattern[0][1]];
            if(second == 0) {
                ret.cfa_pattern = GRBG;
            } else if(second == 2) {
                ret.cfa_pattern = GBRG;
            } else {
                ret.cfa_pattern = NONE;
            }
        } else if(first == 2) {
            ret.cfa_pattern = BGGR;
        } else {
            ret.cfa_pattern = NONE;
        }

        ret.black_level = Buffer<T>(4);
        for(int i = 0; i < 4; ++i) {
            ret.black_level(i) = image.black_level[i];
        }

        T * data = static_cast<T *>(malloc(image.samples_per_pixel * image.width * image.height * sizeof(T)));
        memcpy(data, &image.data[0], image.samples_per_pixel * image.width * image.height * sizeof(T));
        Buffer<T> raw_buffer(data, image.width, image.height);
        ret.buffer = raw_buffer.copy();
        free(data);
    } else {
        ret.white_level = 0;
        ret.buffer = Buffer<T>(0);

        if(!warn.empty()) {
            std::cout << "Warning: " << warn << std::endl;
        }
        if(!err.empty()) {
            std::cout << "Error: " << err;
        }
    }

    return ret;
}

#endif
