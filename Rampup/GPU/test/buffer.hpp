#ifndef __BUFFER__
#define __BUFFER__

#include "HalideBuffer.h"

#if defined(HALIDE_RUNTIME_OPENCL)
    #define IS_DEVICE     true
    #define INTERFACE     halide_opencl_device_interface()
    #define INTERFACE_TEX halide_opencl_image_device_interface()
#elif defined(HALIDE_RUNTIME_OPENGLCOMPUTE)
    #define IS_DEVICE     true
    #define INTERFACE     halide_openglcompute_device_interface()
    #define INTERFACE_TEX halide_openglcompute_device_interface()
#else
    #define IS_DEVICE     false
    #define INTERFACE     nullptr
    #define INTERFACE_TEX nullptr
#endif

namespace {
    using namespace Halide::Runtime;

    template<typename T>
    static inline void copy_to_gpu(
        Buffer<T> & buffer,
        const struct halide_device_interface_t * device_interface
    ) {
        if(device_interface != nullptr) {
            buffer.set_host_dirty();
            buffer.copy_to_device(device_interface);
        }
    }

    template<typename T>
    static inline void copy_to_cpu(Buffer<T> & buffer) {
        if(buffer.has_device_allocation()) {
            buffer.set_device_dirty();
            buffer.copy_to_host();
        }
    }

    template<typename T>
    static inline Buffer<T> create_buffer(
        const struct halide_device_interface_t * device_interface,
        const std::vector<int> & extents,
        const std::vector<int> & mins = {}
    ) {
        if(device_interface != nullptr) {
            Buffer<T> buffer(nullptr, extents);
            buffer.set_min(mins);
            buffer.device_malloc(device_interface);
            return buffer;
        } else {
            Buffer<T> buffer(extents);
            buffer.set_min(mins);
            return buffer;
        }
    }

    template<typename T>
    static inline void release_buffer(Buffer<T> & buffer) {
        if(buffer.has_device_allocation()) {
            buffer.device_free();
        }
    }
}

#endif
