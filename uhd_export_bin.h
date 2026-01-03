#pragma once
#include <cstdint>
#include <cstring>
#include <fstream>
#include <vector>

struct UhdBinBuffer {
    std::vector<uint8_t> data;

    bool load(const char* path) {
        std::ifstream ifs(path, std::ios::binary);
        if (!ifs) {
            return false;
        }
        data.assign(std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());
        return !data.empty();
    }

    const uint8_t* ptr(uint64_t offset, uint64_t nbytes = 0) const {
        if (offset >= data.size()) {
            return nullptr;
        }
        if (nbytes > 0 && offset + nbytes > data.size()) {
            return nullptr;
        }
        return data.data() + static_cast<size_t>(offset);
    }

    template <typename T>
    const T* ptr_as(uint64_t offset, uint64_t nbytes = 0) const {
        return reinterpret_cast<const T*>(ptr(offset, nbytes));
    }

    bool read_f32(uint64_t offset, float* out) const {
        const uint8_t* p = ptr(offset, sizeof(float));
        if (!p) {
            return false;
        }
        std::memcpy(out, p, sizeof(float));
        return true;
    }

    bool read_i32(uint64_t offset, int32_t* out) const {
        const uint8_t* p = ptr(offset, sizeof(int32_t));
        if (!p) {
            return false;
        }
        std::memcpy(out, p, sizeof(int32_t));
        return true;
    }
};

inline size_t uhd_num_elements(const int32_t* shape, size_t ndim) {
    size_t total = 1;
    for (size_t i = 0; i < ndim; ++i) {
        total *= static_cast<size_t>(shape[i]);
    }
    return total;
}

inline bool uhd_unpack_lowbit(
    const uint8_t* packed,
    size_t packed_bytes,
    int bits,
    int8_t* out,
    size_t out_elems
) {
    if (!packed || !out) {
        return false;
    }
    if (bits != 2 && bits != 4) {
        return false;
    }
    const int qmax = (1 << (bits - 1)) - 1;
    const int offset = qmax;
    size_t idx = 0;
    for (size_t i = 0; i < packed_bytes && idx < out_elems; ++i) {
        uint8_t byte = packed[i];
        if (bits == 4) {
            uint8_t v0 = byte & 0x0F;
            uint8_t v1 = (byte >> 4) & 0x0F;
            int8_t s0 = static_cast<int8_t>(static_cast<int>(v0) - offset);
            int8_t s1 = static_cast<int8_t>(static_cast<int>(v1) - offset);
            out[idx++] = s0;
            if (idx < out_elems) {
                out[idx++] = s1;
            }
        } else {
            uint8_t v0 = byte & 0x03;
            uint8_t v1 = (byte >> 2) & 0x03;
            uint8_t v2 = (byte >> 4) & 0x03;
            uint8_t v3 = (byte >> 6) & 0x03;
            int8_t s0 = static_cast<int8_t>(static_cast<int>(v0) - offset);
            int8_t s1 = static_cast<int8_t>(static_cast<int>(v1) - offset);
            int8_t s2 = static_cast<int8_t>(static_cast<int>(v2) - offset);
            int8_t s3 = static_cast<int8_t>(static_cast<int>(v3) - offset);
            out[idx++] = s0;
            if (idx < out_elems) out[idx++] = s1;
            if (idx < out_elems) out[idx++] = s2;
            if (idx < out_elems) out[idx++] = s3;
        }
    }
    return idx >= out_elems;
}

inline bool uhd_unpack_lowbit_from_bin(
    const UhdBinBuffer& bin,
    uint64_t offset,
    uint64_t nbytes,
    int bits,
    const int32_t* shape,
    size_t ndim,
    std::vector<int8_t>& out
) {
    size_t elems = uhd_num_elements(shape, ndim);
    out.resize(elems);
    const uint8_t* packed = bin.ptr(offset, nbytes);
    if (!packed) {
        return false;
    }
    return uhd_unpack_lowbit(packed, static_cast<size_t>(nbytes), bits, out.data(), elems);
}
