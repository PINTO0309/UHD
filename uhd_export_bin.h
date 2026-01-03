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
