// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Model.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>

namespace wsp {
namespace {

/** Reads the file piece by piece: a large network is never held twice. */
struct Reader {
    std::ifstream f;
    bool ok = true;

    bool read(void* p, size_t n) {
        if (ok && n > 0 && !f.read(static_cast<char*>(p), static_cast<std::streamsize>(n))) ok = false;
        return ok;
    }
    uint32_t u32() {
        uint32_t v = 0;
        read(&v, 4);
        return v;
    }
    uint16_t u16() {
        uint16_t v = 0;
        read(&v, 2);
        return v;
    }
    uint8_t u8() {
        uint8_t v = 0;
        read(&v, 1);
        return v;
    }
    std::string bytes(size_t n) {
        std::string s(n, '\0');
        read(s.data(), n);
        return s;
    }
};

}  // namespace


float HalfToFloat(uint16_t h) {
    const uint32_t sign = static_cast<uint32_t>(h & 0x8000) << 16;
    uint32_t exp = (h >> 10) & 0x1f, man = h & 0x3ff, bits;
    if (exp == 0) {
        if (man == 0) {
            bits = sign;
        } else {   // subnormal: normalise
            exp = 127 - 15 + 1;
            while (!(man & 0x400)) man <<= 1, --exp;
            bits = sign | (exp << 23) | ((man & 0x3ff) << 13);
        }
    } else if (exp == 31) {
        bits = sign | 0x7f800000 | (man << 13);
    } else {
        bits = sign | ((exp - 15 + 127) << 23) | (man << 13);
    }
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}


bool Model::load(const std::string& path, std::string* err) {
    auto fail = [&](const std::string& why) {
        if (err) *err = path + ": " + why;
        return false;
    };
    Reader r;
    r.f.open(path, std::ios::binary);
    if (!r.f) return fail("cannot open");
    char magic[4] = {};
    if (!r.read(magic, 4) || std::memcmp(magic, "WSP1", 4) != 0) return fail("not a WSP1 file");
    int* h[] = {&hp.nMels, &hp.audioCtx, &hp.audioState, &hp.audioHeads, &hp.audioLayers,
                &hp.textCtx, &hp.textState, &hp.textHeads, &hp.textLayers, &hp.vocab};
    for (int* v : h) *v = static_cast<int>(r.u32());
    int* s[] = {&eot, &sot, &transcribe, &translate, &noTimestamps, &timestampBegin};
    for (int* v : s) *v = static_cast<int>(r.u32());
    const uint32_t nLang = r.u32();
    for (uint32_t i = 0; i < nLang && r.ok; ++i) {
        const std::string code = r.bytes(r.u8());
        lang[code] = static_cast<int>(r.u32());
    }
    for (std::vector<int>* v : {&suppress, &beginSuppress}) {
        const uint32_t n = r.u32();
        if (!r.ok || n > (1u << 20)) return fail("bad header");
        v->resize(n);
        for (int& t : *v) t = static_cast<int>(r.u32());
    }
    if (!r.ok || hp.nMels <= 0 || hp.nMels > 1024 || hp.vocab <= 0 || hp.vocab > (1 << 24)) return fail("bad header");
    melFilters.resize(static_cast<size_t>(hp.nMels) * 201);
    r.read(melFilters.data(), melFilters.size() * 4);
    pieces.resize(hp.vocab);
    for (std::string& p : pieces) p = r.bytes(r.u16());
    const uint32_t nt = r.u32();
    std::vector<uint16_t> half;
    for (uint32_t i = 0; i < nt && r.ok; ++i) {
        const std::string name = r.bytes(r.u16());
        const int type = r.u8(), nd = r.u8();
        if (nd > 4) return fail("bad tensor " + name);
        Tensor t;
        size_t count = 1;
        for (int k = 0; k < nd; ++k) {
            t.shape.push_back(static_cast<int>(r.u32()));
            count *= static_cast<size_t>(t.shape.back());
        }
        if (!r.ok || count > (size_t{1} << 33)) return fail("bad tensor " + name);
        if (type == 2 || type == 3 || type == 4) {   // quantized (Tensor); 4: as 3, the scales in f16
            if (nd != 2 || t.shape[0] <= 0 || t.shape[1] <= 0) return fail("bad quantized tensor " + name);
            const int cols = t.shape[1];
            t.bits = 8;
            t.group = cols;
            if (type != 2) {
                t.bits = r.u8();
                t.group = static_cast<int>(r.u32());
                if (t.bits < 2 || t.bits > 8 || t.group <= 0 || t.group > cols) return fail("bad quantized tensor " + name);
            }
            const size_t groups = (cols + t.group - 1) / t.group;
            t.scale.resize(static_cast<size_t>(t.shape[0]) * groups);
            t.q.resize(count);
            if (type == 4) {
                half.resize(t.scale.size());
                r.read(half.data(), half.size() * 2);
                for (size_t k = 0; k < half.size(); ++k) t.scale[k] = HalfToFloat(half[k]);
            } else {
                r.read(t.scale.data(), t.scale.size() * 4);
            }
            if (type == 2) {
                r.read(t.q.data(), count);
            } else {   // bits-bit fields, offset by 2^(bits−1), least significant bit first
                std::vector<uint8_t> packed((count * t.bits + 7) / 8);
                r.read(packed.data(), packed.size());
                const int off = 1 << (t.bits - 1), mask = (1 << t.bits) - 1;
                for (size_t i = 0, bit = 0; i < count; ++i, bit += t.bits) {
                    const uint32_t w = packed[bit / 8] | (bit / 8 + 1 < packed.size() ? packed[bit / 8 + 1] << 8 : 0);
                    t.q[i] = static_cast<int8_t>(static_cast<int>((w >> (bit % 8)) & mask) - off);
                }
            }
            tensors[name] = std::move(t);
            continue;
        }
        t.data.resize(count);
        if (type == 0) {
            r.read(t.data.data(), count * 4);
        } else if (type == 1) {
            half.resize(count);
            r.read(half.data(), count * 2);
            for (size_t k = 0; k < count; ++k) t.data[k] = HalfToFloat(half[k]);
        } else {
            return fail("unknown tensor type in " + name);
        }
        tensors[name] = std::move(t);
    }
    if (!r.ok) return fail("truncated");
    if (r.f.peek() != std::char_traits<char>::eof()) return fail("trailing bytes");
    return true;
}


const float* Model::get(const std::string& name, size_t count) const {
    auto it = tensors.find(name);
    if (it == tensors.end() || it->second.data.size() != count) return nullptr;
    return it->second.data.data();
}

}  // namespace wsp
