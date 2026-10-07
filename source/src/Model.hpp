// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// A Whisper network as read from its file (.nll): the hyperparameters,
// the special tokens, the mel filters, the vocabulary and the tensors.

#include <cstdint>
#include <map>
#include <string>
#include <vector>

namespace wsp {

struct Tensor {
    std::vector<int> shape;
    std::vector<float> data;    // float32 in memory (f32 or f16 in the file); empty if q is used
    // Quantized weights (types 2, 3): w[r][c] = q[r·cols + c]·scale[r·groups + c/group],
    // q within ±(2^(bits−1) − 1); group = cols: one scale per row.
    std::vector<int8_t> q;
    std::vector<float> scale;
    int bits = 8, group = 0;
    bool packed = false;        // a weight matrix in panels (Ops.hpp), once bound
};

struct Hparams {
    int nMels = 0, audioCtx = 0, audioState = 0, audioHeads = 0, audioLayers = 0;
    int textCtx = 0, textState = 0, textHeads = 0, textLayers = 0, vocab = 0;
};

struct Model {
    Hparams hp;
    int eot = 0, sot = 0, transcribe = 0, translate = 0, noTimestamps = 0, timestampBegin = 0;
    std::map<std::string, int> lang;            // code → token
    std::vector<int> suppress, beginSuppress;   // tokens never drawn / not drawn first
    std::vector<float> melFilters;              // nMels × 201
    std::vector<std::string> pieces;            // token → its bytes
    std::map<std::string, Tensor> tensors;

    bool load(const std::string& path, std::string* err);
    /** The tensor of that name, with that many values; null if missing. */
    const float* get(const std::string& name, size_t count) const;
};

/** IEEE half precision, as type-4 scales are stored. */
float HalfToFloat(uint16_t h);


}  // namespace wsp
