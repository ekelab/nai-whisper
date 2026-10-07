// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// Whisper's computation on a Model: the encoder (two convolutions, then
// pre-norm attention blocks) and the decoder (self-attention with a cache,
// cross-attention to the encoder's output), greedy decoding.

#include <functional>
#include <utility>
#include <string>
#include <vector>

#include "Model.hpp"
#include "Ops.hpp"

namespace wsp {

/** What a greedy decoding leaves besides the tokens. */
struct DecodeInfo {
    std::vector<float> logprob;   // log P of each chosen token (the end included)
    bool hitLimit = false;        // stopped by the length limit, not by the end token
    int lang = -1;                // the language token (detected when asked for: langToken < 0)
    std::vector<std::pair<int, float>> langProbs;   // detection: language token -> P, most likely first
};

/** Signs that a decoding went into a loop, from the tokens only: the share
 *  of tokens inside a 4-gram seen before, tokens per second of audio, the
 *  mean log-probability, the length limit reached. */
struct LoopSigns {
    double repeat = 0, perSecond = 0, avgLogprob = 0;
    bool limit = false;
};
LoopSigns Signs(const std::vector<int>& tokens, const DecodeInfo& info, double seconds);

/** How to decode. detectOnly: stop after choosing the language. */
struct DecodeOptions {
    bool detectOnly = false;
};

class Whisper {
public:
    /** Binds the tensors of m (which must outlive this), packing its weight
     *  matrices in place (Ops.hpp); false and err if some are missing or of
     *  the wrong size. */
    bool bind(Model& m, std::string* err);

    /** mel (nMels × 3000) → the encoder's output (T × audioState): T =
     *  audioCtx (the whole 30 s, as the model was trained), or ctx positions
     *  - the first 2·ctx frames only (faster for short audio; not exact). */
    void encode(Pool& pool, const std::vector<float>& mel, std::vector<float>& out, int ctx = 0) const;


    /** Greedy decoding after the prompt (start, language, task, no
     *  timestamps): the drawn tokens, the end token excluded. langToken < 0:
     *  the language is detected first (info->lang).
     *  If firstLogits, the logits of the first drawn token (before
     *  suppression) go there. forced: the tokens are given instead of drawn
     *  (teacher forcing); onLogits sees the logits before each token (and
     *  before the end). */
    std::vector<int> decode(Pool& pool, const std::vector<float>& audio, int langToken,
                            std::vector<float>* firstLogits = nullptr, const std::vector<int>* forced = nullptr,
                            const std::function<void(const std::vector<float>&)>& onLogits = {},
                            DecodeInfo* info = nullptr, const DecodeOptions* opt = nullptr) const;

    /** The text of tokens (special ones skipped). */
    std::string text(const std::vector<int>& tokens) const;

    const Model& model() const { return *m_; }

private:
    struct Lin {
        std::string name;   // the weight tensor's
        Weight w;
        const float* b = nullptr;
        int out = 0, in = 0;
    };
    struct Norm {
        const float* g = nullptr;
        const float* b = nullptr;
    };
    struct Layer {
        Norm ln1, ln2, ln3;
        Lin q, k, v, o;        // self-attention
        Lin cq, ck, cv, co;    // cross-attention (decoder)
        Lin fc1, fc2;
    };
    const Model* m_ = nullptr;
    Weight conv1w_;
    const float* conv1b_ = nullptr;
    Weight conv2w_;
    const float* conv2b_ = nullptr;
    const float* encPos_ = nullptr;
    const float* decPos_ = nullptr;
    Weight embed_;
    Norm encLn_, decLn_;
    std::vector<Layer> enc_, dec_;
    // encode = the front, every layer, the end.
    void encodeFront(Pool& pool, const std::vector<float>& mel, std::vector<float>& x, int ctx) const;
    void encodeLayer(Pool& pool, int layer, std::vector<float>& x) const;
    void encodeEnd(Pool& pool, std::vector<float>& x) const;
};

}  // namespace wsp
