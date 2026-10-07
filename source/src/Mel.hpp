// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// The log-mel spectrogram Whisper reads: 30 s of 16 kHz audio (cut or padded
// with zeros) → nMels × 3000 values.

#include <vector>

#include "Ops.hpp"

namespace wsp {

constexpr int kSampleRate = 16000, kChunk = 30 * kSampleRate, kFft = 400, kHop = 160, kFrames = 3000;

/** mel: nMels × kFrames (a row per mel band). filters: nMels × 201. */
void LogMel(Pool& pool, const float* pcm, int samples, const std::vector<float>& filters, int nMels,
            std::vector<float>& mel);

}  // namespace wsp
