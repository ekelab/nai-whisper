// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// Recordings of any length to timed text, on top of Whisper:
//
//   a window   up to 30 s of audio, one decoding. The encoder sees the audio
//              and 1 s more but at least 15 s (faster than the whole 30 s it
//              was trained on); if the decoding looks broken (Signs: repeats,
//              too many tokens per second, the length limit) it is done again
//              with the whole window.
//   long audio cut at pauses into pieces of 5-15 s (Segment.hpp), one window
//              each; a segment per piece. Audio of 30 s or less is one piece.

#include <functional>
#include <string>
#include <vector>

#include "Whisper.hpp"

namespace wsp {

/** The short window's bounds (chosen on development sets; the numbers in
 *  the public documentation). */
struct WindowRule {
    int margin = 50;          // positions of 20 ms past the audio
    int minCtx = 750;         // the window is never under 15 s
    double maxRepeat = 0.282; // share of tokens inside a repeated 4-gram
    double maxPerSecond = 6.03;
};

struct TimedSegment {
    double t0 = 0, t1 = 0;    // seconds from the start of the audio
    std::string text;
    int lang = -1;            // its language token
};

struct LongOptions {
    int langToken = -1;           // < 0: detected on the first window
    bool detectEachWindow = false;   // the language again for every window (mixed-language audio)
    WindowRule rule;
    std::function<void(double done, double total)> progress;   // seconds of audio
    std::function<void(const TimedSegment&)> onSegment;     // as each is final
};

struct LongResult {
    std::vector<TimedSegment> segments;
    int lang = -1;                // the (first) detected or given language token
    float langProb = 0;           // its probability when detected
    int windows = 0, redone = 0;  // windows decoded; of them done again whole
};

/** One window of up to 30 s (pcm from its start): tokens of its decoding
 *  under the short-window rule; info and whether it was done again whole. */
std::vector<int> DecodeWindow(const Whisper& net, Pool& pool, const float* pcm, int samples, int langToken,
                              const WindowRule& rule, const DecodeOptions& opt, DecodeInfo* info, bool* redone,
                              int extraMargin = 0);

LongResult TranscribeLong(const Whisper& net, Pool& pool, const float* pcm, size_t samples, const LongOptions& opt);

}  // namespace wsp
