// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Transcribe.hpp"

#include <algorithm>

#include "Mel.hpp"
#include "Segment.hpp"

namespace wsp {

namespace {

constexpr int kPos = 2 * kHop;   // samples per encoder position (20 ms)

}  // namespace

std::vector<int> DecodeWindow(const Whisper& net, Pool& pool, const float* pcm, int samples, int langToken,
                              const WindowRule& rule, const DecodeOptions& opt, DecodeInfo* info, bool* redone,
                              int extraMargin) {
    const Model& m = net.model();
    const int n = std::min(samples, kChunk);
    std::vector<float> mel, audio;
    LogMel(pool, pcm, n, m.melFilters, m.hp.nMels, mel);
    // As measured: the audio's whole positions, one more, the margin.
    const int ctx = std::max(rule.minCtx, n / kPos + 1 + rule.margin + extraMargin);
    DecodeInfo local;
    DecodeInfo& in = info ? *info : local;
    if (redone) *redone = false;
    if (ctx < m.hp.audioCtx) {
        in = DecodeInfo();
        net.encode(pool, mel, audio, ctx);
        std::vector<int> toks = net.decode(pool, audio, langToken, nullptr, nullptr, {}, &in, &opt);
        const LoopSigns g = Signs(toks, in, static_cast<double>(n) / kSampleRate);
        if (!(g.limit || g.repeat > rule.maxRepeat || g.perSecond > rule.maxPerSecond)) return toks;
        if (redone) *redone = true;
    }
    in = DecodeInfo();
    net.encode(pool, mel, audio);
    return net.decode(pool, audio, langToken, nullptr, nullptr, {}, &in, &opt);
}

LongResult TranscribeLong(const Whisper& net, Pool& pool, const float* pcm, size_t samples, const LongOptions& opt) {
    LongResult r;
    int lang = opt.langToken;
    const double total = static_cast<double>(samples) / kSampleRate;
    auto emit = [&](double t0, double t1, const std::vector<int>& toks, int tokLang) {
        TimedSegment s;
        s.t0 = t0;
        s.t1 = std::max(t0, t1);
        s.text = net.text(toks);
        s.lang = tokLang;
        if (s.text.find_first_not_of(' ') == std::string::npos) return;
        r.segments.push_back(s);
        if (opt.onSegment) opt.onSegment(r.segments.back());
    };
    auto window = [&](size_t start, int n, const DecodeOptions& o, int extraMargin, DecodeInfo& info) {
        bool again = false;
        const int want = opt.detectEachWindow ? -1 : lang;
        std::vector<int> toks = DecodeWindow(net, pool, pcm + start, n, want, opt.rule, o, &info, &again, extraMargin);
        ++r.windows;
        r.redone += again;
        if (lang < 0 || r.lang < 0) {
            r.lang = info.lang;
            r.langProb = info.langProbs.empty() ? 0.0f : info.langProbs[0].second;
        }
        if (lang < 0 && !opt.detectEachWindow) lang = info.lang;
        return toks;
    };
    if (samples == 0) return r;

    // Pieces cut at pauses (one piece if the audio fits a window).
    std::vector<std::pair<size_t, size_t>> pieces;
    if (samples <= static_cast<size_t>(kChunk)) {
        pieces.push_back({0, samples});
    } else {
        Cutter c;
        c.push(pcm, static_cast<int>(std::min<size_t>(samples, 0x7fffffff)));
        size_t start = 0;
        for (bool fin = false;;) {
            const long long cut = c.next(fin);
            if (cut < 0) {
                if (fin) break;
                fin = true;
                continue;
            }
            if (static_cast<size_t>(cut) > start) pieces.push_back({start, static_cast<size_t>(cut)});
            start = static_cast<size_t>(cut);
            if (start >= samples) break;
        }
    }
    const DecodeOptions o;
    for (const auto& [a, b] : pieces) {
        const int n = static_cast<int>(std::min<size_t>(b - a, kChunk));
        if (n > kSampleRate / 10) {
            DecodeInfo info;
            // A piece cut from longer audio ends inside a pause: 5 s past it.
            const std::vector<int> toks = window(a, n, o, samples > static_cast<size_t>(kChunk) ? 200 : 0, info);
            emit(static_cast<double>(a) / kSampleRate, static_cast<double>(b) / kSampleRate, toks, info.lang);
        }
        if (opt.progress) opt.progress(static_cast<double>(b) / kSampleRate, total);
    }
    return r;
}

}  // namespace wsp
