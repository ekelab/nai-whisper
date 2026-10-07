// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Segment.hpp"

#include <algorithm>
#include <cmath>

#include "Mel.hpp"

namespace wsp {
namespace {
constexpr int kFrame = kSampleRate / 50;   // 20 ms
}

void Cutter::push(const float* pcm, int n) {
    heard_ += n;
    partial_.insert(partial_.end(), pcm, pcm + n);
    size_t used = 0;
    for (; used + kFrame <= partial_.size(); used += kFrame) {
        double s = 0;
        for (int i = 0; i < kFrame; ++i) s += static_cast<double>(partial_[used + i]) * partial_[used + i];
        energy_.push_back(static_cast<float>(10 * std::log10(s / kFrame + 1e-12)));
    }
    partial_.erase(partial_.begin(), partial_.begin() + static_cast<std::ptrdiff_t>(used));
}

double Cutter::threshold() const {
    // Over the last 60 s heard: floor = 10th percentile, speech = 90th.
    const size_t n = energy_.size(), from = n > 3000 ? n - 3000 : 0;
    std::vector<float> e(energy_.begin() + static_cast<std::ptrdiff_t>(from), energy_.end());
    if (e.size() < 10) return -1e9;
    std::sort(e.begin(), e.end());
    const double lo = e[e.size() / 10], hi = e[e.size() * 9 / 10];
    return lo + std::max(6.0, 0.3 * (hi - lo));
}

long long Cutter::next(bool final) {
    const long long f0 = start_ / kFrame, f1 = static_cast<long long>(energy_.size());
    const long long minF = static_cast<long long>(o_.minSec * 50), maxF = static_cast<long long>(o_.maxSec * 50);
    const long long pauseF = std::max<long long>(1, static_cast<long long>(o_.pauseSec * 50));
    const double thr = threshold();
    // The first pause after minSec.
    long long run = 0;
    for (long long f = f0; f < f1 && f - f0 < maxF; ++f) {
        run = energy_[f] < thr ? run + 1 : 0;
        if (run >= pauseF && f + 1 - run - f0 >= minF) {
            const long long cut = (f + 1 - run + f + 1) / 2;   // the middle of the pause so far
            start_ = cut * kFrame;
            return start_;
        }
    }
    if (f1 - f0 >= maxF) {   // no pause in time: the quietest frame of the last 5 s
        long long best = f0 + maxF - 1;
        for (long long f = std::max(f0 + minF, f0 + maxF - 250); f < f0 + maxF; ++f)
            if (energy_[f] < energy_[best]) best = f;
        start_ = (best + 1) * kFrame;
        return start_;
    }
    if (final && heard_ > start_) {
        start_ = heard_;
        return start_;
    }
    return -1;
}

}  // namespace wsp
