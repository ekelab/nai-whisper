// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Mel.hpp"

#include <algorithm>
#include <cmath>
#include <complex>

namespace wsp {

void LogMel(Pool& pool, const float* pcm, int samples, const std::vector<float>& filters, int nMels,
            std::vector<float>& mel) {
    constexpr int kBins = kFft / 2 + 1;
    constexpr double kPi = 3.14159265358979323846;
    // The signal padded to 30 s, then by kFft/2 on each side by reflection
    // (the frames are centred on t·kHop).
    std::vector<float> x(kChunk + kFft, 0.0f);
    const int n = std::min(samples, kChunk);
    std::copy_n(pcm, n, &x[kFft / 2]);
    for (int i = 1; i <= kFft / 2; ++i) {
        x[kFft / 2 - i] = x[kFft / 2 + i];
        x[kFft / 2 + kChunk - 1 + i] = x[kFft / 2 + kChunk - 1 - i];
    }
    // A mixed-radix FFT of 400 = 4·4·5·5 points (decimation in time, a
    // generic butterfly per radix), on the windowed frame.
    using C = std::complex<double>;
    struct Tables {
        std::vector<C> tw;
        std::vector<double> win;
        Tables() : tw(kFft), win(kFft) {
            for (int j = 0; j < kFft; ++j) tw[j] = std::polar(1.0, -2 * kPi * j / kFft);
            for (int t = 0; t < kFft; ++t) win[t] = 0.5 - 0.5 * std::cos(2 * kPi * t / kFft);   // periodic Hann
        }
    };
    static const Tables tables;   // built once, thread-safe
    static constexpr int kFactors[] = {4, 100, 4, 25, 5, 5, 5, 1};   // (radix, remaining length) pairs
    struct Fft {
        const std::vector<C>& tw;
        void work(C* out, const C* f, int fstride, const int* factors) const {
            const int p = factors[0], m = factors[1];
            C* const beg = out;
            if (m == 1) {
                for (int i = 0; i < p; ++i) out[i] = f[i * fstride];
            } else {
                for (int i = 0; i < p; ++i, out += m, f += fstride) work(out, f, fstride * p, factors + 2);
            }
            out = beg;
            C scratch[5];
            for (int u = 0; u < m; ++u) {
                for (int q = 0, k = u; q < p; ++q, k += m) scratch[q] = out[k];
                for (int q1 = 0, k = u; q1 < p; ++q1, k += m) {
                    C v = scratch[0];
                    for (int q = 1, idx = 0; q < p; ++q) {
                        idx += fstride * k;
                        idx %= kFft;
                        v += scratch[q] * tw[idx];
                    }
                    out[k] = v;
                }
            }
        }
    };
    const Fft fft{tables.tw};
    const std::vector<double>& win = tables.win;
    mel.assign(static_cast<size_t>(nMels) * kFrames, 0.0f);
    constexpr int kPerTask = 50;
    pool.run(kFrames / kPerTask, [&](int task) {
        C frame[kFft], spec[kFft];
        double power[kBins];
        for (int f = task * kPerTask; f < (task + 1) * kPerTask; ++f) {
            for (int t = 0; t < kFft; ++t) frame[t] = x[static_cast<size_t>(f) * kHop + t] * win[t];
            fft.work(spec, frame, 1, kFactors);
            for (int k = 0; k < kBins; ++k) power[k] = std::norm(spec[k]);
            for (int m = 0; m < nMels; ++m) {
                const float* w = &filters[static_cast<size_t>(m) * kBins];
                double v = 0;
                for (int k = 0; k < kBins; ++k) v += w[k] * power[k];
                mel[static_cast<size_t>(m) * kFrames + f] = static_cast<float>(std::log10(std::max(v, 1e-10)));
            }
        }
    });
    const float mx = *std::max_element(mel.begin(), mel.end());
    for (float& v : mel) v = (std::max(v, mx - 8.0f) + 4.0f) / 4.0f;
}

}  // namespace wsp
