// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Ops.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <cstring>

namespace wsp {

Pool::Pool(int threads) {
    for (int i = 1; i < std::max(threads, 1); ++i) workers_.emplace_back([this] { work(); });
}

Pool::~Pool() {
    {
        std::lock_guard<std::mutex> lock(mu_);
        stop_ = true;
    }
    start_.notify_all();
    for (std::thread& t : workers_) t.join();
}

void Pool::drain() {
    // Takes tasks until none are left; mu_ is held on entry and on exit.
    while (next_ < n_) {
        const int i = next_++;
        const std::function<void(int)>& f = *job_;
        mu_.unlock();
        f(i);
        mu_.lock();
        if (++finished_ == n_) done_.notify_all();
    }
}

void Pool::work() {
    long long seen = 0;
    std::unique_lock<std::mutex> lock(mu_);
    for (;;) {
        start_.wait(lock, [&] { return stop_ || epoch_ != seen; });
        if (stop_) return;
        seen = epoch_;
        drain();
    }
}

void Pool::run(int n, const std::function<void(int)>& f) {
    if (n <= 0) return;
    if (workers_.empty() || n == 1) {
        for (int i = 0; i < n; ++i) f(i);
        return;
    }
    std::unique_lock<std::mutex> lock(mu_);
    job_ = &f;
    n_ = n;
    next_ = 0;
    finished_ = 0;
    ++epoch_;
    start_.notify_all();
    drain();
    done_.wait(lock, [&] { return finished_ == n_; });
    job_ = nullptr;
    n_ = 0;
}

float Dot(const float* a, const float* b, int n) {
    float s[16] = {};
    int i = 0;
    for (; i + 16 <= n; i += 16)
        for (int j = 0; j < 16; ++j) s[j] += a[i + j] * b[i + j];
    float t = 0;
    for (; i < n; ++i) t += a[i] * b[i];
    for (int j = 0; j < 16; ++j) t += s[j];
    return t;
}

size_t PackedSize(int out, int in) {
    return static_cast<size_t>((out + kPanel - 1) / kPanel) * kPanel * in;
}

void PackRows(const float* W, int out, int in, int ld, float* P) {
    const int panels = (out + kPanel - 1) / kPanel;
    for (int p = 0; p < panels; ++p)
        for (int k = 0; k < in; ++k)
            for (int j = 0; j < kPanel; ++j) {
                const int o = p * kPanel + j;
                P[(static_cast<size_t>(p) * in + k) * kPanel + j] = o < out ? W[static_cast<size_t>(o) * ld + k] : 0.0f;
            }
}

void PackCols(const float* V, int k, int out, int ld, float* P) {
    const int panels = (out + kPanel - 1) / kPanel;
    for (int p = 0; p < panels; ++p)
        for (int r = 0; r < k; ++r)
            for (int j = 0; j < kPanel; ++j) {
                const int o = p * kPanel + j;
                P[(static_cast<size_t>(p) * k + r) * kPanel + j] = o < out ? V[static_cast<size_t>(r) * ld + o] : 0.0f;
            }
}

namespace {

typedef float v8 __attribute__((vector_size(32)));

inline v8 Load8(const float* p) {
    v8 v;
    std::memcpy(&v, p, sizeof v);
    return v;
}

/** R rows of X (k0 … k1 of each) by one panel: Y (nc valid outputs) set
 *  (first: + b) or added to. */
template <int R>
void Kernel(const float* x, int ldx, const float* p, int k0, int k1, float* y, int ldy, int nc, const float* b,
            bool first) {
    v8 c0[R], c1[R];
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) c0[r] = c1[r] = v8{};
    for (int k = k0; k < k1; ++k) {
        const v8 w0 = Load8(p + static_cast<size_t>(k) * kPanel), w1 = Load8(p + static_cast<size_t>(k) * kPanel + 8);
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) {
            const float a = x[static_cast<size_t>(r) * ldx + k];
            c0[r] += a * w0;
            c1[r] += a * w1;
        }
    }
    for (int r = 0; r < R; ++r) {
        float t[kPanel];
        std::memcpy(t, &c0[r], sizeof c0[r]);
        std::memcpy(t + 8, &c1[r], sizeof c1[r]);
        float* yr = y + static_cast<size_t>(r) * ldy;
        if (first)
            for (int j = 0; j < nc; ++j) yr[j] = t[j] + (b ? b[j] : 0.0f);
        else
            for (int j = 0; j < nc; ++j) yr[j] += t[j];
    }
}

typedef int8_t i8x8 __attribute__((vector_size(8)));

inline v8 LoadQ8(const int8_t* p) {
    i8x8 v;
    std::memcpy(&v, p, sizeof v);
    return __builtin_convertvector(v, v8);
}

/** Kernel with integer panels: the sums of each group of inputs are scaled
 *  by the group's scales (sc: the panel's, kPanel per group). */
template <int R>
void KernelQ(const float* x, int ldx, const int8_t* p, int k0, int k1, float* y, int ldy, int nc, const float* b,
             const float* sc, int group, bool first) {
    v8 t0[R], t1[R];
#pragma GCC unroll 8
    for (int r = 0; r < R; ++r) t0[r] = t1[r] = v8{};
    for (int k = k0; k < k1;) {
        const int g = k / group, kend = std::min(k1, (g + 1) * group);
        v8 c0[R], c1[R];
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) c0[r] = c1[r] = v8{};
        for (; k < kend; ++k) {
            const v8 w0 = LoadQ8(p + static_cast<size_t>(k) * kPanel), w1 = LoadQ8(p + static_cast<size_t>(k) * kPanel + 8);
#pragma GCC unroll 8
            for (int r = 0; r < R; ++r) {
                const float a = x[static_cast<size_t>(r) * ldx + k];
                c0[r] += a * w0;
                c1[r] += a * w1;
            }
        }
        const v8 s0 = Load8(sc + static_cast<size_t>(g) * kPanel), s1 = Load8(sc + static_cast<size_t>(g) * kPanel + 8);
#pragma GCC unroll 8
        for (int r = 0; r < R; ++r) {
            t0[r] += c0[r] * s0;
            t1[r] += c1[r] * s1;
        }
    }
    for (int r = 0; r < R; ++r) {
        float t[kPanel];
        std::memcpy(t, &t0[r], sizeof t0[r]);
        std::memcpy(t + 8, &t1[r], sizeof t1[r]);
        float* yr = y + static_cast<size_t>(r) * ldy;
        if (first)
            for (int j = 0; j < nc; ++j) yr[j] = t[j] + (b ? b[j] : 0.0f);
        else
            for (int j = 0; j < nc; ++j) yr[j] += t[j];
    }
}

void RowsQ(int R, const float* x, int ldx, const int8_t* p, int k0, int k1, float* y, int ldy, int nc, const float* b,
           const float* sc, int group, bool first) {
    switch (R) {
        case 1: KernelQ<1>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
        case 2: KernelQ<2>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
        case 3: KernelQ<3>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
        case 4: KernelQ<4>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
        case 5: KernelQ<5>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
        default: KernelQ<6>(x, ldx, p, k0, k1, y, ldy, nc, b, sc, group, first); break;
    }
}

void Rows(int R, const float* x, int ldx, const float* p, int k0, int k1, float* y, int ldy, int nc, const float* b,
          bool first) {
    switch (R) {
        case 1: Kernel<1>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
        case 2: Kernel<2>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
        case 3: Kernel<3>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
        case 4: Kernel<4>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
        case 5: Kernel<5>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
        default: Kernel<6>(x, ldx, p, k0, k1, y, ldy, nc, b, first); break;
    }
}

}  // namespace

void Gemm(Pool* pool, const float* X, int n, int ldx, const float* P, int in, int out, const float* b, float* Y,
          int ldy, bool add) {
    Weight w;
    w.f = P;
    Gemm(pool, X, n, ldx, w, in, out, b, Y, ldy, add);
}

void PackScales(const float* s, int out, int groups, float* P) {
    const int panels = (out + kPanel - 1) / kPanel;
    for (int p = 0; p < panels; ++p)
        for (int g = 0; g < groups; ++g)
            for (int j = 0; j < kPanel; ++j) {
                const int o = p * kPanel + j;
                P[(static_cast<size_t>(p) * groups + g) * kPanel + j] = o < out ? s[static_cast<size_t>(o) * groups + g] : 0.0f;
            }
}

void PackRowsQ(const int8_t* W, int out, int in, int8_t* P) {
    const int panels = (out + kPanel - 1) / kPanel;
    for (int p = 0; p < panels; ++p)
        for (int k = 0; k < in; ++k)
            for (int j = 0; j < kPanel; ++j) {
                const int o = p * kPanel + j;
                P[(static_cast<size_t>(p) * in + k) * kPanel + j] = o < out ? W[static_cast<size_t>(o) * in + k] : 0;
            }
}

void Gemm(Pool* pool, const float* X, int n, int ldx, const Weight& W, int in, int out, const float* b, float* Y,
          int ldy, bool add) {
    // Blocks: kRowBlock rows × kGroup panels per task; the inputs in slices
    // of kSlice, so that a panel's slice (32 KB) and the rows' slice stay in
    // the cache while the block is done.
    constexpr int kRowBlock = 48, kGroup = 8, kSlice = 512;
    const int panels = (out + kPanel - 1) / kPanel;
    const int rb = (n + kRowBlock - 1) / kRowBlock, gb = (panels + kGroup - 1) / kGroup;
    auto task = [&](int t) {
        alignas(32) float buf[kSlice * kPanel];   // a panel's slice of int8 weights as float
        const int r0 = (t / gb) * kRowBlock, r1 = std::min(n, r0 + kRowBlock);
        const int p0 = (t % gb) * kGroup, p1 = std::min(panels, p0 + kGroup);
        for (int k0 = 0; k0 < in; k0 += kSlice) {
            const int k1 = std::min(in, k0 + kSlice);
            for (int p = p0; p < p1; ++p) {
                const size_t off = static_cast<size_t>(p) * in * kPanel;
                const int nc = std::min(kPanel, out - p * kPanel);
                const float* bp = b ? b + p * kPanel : nullptr;
                if (W.q && r1 - r0 >= 12) {
                    // Many rows: the panel's slice to float once (32 KB, stays
                    // in the cache), then the float kernel for all of them.
                    const int8_t* q = W.q + off + static_cast<size_t>(k0) * kPanel;
                    const float* scp = W.scale + static_cast<size_t>(p) * W.groups * kPanel;
                    for (int k = 0; k < k1 - k0; ++k) {
                        const float* sc = scp + static_cast<size_t>((k0 + k) / W.group) * kPanel;
                        for (int j = 0; j < kPanel; ++j) buf[k * kPanel + j] = q[k * kPanel + j] * sc[j];
                    }
                    for (int r = r0; r < r1; r += 6)
                        Rows(std::min(6, r1 - r), X + static_cast<size_t>(r) * ldx + k0, ldx, buf, 0, k1 - k0,
                             Y + static_cast<size_t>(r) * ldy + p * kPanel, ldy, nc, bp, k0 == 0 && !add);
                    continue;
                }
                for (int r = r0; r < r1; r += 6) {
                    const float* x = X + static_cast<size_t>(r) * ldx;
                    float* y = Y + static_cast<size_t>(r) * ldy + p * kPanel;
                    if (W.q)
                        RowsQ(std::min(6, r1 - r), x, ldx, W.q + off, k0, k1, y, ldy, nc, bp,
                              W.scale + static_cast<size_t>(p) * W.groups * kPanel, W.group, k0 == 0 && !add);
                    else
                        Rows(std::min(6, r1 - r), x, ldx, W.f + off, k0, k1, y, ldy, nc, bp, k0 == 0 && !add);
                }
            }
        }
    };
    if (!pool || static_cast<double>(n) * out * in < 2e5) {
        for (int t = 0; t < rb * gb; ++t) task(t);
    } else {
        pool->run(rb * gb, task);
    }
}

void LayerNorm(Pool& pool, float* x, int n, int d, const float* g, const float* b) {
    auto row = [&](int r) {
        float* v = x + static_cast<size_t>(r) * d;
        double m = 0, s = 0;
        for (int i = 0; i < d; ++i) m += v[i];
        m /= d;
        for (int i = 0; i < d; ++i) s += (v[i] - m) * (v[i] - m);
        const double inv = 1.0 / std::sqrt(s / d + 1e-5);
        for (int i = 0; i < d; ++i) v[i] = static_cast<float>((v[i] - m) * inv) * g[i] + b[i];
    };
    if (n < 64) {
        for (int r = 0; r < n; ++r) row(r);
        return;
    }
    constexpr int kBlock = 32;
    pool.run((n + kBlock - 1) / kBlock, [&](int t) {
        for (int r = t * kBlock; r < std::min(n, (t + 1) * kBlock); ++r) row(r);
    });
}

void ExpInPlace(float* x, int n) {
    // Cephes expf: x = m·ln2 + r, |r| ≤ ln2/2; e^r by a polynomial; 2^m by
    // the exponent bits.
    for (int i = 0; i < n; ++i) {
        float v = x[i];
        v = v < -87.3f ? -87.3f : v;   // the callers pass x ≤ 0 (one clamp: two block vectorization)
        // m = floor(t) without std::floor (it blocks vectorization here).
        const float t = v * 1.44269504088896341f + 0.5f;
        int32_t mi = static_cast<int32_t>(t);
        mi -= static_cast<float>(mi) > t ? 1 : 0;
        const float m = static_cast<float>(mi);
        const float r = v - m * 0.693359375f + m * 2.12194440e-4f;
        float p = 1.9875691500e-4f;
        p = p * r + 1.3981999507e-3f;
        p = p * r + 8.3334519073e-3f;
        p = p * r + 4.1665795894e-2f;
        p = p * r + 1.6666665459e-1f;
        p = p * r + 5.0000001201e-1f;
        p = p * r * r + r + 1.0f;
        x[i] = p * std::bit_cast<float>((mi + 127) << 23);
    }
}

void Gelu(Pool& pool, float* x, size_t n) {
    // erf(z) = 1 − (a1·t + … + a5·t⁵)·e^(−z²), t = 1/(1 + p·|z|) (Abramowitz
    // and Stegun 7.1.26, |error| ≤ 1.5·10⁻⁷), with the sign of z.
    constexpr size_t kBlock = 4096;
    const int tasks = static_cast<int>((n + kBlock - 1) / kBlock);
    auto block = [&](int t) {
        float* v = x + static_cast<size_t>(t) * kBlock;
        const int m = static_cast<int>(std::min(kBlock, n - static_cast<size_t>(t) * kBlock));
        float e[kBlock];
        for (int i = 0; i < m; ++i) {
            const float z = v[i] * 0.70710678118654752f;
            e[i] = -z * z;
        }
        ExpInPlace(e, m);
        for (int i = 0; i < m; ++i) {
            const float z = v[i] * 0.70710678118654752f, az = std::fabs(z);
            const float t1 = 1.0f / (1.0f + 0.3275911f * az);
            const float poly =
                t1 * (0.254829592f + t1 * (-0.284496736f + t1 * (1.421413741f + t1 * (-1.453152027f + t1 * 1.061405429f))));
            const float erfAbs = 1.0f - poly * e[i];
            const float erf = z < 0 ? -erfAbs : erfAbs;
            v[i] = 0.5f * v[i] * (1.0f + erf);
        }
    };
    if (tasks <= 1) {
        for (int t = 0; t < tasks; ++t) block(t);
    } else {
        pool.run(tasks, block);
    }
}

void SoftmaxRow(float* s, int n, float scale) {
    // In lanes of 16, so that the loops vectorize (a max or a sum over one
    // variable does not).
    constexpr int kLanes = 16;
    float lane[kLanes];
    for (int c = 0; c < kLanes; ++c) lane[c] = -INFINITY;
    int j = 0;
    for (; j + kLanes <= n; j += kLanes)
        for (int c = 0; c < kLanes; ++c) {
            const float v = s[j + c] * scale;
            s[j + c] = v;
            lane[c] = v > lane[c] ? v : lane[c];
        }
    float mx = -INFINITY;
    for (; j < n; ++j) mx = std::max(mx, s[j] *= scale);
    for (int c = 0; c < kLanes; ++c) mx = std::max(mx, lane[c]);
    for (j = 0; j < n; ++j) s[j] -= mx;
    ExpInPlace(s, n);
    for (int c = 0; c < kLanes; ++c) lane[c] = 0;
    for (j = 0; j + kLanes <= n; j += kLanes)
        for (int c = 0; c < kLanes; ++c) lane[c] += s[j + c];
    double z = 0;
    for (; j < n; ++j) z += s[j];
    for (int c = 0; c < kLanes; ++c) z += lane[c];
    const float inv = static_cast<float>(1.0 / z);
    for (j = 0; j < n; ++j) s[j] *= inv;
}

void Attention(Pool& pool, const float* Q, int nq, const float* K, const float* V, int nk, int d, int H, bool causal,
               float* O) {
    const int dh = d / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    if (nq < 8) {   // a decoder step: one query, dot products straight
        pool.run(H, [&](int h) {
            std::vector<float> p(nk);
            for (int i = 0; i < nq; ++i) {
                const int n = causal ? i + (nk - nq) + 1 : nk;
                const float* q = Q + static_cast<size_t>(i) * d + h * dh;
                float mx = -INFINITY;
                for (int j = 0; j < n; ++j) {
                    p[j] = Dot(q, K + static_cast<size_t>(j) * d + h * dh, dh) * scale;
                    mx = std::max(mx, p[j]);
                }
                for (int j = 0; j < n; ++j) p[j] -= mx;
                ExpInPlace(p.data(), n);
                double z = 0;
                for (int j = 0; j < n; ++j) z += p[j];
                const float inv = static_cast<float>(1.0 / z);
                float* o = O + static_cast<size_t>(i) * d + h * dh;
                std::fill_n(o, dh, 0.0f);
                for (int j = 0; j < n; ++j) {
                    const float w = p[j] * inv;
                    const float* v = V + static_cast<size_t>(j) * d + h * dh;
                    for (int c = 0; c < dh; ++c) o[c] += w * v[c];
                }
            }
        });
        return;
    }
    // Many queries: per head S = Q·Kᵀ, softmax by rows, O = S·V - both
    // products by the packed kernel.
    std::vector<std::vector<float>> kp(H), vp(H);
    PackHeads(pool, K, V, nk, d, H, kp, vp);
    AttentionPacked(pool, Q, nq, kp, vp, nk, d, H, causal, O);
}

void PackHeads(Pool& pool, const float* K, const float* V, int nk, int d, int H, std::vector<std::vector<float>>& kp,
               std::vector<std::vector<float>>& vp) {
    const int dh = d / H;
    kp.resize(H);
    vp.resize(H);
    pool.run(H, [&](int h) {
        kp[h].resize(PackedSize(nk, dh));
        vp[h].resize(PackedSize(dh, nk));
        PackRows(K + h * dh, nk, dh, d, kp[h].data());
        PackCols(V + h * dh, nk, dh, d, vp[h].data());
    });
}

void AttentionPacked(Pool& pool, const float* Q, int nq, const std::vector<std::vector<float>>& kp,
                     const std::vector<std::vector<float>>& vp, int nk, int d, int H, bool causal, float* O) {
    const int dh = d / H;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));
    constexpr int kBlock = 48;
    const int qb = (nq + kBlock - 1) / kBlock;
    pool.run(H * qb, [&](int t) {
        const int h = t / qb, q0 = (t % qb) * kBlock, q1 = std::min(nq, q0 + kBlock), rows = q1 - q0;
        std::vector<float> S(static_cast<size_t>(rows) * nk);
        Gemm(nullptr, Q + static_cast<size_t>(q0) * d + h * dh, rows, d, kp[h].data(), dh, nk, nullptr, S.data(), nk);
        for (int i = 0; i < rows; ++i) {
            float* s = &S[static_cast<size_t>(i) * nk];
            const int n = causal ? q0 + i + (nk - nq) + 1 : nk;
            SoftmaxRow(s, n, scale);
            std::fill(s + n, s + nk, 0.0f);
        }
        Gemm(nullptr, S.data(), rows, nk, vp[h].data(), nk, dh, nullptr, O + static_cast<size_t>(q0) * d + h * dh, d);
    });
}

}  // namespace wsp
