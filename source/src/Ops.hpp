// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// The few operations a transformer needs, on float32 arrays in row-major
// order, spread over a pool of threads.

#include <condition_variable>
#include <cstdint>
#include <functional>
#include <mutex>
#include <thread>
#include <vector>

namespace wsp {

/** A fixed set of worker threads; run(n, f) calls f(0) … f(n−1) on them and
 *  the caller, and returns when all are done. */
class Pool {
public:
    explicit Pool(int threads);
    ~Pool();
    Pool(const Pool&) = delete;
    Pool& operator=(const Pool&) = delete;
    int size() const { return static_cast<int>(workers_.size()) + 1; }
    void run(int n, const std::function<void(int)>& f);

private:
    void work();
    void drain();
    std::vector<std::thread> workers_;
    std::mutex mu_;
    std::condition_variable start_, done_;
    const std::function<void(int)>* job_ = nullptr;
    int n_ = 0, next_ = 0, finished_ = 0;
    long long epoch_ = 0;
    bool stop_ = false;
};

// Weight matrices are kept packed: panels of kPanel outputs, the panel's
// values for input k side by side - P[(p·in + k)·kPanel + j] = W[p·kPanel + j][k],
// zeros past the last output. The product then reads each panel once per
// block of input rows and keeps a 6 × 16 block of outputs in registers.
constexpr int kPanel = 16;
size_t PackedSize(int out, int in);
/** W: out × in, row o at W + o·ld. */
void PackRows(const float* W, int out, int in, int ld, float* P);
/** The matrix Vᵀ, V being k × out with row r at V + r·ld (out = its columns). */
void PackCols(const float* V, int k, int out, int ld, float* P);
/** W[o][k] of a packed matrix. */
inline float PackedAt(const float* P, int in, int o, int k) {
    return P[(static_cast<size_t>(o / kPanel) * in + k) * kPanel + o % kPanel];
}

/** Y[n×out] (row stride ldy) = X[n×in] (row stride ldx)·Wᵀ + b; P packed W,
 *  b may be null; pool null: on this thread; add: Y += instead of Y =. */
void Gemm(Pool* pool, const float* X, int n, int ldx, const float* P, int in, int out, const float* b, float* Y,
          int ldy, bool add = false);

/** A packed weight matrix: float panels (f), or integer panels (q) with a
 *  scale per output and group of `group` inputs - panel p's scales for
 *  group g at scale[(p·groups + g)·kPanel …]. */
struct Weight {
    const float* f = nullptr;
    const int8_t* q = nullptr;
    const float* scale = nullptr;
    int group = 1 << 30, groups = 1;
};
/** Scales s[o·groups + g] into the panel layout of Weight (zeros past out). */
void PackScales(const float* s, int out, int groups, float* P);
/** int8 W (out × in, rows dense) into panels, as PackRows. */
void PackRowsQ(const int8_t* W, int out, int in, int8_t* P);
/** W[o][k]. */
inline float WeightAt(const Weight& W, int in, int o, int k) {
    const size_t i = (static_cast<size_t>(o / kPanel) * in + k) * kPanel + o % kPanel;
    return W.q ? W.q[i] * W.scale[(static_cast<size_t>(o / kPanel) * W.groups + k / W.group) * kPanel + o % kPanel]
               : W.f[i];
}
/** Gemm with either kind of weights. */
void Gemm(Pool* pool, const float* X, int n, int ldx, const Weight& W, int in, int out, const float* b, float* Y,
          int ldy, bool add = false);

/** Y[n×out] = X[n×in]·Wᵀ + b, rows dense. */
inline void Linear(Pool& pool, const float* X, int n, int in, const Weight& W, const float* b, int out, float* Y) {
    Gemm(&pool, X, n, in, W, in, out, b, Y, out);
}

/** Layer norm of each of n rows of x (d values), in place: (x − mean)/σ·g + b. */
void LayerNorm(Pool& pool, float* x, int n, int d, const float* g, const float* b);

/** GELU (erf to float precision), in place. */
void Gelu(Pool& pool, float* x, size_t n);

/** e^x for each value x ≤ 88, in place (float precision, vectorizes). */
void ExpInPlace(float* x, int n);

float Dot(const float* a, const float* b, int n);

/** Multi-head attention: O[nq×d] from Q[nq×d] over K, V[nk×d] (rows of d),
 *  H heads of d/H; causal: query i sees keys 0 … i + (nk − nq). */
void Attention(Pool& pool, const float* Q, int nq, const float* K, const float* V, int nk, int d, int H, bool causal,
               float* O);

/** The keys and values of each head packed (kp[h]: K_h as nk outputs of dh;
 *  vp[h]: V_hᵀ as dh outputs of nk) - once, when the same keys serve many
 *  queries (the decoder's cross-attention). */
void PackHeads(Pool& pool, const float* K, const float* V, int nk, int d, int H, std::vector<std::vector<float>>& kp,
               std::vector<std::vector<float>>& vp);
/** Attention over packed heads (PackHeads). */
void AttentionPacked(Pool& pool, const float* Q, int nq, const std::vector<std::vector<float>>& kp,
                     const std::vector<std::vector<float>>& vp, int nk, int d, int H, bool causal, float* O);

}  // namespace wsp
