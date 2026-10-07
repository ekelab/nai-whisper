// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "Whisper.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <set>

namespace wsp {
bool Whisper::bind(Model& m, std::string* err) {
    m_ = &m;
    const Hparams& hp = m.hp;
    std::string missing;
    auto get = [&](const std::string& name, size_t count) {
        const float* p = m.get(name, count);
        if (!p && missing.empty()) missing = name;
        return p;
    };
    auto shape0 = [&](const std::string& name) {
        auto it = m.tensors.find(name);
        return it == m.tensors.end() || it->second.shape.empty() ? 0 : it->second.shape[0];
    };
    // A weight matrix (out × in, or out × c × k flattened) packed in place:
    // float or int8 (Ops.hpp).
    auto packed = [&](const std::string& name, int out, int in) -> Weight {
        Weight w;
        auto it = m.tensors.find(name);
        if (it == m.tensors.end()) {
            if (missing.empty()) missing = name;
            return w;
        }
        Tensor& t = it->second;
        const size_t count = static_cast<size_t>(out) * in;
        if (!t.packed) {
            if ((t.q.empty() ? t.data.size() : t.q.size()) != count) {
                if (missing.empty()) missing = name;
                return w;
            }
            if (t.q.empty()) {
                std::vector<float> p(PackedSize(out, in));
                PackRows(t.data.data(), out, in, in, p.data());
                t.data = std::move(p);
            } else {
                std::vector<int8_t> p(PackedSize(out, in));
                PackRowsQ(t.q.data(), out, in, p.data());
                t.q = std::move(p);
                const int groups = (in + t.group - 1) / t.group;
                std::vector<float> s(static_cast<size_t>((out + kPanel - 1) / kPanel) * kPanel * groups);
                PackScales(t.scale.data(), out, groups, s.data());
                t.scale = std::move(s);
            }
            t.packed = true;
        }
        if (t.q.empty()) {
            w.f = t.data.data();
        } else {
            w.q = t.q.data();
            w.scale = t.scale.data();
            w.group = t.group;
            w.groups = (in + t.group - 1) / t.group;
        }
        return w;
    };
    auto lin = [&](const std::string& p, int out, int in, bool bias) {
        Lin l;
        l.name = p + ".weight";
        l.out = out;
        l.in = in;
        l.w = packed(p + ".weight", out, in);
        if (bias) l.b = get(p + ".bias", out);
        return l;
    };
    auto norm = [&](const std::string& p, int d) { return Norm{get(p + ".weight", d), get(p + ".bias", d)}; };
    const int da = hp.audioState, dt = hp.textState;
    conv1w_ = packed("encoder.conv1.weight", da, hp.nMels * 3);
    conv1b_ = get("encoder.conv1.bias", da);
    conv2w_ = packed("encoder.conv2.weight", da, da * 3);
    conv2b_ = get("encoder.conv2.bias", da);
    encPos_ = get("encoder.embed_positions.weight", static_cast<size_t>(hp.audioCtx) * da);
    encLn_ = norm("encoder.layer_norm", da);
    enc_.assign(hp.audioLayers, Layer{});
    for (int i = 0; i < hp.audioLayers; ++i) {
        const std::string p = "encoder.layers." + std::to_string(i) + ".";
        Layer& l = enc_[i];
        const int ffn = shape0(p + "fc1.weight");
        l.ln1 = norm(p + "self_attn_layer_norm", da);
        l.q = lin(p + "self_attn.q_proj", da, da, true);
        l.k = lin(p + "self_attn.k_proj", da, da, false);
        l.v = lin(p + "self_attn.v_proj", da, da, true);
        l.o = lin(p + "self_attn.out_proj", da, da, true);
        l.ln2 = norm(p + "final_layer_norm", da);
        l.fc1 = lin(p + "fc1", ffn, da, true);
        l.fc2 = lin(p + "fc2", da, ffn, true);
    }
    embed_ = packed("decoder.embed_tokens.weight", hp.vocab, dt);
    decPos_ = get("decoder.embed_positions.weight", static_cast<size_t>(hp.textCtx) * dt);
    decLn_ = norm("decoder.layer_norm", dt);
    dec_.assign(hp.textLayers, Layer{});
    for (int i = 0; i < hp.textLayers; ++i) {
        const std::string p = "decoder.layers." + std::to_string(i) + ".";
        Layer& l = dec_[i];
        const int ffn = shape0(p + "fc1.weight");
        l.ln1 = norm(p + "self_attn_layer_norm", dt);
        l.q = lin(p + "self_attn.q_proj", dt, dt, true);
        l.k = lin(p + "self_attn.k_proj", dt, dt, false);
        l.v = lin(p + "self_attn.v_proj", dt, dt, true);
        l.o = lin(p + "self_attn.out_proj", dt, dt, true);
        l.ln2 = norm(p + "encoder_attn_layer_norm", dt);
        l.cq = lin(p + "encoder_attn.q_proj", dt, dt, true);
        l.ck = lin(p + "encoder_attn.k_proj", dt, da, false);
        l.cv = lin(p + "encoder_attn.v_proj", dt, da, true);
        l.co = lin(p + "encoder_attn.out_proj", dt, dt, true);
        l.ln3 = norm(p + "final_layer_norm", dt);
        l.fc1 = lin(p + "fc1", ffn, dt, true);
        l.fc2 = lin(p + "fc2", dt, ffn, true);
    }
    if (!missing.empty()) {
        if (err) *err = "missing or misshapen tensor " + missing;
        return false;
    }
    return true;
}

void Whisper::encode(Pool& pool, const std::vector<float>& mel, std::vector<float>& out, int ctx) const {
    encodeFront(pool, mel, out, ctx);
    for (int i = 0; i < static_cast<int>(enc_.size()); ++i) encodeLayer(pool, i, out);
    encodeEnd(pool, out);
}

void Whisper::encodeFront(Pool& pool, const std::vector<float>& mel, std::vector<float>& out, int ctx) const {
    const Hparams& hp = m_->hp;
    const int full = 2 * hp.audioCtx;   // mel frames (a row of mel)
    const int T = ctx > 0 && ctx < hp.audioCtx ? ctx : hp.audioCtx, T0 = 2 * T, d = hp.audioState, M = hp.nMels;
    // conv1 (kernel 3, padding 1) as a product: col[t][c·3 + k] = mel[c][t + k − 1].
    std::vector<float> col(static_cast<size_t>(T0) * M * 3, 0.0f), h1(static_cast<size_t>(T0) * d);
    for (int t = 0; t < T0; ++t)
        for (int c = 0; c < M; ++c)
            for (int k = 0; k < 3; ++k) {
                const int s = t + k - 1;
                if (s >= 0 && s < full) col[(static_cast<size_t>(t) * M + c) * 3 + k] = mel[static_cast<size_t>(c) * full + s];
            }
    Linear(pool, col.data(), T0, M * 3, conv1w_, conv1b_, d, h1.data());
    Gelu(pool, h1.data(), h1.size());
    // conv2 (kernel 3, stride 2, padding 1): col[t][c·3 + k] = h1[2t + k − 1][c].
    col.assign(static_cast<size_t>(T) * d * 3, 0.0f);
    for (int t = 0; t < T; ++t)
        for (int k = 0; k < 3; ++k) {
            const int s = 2 * t + k - 1;
            if (s < 0 || s >= T0) continue;
            for (int c = 0; c < d; ++c) col[(static_cast<size_t>(t) * d + c) * 3 + k] = h1[static_cast<size_t>(s) * d + c];
        }
    std::vector<float>& x = out;
    x.assign(static_cast<size_t>(T) * d, 0.0f);
    Linear(pool, col.data(), T, d * 3, conv2w_, conv2b_, d, x.data());
    Gelu(pool, x.data(), x.size());
    for (size_t i = 0; i < x.size(); ++i) x[i] += encPos_[i];
}

void Whisper::encodeLayer(Pool& pool, int layer, std::vector<float>& x) const {
    const Hparams& hp = m_->hp;
    const int d = hp.audioState, T = static_cast<int>(x.size() / d);
    const size_t n = static_cast<size_t>(T) * d;
    std::vector<float> h(n), q(n), k(n), v(n), a(n), o(n), f;
    {
        const Layer& l = enc_[layer];
        h = x;
        LayerNorm(pool, h.data(), T, d, l.ln1.g, l.ln1.b);
        Linear(pool, h.data(), T, d, l.q.w, l.q.b, d, q.data());
        Linear(pool, h.data(), T, d, l.k.w, nullptr, d, k.data());
        Linear(pool, h.data(), T, d, l.v.w, l.v.b, d, v.data());
        Attention(pool, q.data(), T, k.data(), v.data(), T, d, hp.audioHeads, false, a.data());
        Linear(pool, a.data(), T, d, l.o.w, l.o.b, d, o.data());
        for (size_t i = 0; i < n; ++i) x[i] += o[i];
        h = x;
        LayerNorm(pool, h.data(), T, d, l.ln2.g, l.ln2.b);
        f.resize(static_cast<size_t>(T) * l.fc1.out);
        Linear(pool, h.data(), T, d, l.fc1.w, l.fc1.b, l.fc1.out, f.data());
        Gelu(pool, f.data(), f.size());
        Linear(pool, f.data(), T, l.fc1.out, l.fc2.w, l.fc2.b, d, o.data());
        for (size_t i = 0; i < n; ++i) x[i] += o[i];
    }
}

void Whisper::encodeEnd(Pool& pool, std::vector<float>& x) const {
    const int d = m_->hp.audioState;
    LayerNorm(pool, x.data(), static_cast<int>(x.size() / d), d, encLn_.g, encLn_.b);
}

std::vector<int> Whisper::decode(Pool& pool, const std::vector<float>& audio, int langToken,
                                 std::vector<float>* firstLogits, const std::vector<int>* forced,
                                 const std::function<void(const std::vector<float>&)>& onLogits,
                                 DecodeInfo* info, const DecodeOptions* opt) const {
    const DecodeOptions defaults;
    if (!opt) opt = &defaults;
    const Hparams& hp = m_->hp;
    const int d = hp.textState, T = static_cast<int>(audio.size() / hp.audioState), L = hp.textLayers, ctx = hp.textCtx;
    // Cross-attention keys and values, once per recording.
    std::vector<std::vector<std::vector<float>>> ckp(L), cvp(L);
    std::vector<std::vector<float>> sk(L), sv(L);
    std::vector<float> ck(static_cast<size_t>(T) * d), cv(ck.size());
    for (int i = 0; i < L; ++i) {
        const Layer& l = dec_[i];
        Linear(pool, audio.data(), T, hp.audioState, l.ck.w, nullptr, d, ck.data());
        Linear(pool, audio.data(), T, hp.audioState, l.cv.w, l.cv.b, d, cv.data());
        PackHeads(pool, ck.data(), cv.data(), T, d, hp.textHeads, ckp[i], cvp[i]);
        sk[i].assign(static_cast<size_t>(ctx) * d, 0.0f);
        sv[i].assign(static_cast<size_t>(ctx) * d, 0.0f);
    }
    std::vector<float> x(d), h(d), q(d), a(d), o(d), f, logits(hp.vocab);
    // One token at position pos through the decoder → logits.
    auto step = [&](int token, int pos) {
        for (int c = 0; c < d; ++c)
            x[c] = WeightAt(embed_, d, token, c) + decPos_[static_cast<size_t>(pos) * d + c];
        for (int i = 0; i < L; ++i) {
            const Layer& l = dec_[i];
            h = x;
            LayerNorm(pool, h.data(), 1, d, l.ln1.g, l.ln1.b);
            Linear(pool, h.data(), 1, d, l.q.w, l.q.b, d, q.data());
            Linear(pool, h.data(), 1, d, l.k.w, nullptr, d, &sk[i][static_cast<size_t>(pos) * d]);
            Linear(pool, h.data(), 1, d, l.v.w, l.v.b, d, &sv[i][static_cast<size_t>(pos) * d]);
            Attention(pool, q.data(), 1, sk[i].data(), sv[i].data(), pos + 1, d, hp.textHeads, true, a.data());
            Linear(pool, a.data(), 1, d, l.o.w, l.o.b, d, o.data());
            for (int c = 0; c < d; ++c) x[c] += o[c];
            h = x;
            LayerNorm(pool, h.data(), 1, d, l.ln2.g, l.ln2.b);
            Linear(pool, h.data(), 1, d, l.cq.w, l.cq.b, d, q.data());
            AttentionPacked(pool, q.data(), 1, ckp[i], cvp[i], T, d, hp.textHeads, false, a.data());
            Linear(pool, a.data(), 1, d, l.co.w, l.co.b, d, o.data());
            for (int c = 0; c < d; ++c) x[c] += o[c];
            h = x;
            LayerNorm(pool, h.data(), 1, d, l.ln3.g, l.ln3.b);
            f.resize(l.fc1.out);
            Linear(pool, h.data(), 1, d, l.fc1.w, l.fc1.b, l.fc1.out, f.data());
            Gelu(pool, f.data(), f.size());
            Linear(pool, f.data(), 1, l.fc1.out, l.fc2.w, l.fc2.b, d, o.data());
            for (int c = 0; c < d; ++c) x[c] += o[c];
        }
        h = x;
        LayerNorm(pool, h.data(), 1, d, decLn_.g, decLn_.b);
        Linear(pool, h.data(), 1, d, embed_, nullptr, hp.vocab, logits.data());
    };
    int pos = 0;
    step(m_->sot, pos++);
    if (langToken < 0) {   // the most likely language token after the start
        double mx = -INFINITY, z = 0;
        for (const auto& [code, tok] : m_->lang) mx = std::max(mx, static_cast<double>(logits[tok]));
        for (const auto& [code, tok] : m_->lang) z += std::exp(logits[tok] - mx);
        std::vector<std::pair<int, float>> probs;
        for (const auto& [code, tok] : m_->lang)
            probs.push_back({tok, static_cast<float>(std::exp(logits[tok] - mx) / z)});
        std::sort(probs.begin(), probs.end(), [](const auto& a, const auto& b) { return a.second > b.second; });
        langToken = probs.empty() ? m_->lang.begin()->second : probs[0].first;
        if (info) info->langProbs = std::move(probs);
    }
    if (info) info->lang = langToken;
    if (opt->detectOnly) return {};
    step(langToken, pos++);
    step(m_->transcribe, pos++);
    step(m_->noTimestamps, pos++);
    std::vector<int> out;
    if (forced) {   // teacher forcing: the logits before each given token and before the end
        for (size_t i = 0; i <= forced->size() && pos < ctx; ++i) {
            if (onLogits) onLogits(logits);
            if (i == forced->size() || pos + 1 >= ctx) break;
            out.push_back((*forced)[i]);
            step((*forced)[i], pos++);
        }
        return out;
    }
    while (pos < ctx) {
        if (out.empty() && firstLogits) *firstLogits = logits;
        if (onLogits) onLogits(logits);
        for (int t : m_->suppress) logits[t] = -INFINITY;
        if (out.empty())
            for (int t : m_->beginSuppress) logits[t] = -INFINITY;
        const int best = static_cast<int>(std::max_element(logits.begin(), logits.end()) - logits.begin());
        if (info) {   // log P(best) after the suppression
            double z = 0;
            for (float v : logits) z += std::exp(static_cast<double>(v) - logits[best]);
            info->logprob.push_back(static_cast<float>(-std::log(z)));
        }
        if (best == m_->eot) break;
        out.push_back(best);
        if (pos + 1 >= ctx) {
            if (info) info->hitLimit = true;
            break;
        }
        step(best, pos++);
    }
    return out;
}

LoopSigns Signs(const std::vector<int>& tokens, const DecodeInfo& info, double seconds) {
    LoopSigns g;
    g.limit = info.hitLimit;
    g.perSecond = tokens.size() / std::max(seconds, 0.5);
    double lp = 0;
    for (float v : info.logprob) lp += v;
    g.avgLogprob = info.logprob.empty() ? 0 : lp / info.logprob.size();
    // Positions covered by a 4-gram seen earlier.
    constexpr size_t kN = 4;
    if (tokens.size() >= 2 * kN) {
        std::set<std::array<int, kN>> seen;
        std::vector<char> covered(tokens.size(), 0);
        for (size_t i = 0; i + kN <= tokens.size(); ++i) {
            std::array<int, kN> g4;
            std::copy_n(&tokens[i], kN, g4.begin());
            if (!seen.insert(g4).second) std::fill_n(&covered[i], kN, 1);
        }
        size_t c = 0;
        for (char v : covered) c += v;
        g.repeat = static_cast<double>(c) / tokens.size();
    }
    return g;
}

std::string Whisper::text(const std::vector<int>& tokens) const {
    std::string s;
    for (int t : tokens)
        if (t >= 0 && t < m_->eot) s += m_->pieces[t];
    return s;
}

}  // namespace wsp
