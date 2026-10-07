// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#include "wsp_runtime.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>
#include <map>
#include <thread>

#include "Mel.hpp"
#include "Segment.hpp"
#include "Transcribe.hpp"
#include "Whisper.hpp"

namespace {
// A piece cut from a stream ends inside a pause, not with the speaker's own
// trailing silence: with 1 s past it the short window broke into loops 3x as
// often as with 5 s, so stream pieces get 4 s more.
constexpr int kStreamExtra = 200;
}  // namespace

struct wsp_model {
    wsp::Model model;
    wsp::Whisper net;
    std::unique_ptr<wsp::Pool> pool;
    std::map<int, std::string> codeOf;   // language token → code
};

struct wsp_stream {
    wsp_model* m = nullptr;
    int lang = 0;
    wsp::Cutter cutter;
    std::vector<float> buf;      // the samples from the current piece's start
    long long bufStart = 0;      // the sample index of buf[0]
    std::string text;            // finished, not yet taken
    int pieces = 0, redone = 0;
};

struct wsp_result {
    std::vector<wsp::TimedSegment> segments;
    std::vector<std::string> langs;
    std::string lang;
    float langProb = 0;
    int windows = 0, redone = 0;
};

namespace {

void Put(const std::string& s, char* out, int cap) {
    if (cap <= 0) return;
    const size_t len = std::min(s.size(), static_cast<size_t>(cap - 1));
    std::memcpy(out, s.data(), len);
    out[len] = 0;
}

/** The text of up to 30 s, one window under the short-window rule. */
std::string Transcribe(wsp_model* m, const float* pcm, int samples, int langToken, int* redone = nullptr,
                       int extraMargin = 0) {
    bool again = false;
    const std::vector<int> toks = wsp::DecodeWindow(m->net, *m->pool, pcm, samples, langToken, wsp::WindowRule(),
                                                    wsp::DecodeOptions(), nullptr, &again, extraMargin);
    if (redone && again) ++*redone;
    return m->net.text(toks);
}

/** Transcribes every piece the cutter has decided (final: the rest too). */
void Drain(wsp_stream* s, bool final) {
    for (;;) {
        const long long cut = s->cutter.next(final);
        if (cut < 0) return;
        const int n = static_cast<int>(cut - s->bufStart);
        if (n > wsp::kSampleRate / 10) {   // pieces under 0.1 s carry no speech
            std::string t = Transcribe(s->m, s->buf.data(), n, s->lang, &s->redone, kStreamExtra);
            const size_t a = t.find_first_not_of(' ');
            if (a != std::string::npos) {
                if (!s->text.empty() && s->text.back() != ' ') s->text += ' ';
                s->text += t.substr(a);
            }
            ++s->pieces;
        }
        s->buf.erase(s->buf.begin(), s->buf.begin() + n);
        s->bufStart = cut;
        if (final && s->buf.empty()) return;
    }
}

/** The token of a language code; "auto", "" or null: −1 (detect); −2: unknown. */
int LangToken(const wsp_model* m, const char* lang) {
    if (!lang || !*lang || !std::strcmp(lang, "auto")) return -1;
    auto it = m->model.lang.find(lang);
    return it == m->model.lang.end() ? -2 : it->second;
}

}  // namespace

extern "C" {

wsp_model* wsp_load(const char* path) {
    if (!path) return nullptr;
    auto m = std::make_unique<wsp_model>();
    if (!m->model.load(path, nullptr) || !m->net.bind(m->model, nullptr)) return nullptr;
    m->pool = std::make_unique<wsp::Pool>(static_cast<int>(std::max(1u, std::thread::hardware_concurrency())));
    for (const auto& [code, tok] : m->model.lang) m->codeOf[tok] = code;
    return m.release();
}

void wsp_free(wsp_model* m) { delete m; }

void wsp_set_threads(wsp_model* m, int threads) {
    if (m && threads > 0 && threads != m->pool->size()) m->pool = std::make_unique<wsp::Pool>(threads);
}

int wsp_transcribe(wsp_model* m, const float* pcm, int samples, const char* lang, char* out, int cap) {
    if (!m || !pcm || samples < 0 || (cap > 0 && !out)) return -1;
    const int tok = LangToken(m, lang);
    if (tok < -1) return -1;
    const std::string s = Transcribe(m, pcm, samples, tok);
    Put(s, out, cap);
    return static_cast<int>(s.size());
}

int wsp_detect_language(wsp_model* m, const float* pcm, int samples, char* code, int cap, float* prob) {
    if (!m || !pcm || samples < 0 || (cap > 0 && !code)) return -1;
    std::vector<float> mel, audio;
    const int n = std::min(samples, wsp::kChunk);
    wsp::LogMel(*m->pool, pcm, n, m->model.melFilters, m->model.hp.nMels, mel);
    m->net.encode(*m->pool, mel, audio);
    wsp::DecodeInfo info;
    wsp::DecodeOptions o;
    o.detectOnly = true;
    m->net.decode(*m->pool, audio, -1, nullptr, nullptr, {}, &info, &o);
    const std::string c = m->codeOf.count(info.lang) ? m->codeOf.at(info.lang) : std::string();
    Put(c, code, cap);
    if (prob) *prob = info.langProbs.empty() ? 0.0f : info.langProbs[0].second;
    return static_cast<int>(c.size());
}

wsp_params wsp_default_params(void) {
    wsp_params p{};
    p.language = "auto";
    p.detect_each_window = 0;
    return p;
}

wsp_result* wsp_run(wsp_model* m, const float* pcm, long long samples, const wsp_params* params) {
    if (!m || (!pcm && samples > 0) || samples < 0) return nullptr;
    const wsp_params p = params ? *params : wsp_default_params();
    const int tok = LangToken(m, p.language);
    if (tok < -1) return nullptr;
    auto r = std::make_unique<wsp_result>();
    wsp::LongOptions o;
    o.langToken = tok;
    o.detectEachWindow = p.detect_each_window != 0;
    if (p.progress) o.progress = [&](double done, double total) { p.progress(done, total, p.user); };
    if (p.segment)
        o.onSegment = [&](const wsp::TimedSegment& s) {
            const auto it = m->codeOf.find(s.lang);
            p.segment(s.t0, s.t1, s.text.c_str(), it == m->codeOf.end() ? "" : it->second.c_str(), p.user);
        };
    wsp::LongResult lr = wsp::TranscribeLong(m->net, *m->pool, pcm, static_cast<size_t>(samples), o);
    r->segments = std::move(lr.segments);
    for (const auto& s : r->segments) {
        const auto it = m->codeOf.find(s.lang);
        r->langs.push_back(it == m->codeOf.end() ? "" : it->second);
    }
    const auto it = m->codeOf.find(lr.lang);
    r->lang = it == m->codeOf.end() ? "" : it->second;
    r->langProb = tok >= 0 ? 1.0f : lr.langProb;
    r->windows = lr.windows;
    r->redone = lr.redone;
    return r.release();
}

int wsp_result_count(const wsp_result* r) { return r ? static_cast<int>(r->segments.size()) : -1; }

double wsp_result_t0(const wsp_result* r, int i) {
    return r && i >= 0 && i < static_cast<int>(r->segments.size()) ? r->segments[i].t0 : -1;
}

double wsp_result_t1(const wsp_result* r, int i) {
    return r && i >= 0 && i < static_cast<int>(r->segments.size()) ? r->segments[i].t1 : -1;
}

const char* wsp_result_text(const wsp_result* r, int i) {
    return r && i >= 0 && i < static_cast<int>(r->segments.size()) ? r->segments[i].text.c_str() : nullptr;
}

const char* wsp_result_segment_language(const wsp_result* r, int i) {
    return r && i >= 0 && i < static_cast<int>(r->langs.size()) ? r->langs[i].c_str() : nullptr;
}

const char* wsp_result_language(const wsp_result* r) { return r ? r->lang.c_str() : nullptr; }

float wsp_result_language_prob(const wsp_result* r) { return r ? r->langProb : 0.0f; }

int wsp_result_redone(const wsp_result* r) { return r ? r->redone : -1; }

void wsp_result_free(wsp_result* r) { delete r; }

wsp_stream* wsp_stream_new(wsp_model* m, const char* lang) {
    if (!m) return nullptr;
    const int tok = LangToken(m, lang);
    if (tok < 0) return nullptr;   // a stream needs its language
    auto s = std::make_unique<wsp_stream>();
    s->m = m;
    s->lang = tok;
    return s.release();
}

int wsp_stream_cut(wsp_stream* s, double min_sec, double max_sec, double pause_sec) {
    if (!s || s->bufStart > 0 || !s->buf.empty() || min_sec < 0 || max_sec <= min_sec || max_sec > 25 || pause_sec <= 0)
        return -1;
    s->cutter = wsp::Cutter(wsp::CutOptions{min_sec, max_sec, pause_sec});
    return 0;
}

int wsp_stream_push(wsp_stream* s, const float* pcm, int samples) {
    if (!s || (!pcm && samples > 0) || samples < 0) return -1;
    s->buf.insert(s->buf.end(), pcm, pcm + samples);
    s->cutter.push(pcm, samples);
    Drain(s, false);
    return s->pieces;
}

int wsp_stream_finish(wsp_stream* s) {
    if (!s) return -1;
    Drain(s, true);
    return s->pieces;
}

int wsp_stream_text(wsp_stream* s, char* out, int cap) {
    if (!s || (cap > 0 && !out)) return -1;
    const int n = static_cast<int>(s->text.size());
    Put(s->text, out, cap);
    s->text.clear();
    return n;
}

int wsp_stream_redone(wsp_stream* s) { return s ? s->redone : -1; }

void wsp_stream_free(wsp_stream* s) { delete s; }

}  // extern "C"
