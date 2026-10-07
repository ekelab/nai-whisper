/*
 * Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
 * use requires the written permission of the author (lab@mindscan.org).
 * See LICENSE.md in the repository.
 */

/* wsp_runtime - the interpreter of a speech-recognition network file (.nll):
 * Whisper large-v3-turbo and its compressed versions. C interface; inside,
 * the C++ standard library only.
 *
 * Audio everywhere: mono, 16 kHz, float samples in [-1, 1].
 * Languages: Whisper's codes ("ru", "en", "de", ...); "auto" (or NULL)
 * detects the language from the audio. */
#ifndef WSP_RUNTIME_H
#define WSP_RUNTIME_H

/* Exported from a shared library (nai_whisper.dll / libnai_whisper.so) when
 * built with WSP_BUILD_SHARED; nothing to define when using it. */
#if defined(_WIN32) && defined(WSP_BUILD_SHARED)
#define WSP_API __declspec(dllexport)
#elif defined(WSP_BUILD_SHARED)
#define WSP_API __attribute__((visibility("default")))
#else
#define WSP_API
#endif

#ifdef __cplusplus
extern "C" {
#endif

typedef struct wsp_model wsp_model;

/** Reads a network file; NULL if it cannot. */
WSP_API wsp_model* wsp_load(const char* path);
WSP_API void wsp_free(wsp_model* m);

/** Threads for the next calls (default: all the CPU has). */
WSP_API void wsp_set_threads(wsp_model* m, int threads);

/** The text of a short recording (its first 30 s). The encoder sees the
 *  audio and 1 s more, at least 15 s; if the result looks broken (repeats,
 *  too many tokens per second) the whole 30 s window, as the model was
 *  trained. Writes UTF-8 with a terminating 0 into out (cap bytes); returns
 *  the text's length in bytes, or -1 (unknown language, bad arguments). */
WSP_API int wsp_transcribe(wsp_model* m, const float* pcm, int samples, const char* lang, char* out, int cap);

/** The language of the first 30 s: its code into code (cap bytes), its
 *  probability into prob (may be NULL). Returns the code's length, or -1. */
WSP_API int wsp_detect_language(wsp_model* m, const float* pcm, int samples, char* code, int cap, float* prob);

/* ---- Audio of any length to timed segments -------------------------------
 * The audio is cut at pauses into pieces of 5-15 s, each transcribed as
 * wsp_transcribe does; a segment per piece, with its start and end. Audio of
 * 30 s or less is one piece - exactly wsp_transcribe. */

typedef struct wsp_params {
    const char* language;     /* code, or "auto"/NULL: detected on the first window */
    int detect_each_window;   /* 1: the language detected again for every piece (mixed speech) */
    /* Optional callbacks, called from the calling thread during wsp_run. */
    void (*progress)(double done_sec, double total_sec, void* user);
    void (*segment)(double t0, double t1, const char* text, const char* lang, void* user);
    void* user;
} wsp_params;

WSP_API wsp_params wsp_default_params(void);

typedef struct wsp_result wsp_result;

/** Transcribes samples (any length); NULL on bad arguments or an unknown
 *  language. Free the result with wsp_result_free. */
WSP_API wsp_result* wsp_run(wsp_model* m, const float* pcm, long long samples, const wsp_params* params);
WSP_API int wsp_result_count(const wsp_result* r);                 /* segments */
WSP_API double wsp_result_t0(const wsp_result* r, int i);          /* seconds from the start */
WSP_API double wsp_result_t1(const wsp_result* r, int i);
WSP_API const char* wsp_result_text(const wsp_result* r, int i);   /* UTF-8, valid until free */
WSP_API const char* wsp_result_segment_language(const wsp_result* r, int i);
WSP_API const char* wsp_result_language(const wsp_result* r);      /* detected (or given) */
WSP_API float wsp_result_language_prob(const wsp_result* r);       /* 1 when given */
WSP_API int wsp_result_redone(const wsp_result* r);                /* windows done again whole */
WSP_API void wsp_result_free(wsp_result* r);

/* ---- A live stream --------------------------------------------------------
 * Audio fed in portions, cut at pauses into pieces (at most 15 s by
 * default), each transcribed as wsp_transcribe does once it is finished. The
 * calls are synchronous: push transcribes, inside the call, every piece the
 * new audio finished (call it from a worker thread if the caller must not
 * wait). */
typedef struct wsp_stream wsp_stream;

/** A stream for language lang (a code; a stream cannot detect); NULL if
 *  there is no such language. */
WSP_API wsp_stream* wsp_stream_new(wsp_model* m, const char* lang);
/** How pieces are cut (before the first push): a piece ends at the first
 *  pause of pause_sec or more once it is min_sec long, or by max_sec
 *  (at most 25) at its quietest point. Defaults 5, 15, 0.6. Returns 0, or -1. */
WSP_API int wsp_stream_cut(wsp_stream* s, double min_sec, double max_sec, double pause_sec);
/** More audio. Returns the pieces transcribed so far, or -1. */
WSP_API int wsp_stream_push(wsp_stream* s, const float* pcm, int samples);
/** The audio is over: the rest is transcribed. Returns the pieces, or -1. */
WSP_API int wsp_stream_finish(wsp_stream* s);
/** The text finished since the last call, UTF-8 with a terminating 0 (cap
 *  bytes); returns its length in bytes (more than cap - 1: cut), or -1. */
WSP_API int wsp_stream_text(wsp_stream* s, char* out, int cap);
/** Pieces done again with the whole 30 s window so far, or -1. */
WSP_API int wsp_stream_redone(wsp_stream* s);
WSP_API void wsp_stream_free(wsp_stream* s);

#ifdef __cplusplus
}
#endif

#endif
