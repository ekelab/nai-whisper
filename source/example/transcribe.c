/*
 * Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
 * use requires the written permission of the author (lab@mindscan.org).
 * See LICENSE.md in the repository.
 */

/* The smallest program with the network: load it, give it audio, print the
 * timed segments.
 *
 *     transcribe model.nll audio.wav [language|auto]
 *
 * The WAV must be 16 kHz (mono or stereo, 16-bit PCM or 32-bit float);
 * convert other audio first, e.g. ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "wsp_runtime.h"

static unsigned u16(const unsigned char* p) { return p[0] | p[1] << 8; }
static unsigned u32(const unsigned char* p) { return p[0] | p[1] << 8 | p[2] << 16 | (unsigned)p[3] << 24; }

/* Mono float samples of a 16 kHz WAV; NULL if it is not one. */
static float* read_wav(const char* path, long long* samples) {
    FILE* f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    unsigned char* d = malloc(size);
    if (!d || fread(d, 1, size, f) != (size_t)size) { fclose(f); free(d); return NULL; }
    fclose(f);
    float* out = NULL;
    unsigned format = 0, channels = 0, rate = 0, bits = 0;
    if (size < 12 || memcmp(d, "RIFF", 4) || memcmp(d + 8, "WAVE", 4)) { free(d); return NULL; }
    for (long p = 12; p + 8 <= size;) {
        unsigned len = u32(d + p + 4);
        const unsigned char* body = d + p + 8;
        if (!memcmp(d + p, "fmt ", 4)) {
            format = u16(body); channels = u16(body + 2); rate = u32(body + 4); bits = u16(body + 14);
            if (format == 0xfffe && len >= 26) format = u16(body + 24);
        } else if (!memcmp(d + p, "data", 4)) {
            if (rate != 16000 || !channels || !((format == 1 && bits == 16) || (format == 3 && bits == 32))) break;
            if (p + 8 + (long)len > size) len = (unsigned)(size - p - 8);
            long long n = len / (bits / 8) / channels;
            out = malloc(sizeof(float) * (n ? n : 1));
            for (long long i = 0; i < n; ++i) {
                float s = 0;
                for (unsigned c = 0; c < channels; ++c) {
                    const unsigned char* q = body + (i * channels + c) * (bits / 8);
                    if (bits == 16) s += (short)u16(q) / 32768.0f;
                    else { float v; memcpy(&v, q, 4); s += v; }
                }
                out[i] = s / channels;
            }
            *samples = n;
            break;
        }
        p += 8 + len + (len & 1);
    }
    free(d);
    return out;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: transcribe model.nll audio.wav [language|auto]\n");
        return 2;
    }
    long long n = 0;
    float* pcm = read_wav(argv[2], &n);
    if (!pcm) {
        fprintf(stderr, "cannot read %s (a 16 kHz WAV, 16-bit or float)\n", argv[2]);
        return 1;
    }
    wsp_model* m = wsp_load(argv[1]);                  /* 1. the network */
    if (!m) {
        fprintf(stderr, "cannot load %s\n", argv[1]);
        return 1;
    }
    wsp_params p = wsp_default_params();               /* 2. the settings */
    p.language = argc > 3 ? argv[3] : "auto";
    wsp_result* r = wsp_run(m, pcm, n, &p);            /* 3. the audio in, the result out */
    if (!r) {
        fprintf(stderr, "unknown language %s\n", p.language);
        return 1;
    }
    printf("language: %s (p = %.3f)\n", wsp_result_language(r), wsp_result_language_prob(r));
    for (int i = 0; i < wsp_result_count(r); ++i)
        printf("[%8.2f --> %8.2f] %s\n", wsp_result_t0(r, i), wsp_result_t1(r, i), wsp_result_text(r, i));
    wsp_result_free(r);
    wsp_free(m);
    free(pcm);
    return 0;
}
