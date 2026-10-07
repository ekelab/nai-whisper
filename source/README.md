# The interpreter

The source of the program that runs the `.nll` network files (Whisper
large-v3-turbo and its compressed versions, from this repository's
release). About 2,100 lines of C++ using nothing but the C++ standard
library, behind a C interface ([include/wsp_runtime.h](include/wsp_runtime.h)),
so a program in any language that can call C can use the network.

Free for non-commercial use; see [../LICENSE.md](../LICENSE.md).

## What you need

- **To build:** GCC or Clang with C++20, CMake 3.15 or newer (or just the
  compiler, see below). MSVC is not supported: the arithmetic uses GCC/Clang
  vector extensions.
  - Windows: MinGW-w64 (for example MSYS2's `ucrt64`).
  - macOS: Xcode 14.3 or newer, or its Command Line Tools
    (`xcode-select --install`), and CMake (for example `brew install cmake`).
  - Linux: GCC 10 or newer, or Clang 14 or newer.
- **To run:** about 1 GB of free memory and a network file from the
  release (`whisper_large_v3_turbo_m5d6.nll` or `whisper_large_v3_turbo_pp2.nll`,
  in the `models/` folder of `NaiWhisper-windows-x64.zip`).
- On x86-64 the build uses AVX2 and FMA (Intel since 2013, AMD since 2015);
  other processors, Apple silicon (M1 and later) included, get the
  compiler's own vector instructions - the code has no processor-specific
  parts.
- Checked on Windows 11 (MinGW-w64 GCC), with AVX2 and without it (the same
  text). macOS and Linux are not checked yet: they should build the same
  way; the speed there is not measured.

## Build

```
cmake -S . -B build
cmake --build build -j
```

| file | what | for |
|---|---|---|
| `build/libnai_whisper.a` | static library | C and C++ programs |
| `build/nai_whisper.dll` (`libnai_whisper.dylib` on macOS, `libnai_whisper.so` on Linux) | shared library: the functions of `wsp_runtime.h`, nothing else exported; on Windows the C++ runtime is inside, nothing else to ship | Python, Go, C#, Java, ... |
| `build/transcribe` | the example: a WAV file to timed text | trying it |

Options: `-DNAI_WHISPER_AVX2=OFF` builds for any x86-64 (slower);
`-DNAI_WHISPER_SHARED=OFF` skips the shared library.

Try it:

```
build/transcribe whisper_large_v3_turbo_m5d6.nll speech.wav auto
```

## Add it to your program

### C or C++, with CMake

```cmake
add_subdirectory(path/to/source nai_whisper)
target_link_libraries(my_program PRIVATE nai_whisper)
```

and `#include "wsp_runtime.h"`.

### C or C++, without CMake

Compile the interpreter's files with the C++ compiler, your program as
usual, and link with the C++ compiler:

```
g++ -std=c++20 -O3 -march=x86-64-v3 -Iinclude -c src/*.cpp
gcc -O2 -Iinclude -c my_program.c
g++ *.o -o my_program -pthread          # on Windows add -static for one self-contained .exe
```

On macOS use `clang++` and `clang` the same way, without `-march=x86-64-v3`
on Apple silicon.

### Any other language: the shared library

Ship `nai_whisper.dll` (`libnai_whisper.dylib`, `libnai_whisper.so`) and a network file with
your program and call the functions of `wsp_runtime.h` through the
language's C interface. [example/transcribe.py](example/transcribe.py) does
it from Python with the standard `ctypes` module:

```
python example/transcribe.py build/nai_whisper.dll whisper_large_v3_turbo_m5d6.nll speech.wav auto
```

`wsp_params` is a plain C struct (a `const char*`, an `int`, two function
pointers and a `void*`, in that order); the rest of the interface takes and
returns pointers, numbers and UTF-8 strings.

## Use

Three steps: load the network, give it audio, read the result.

```c
#include "wsp_runtime.h"

wsp_model* m = wsp_load("whisper_large_v3_turbo_m5d6.nll");   /* 1. the network (a few seconds) */
if (!m) { /* the file is missing or is not a network file */ }

wsp_params p = wsp_default_params();                          /* 2. the settings */
p.language = "auto";                                          /*    or "ru", "en", ... */

wsp_result* r = wsp_run(m, pcm, samples, &p);                 /* 3. audio in, result out */
if (r) {
    printf("language: %s\n", wsp_result_language(r));
    for (int i = 0; i < wsp_result_count(r); ++i)
        printf("%.2f-%.2f %s\n", wsp_result_t0(r, i), wsp_result_t1(r, i), wsp_result_text(r, i));
    wsp_result_free(r);
}
wsp_free(m);                                                  /* when the program no longer needs it */
```

Load the network once and keep it: loading takes a few seconds, running it
is what is repeated.

**Audio.** `pcm` is mono, 16 kHz, float samples in [-1, 1], `samples` their
count; any length. Convert other audio first, for example
`ffmpeg -i input.mp3 -ar 16000 -ac 1 output.wav`, and read the WAV (the
examples show how).

**What comes back.** Audio of up to 30 s is one segment. Longer audio is
cut at pauses into pieces of 5-15 s, each a segment with its start and end
in seconds. Texts are UTF-8 and stay valid until `wsp_result_free`.
`wsp_run` returns NULL only on bad arguments or an unknown language code.

**Threads.** A network uses all the processor's cores by itself
(`wsp_set_threads` to use fewer). One network handles one call at a time:
from several threads, either queue the calls or load the network once per
thread (about 1 GB of memory each).

**Speed** on a 2021 laptop processor (AMD Ryzen 5 5500U): about 6 s for a
10-second recording with `whisper_large_v3_turbo_m5d6.nll`, 5 s with
`whisper_large_v3_turbo_pp2.nll`; see [../docs/RESULTS.md](../docs/RESULTS.md).

### Live audio

For a microphone or any audio that arrives in portions, use a stream:

```c
wsp_stream* s = wsp_stream_new(m, "ru");      /* a stream needs its language */
/* as audio arrives: */
wsp_stream_push(s, portion, n);               /* transcribes each piece a pause has ended */
char text[65536];
if (wsp_stream_text(s, text, sizeof text) > 0) { /* new text since the last call */ }
/* at the end: */
wsp_stream_finish(s);
wsp_stream_text(s, text, sizeof text);
wsp_stream_free(s);
```

`wsp_stream_push` computes inside the call (seconds per piece): call it from
a worker thread, not from the audio callback or the user interface.

## The interface

| function | does |
|---|---|
| `wsp_load(path)` / `wsp_free(m)` | load a network file (NULL if it cannot) / release it |
| `wsp_set_threads(m, n)` | threads to use (default: all) |
| `wsp_default_params()` | settings: `language` "auto", `detect_each_window` 0, no callbacks |
| `wsp_run(m, pcm, samples, params)` | audio of any length to timed segments |
| `wsp_result_count`, `_t0`, `_t1`, `_text`, `_segment_language` | the segments: start and end in seconds, UTF-8 text, language |
| `wsp_result_language`, `_language_prob` | the detected (or given) language and its probability |
| `wsp_result_free(r)` | release the result |
| `wsp_transcribe(m, pcm, samples, lang, out, cap)` | the text of a short recording (its first 30 s) into a buffer |
| `wsp_detect_language(m, pcm, samples, code, cap, &prob)` | the language only |
| `wsp_stream_new / _cut / _push / _finish / _text / _redone / _free` | live audio, above |

`wsp_params` fields:

| field | meaning |
|---|---|
| `language` | a code ("ru", "en", ...; Whisper's codes) or "auto" / NULL to detect it from the first piece |
| `detect_each_window` | 1: detect the language again for every piece (audio that changes language) |
| `progress(done_sec, total_sec, user)` | optional: called after each piece |
| `segment(t0, t1, text, lang, user)` | optional: each segment as soon as it is ready |
| `user` | passed to both callbacks |

## Inside

| file | what |
|---|---|
| `src/Model.*` | reading the `.nll` file: settings, vocabulary, weight arrays (float, 8-bit, 2-8-bit with 16-bit group scales) |
| `src/Mel.*` | the log-mel spectrum (a 400-point FFT) |
| `src/Ops.*` | matrix products in packed panels, attention, softmax, layer norm, GELU; the thread pool |
| `src/Whisper.*` | the encoder and the decoder; greedy decoding, language detection |
| `src/Transcribe.*` | one window (the short encoder window and its fallback) and long audio (pieces cut at pauses) |
| `src/Segment.*` | cutting audio at pauses |
| `src/wsp_runtime.cpp` | the C interface |
