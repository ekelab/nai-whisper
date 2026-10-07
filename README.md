# NAI Whisper - a speech-recognition network compressed for the CPU

OpenAI's **Whisper large-v3-turbo** (open weights, MIT license) converted
into my own file format, run by my own interpreter on an ordinary laptop
processor - no PyTorch, no ONNX Runtime, no whisper.cpp - and compressed in
two versions. This repository is a test: the two networks, a small Windows
program to try them on your own voice, and the measurements.

## What was achieved

- **whisper_large_v3_turbo_m5d6 - 581 MB, 2.8× smaller than the original (1.63 GB).** Its word
  error rate is the original's within measurement noise on all three test
  sets - Russian and English read speech (FLEURS) and short Russian voice
  requests recorded on people's own phones (Golos crowd) - and lower on
  Russian FLEURS: 7.45 % against 8.12 %. It runs 1.7× faster than the
  original network in my interpreter and 2.6× faster than the PyTorch
  reference on the same processor.
- **whisper_large_v3_turbo_pp2 - 451 MB, 3.6× smaller.** 2× faster than the original in my
  interpreter, 3.2× faster than PyTorch. As accurate as the original on
  Russian; measurably worse on English (+0.36 WER points).
- Both run faster than real time on recordings of about 10 s on a 2021
  laptop processor (AMD Ryzen 5 5500U), in a 1.6 MB program with no
  dependencies, from a microphone or a file, in under 1 GB of memory.

## Why my network is better than the original - and where it is not

- **Smaller and faster**: the same network in 2.8–3.6 times less space and
  1.7–2 times less time per recording, on a CPU, without a machine-learning
  framework.
- **Fewer invented words on Russian read speech.** The original inserts
  words nobody said in 3.32 % of reference words; whisper_large_v3_turbo_m5d6 in 2.62 %, with the
  same substitutions and deletions. That is the whole of its lower error
  rate on Russian FLEURS. The gain is not proven beyond chance on 775
  recordings, and why the compressed network invents less is not yet
  established.
- **Elsewhere, the original's accuracy - no better, no worse** beyond
  noise. I did not make the network smarter; I made it cheaper without
  making it worse.

## Results

Word error rate, % (lower is better). Test sets:
[docs/TEST-SETS.md](docs/TEST-SETS.md).

| network | file | FLEURS ru (775 rec.) | FLEURS en (300) | Golos crowd (298) | time per recording ¹ |
|---|---|---|---|---|---|
| original, PyTorch reference | 1.63 GB | 8.12 | 6.77 | 20.98 | 15.5 s |
| original, my interpreter | 1.63 GB | 8.12 | 6.77 | 20.98 | 10.0 s |
| **whisper_large_v3_turbo_m5d6** | **581 MB** | **7.45** | 6.86 | **20.64** | **6.0 s** |
| **whisper_large_v3_turbo_pp2** | **451 MB** | 7.59 | 7.13 | 21.25 | **4.9 s** |

¹ The same 50 Russian recordings (10.3 s each on average), back to back;
AMD Ryzen 5 5500U, 12 threads.

My interpreter running the original network gives the same text as
PyTorch, recording for recording: identical in all 1,373 test recordings.
Differences from the original with their
95 % intervals, error kinds, and times on every set:
[docs/RESULTS.md](docs/RESULTS.md).

## Try it

1. Download `NaiWhisper-windows-x64.zip` from the
   [latest release](../../releases/latest) (the program and both networks,
   1 GB) and unpack it.
2. Run `NaiWhisper.exe`. Needs Windows 10 or 11 and an x86-64 processor
   with AVX2 (Intel since 2013, AMD since 2015); about 1 GB of free memory.
3. Choose the network and the language (Russian or English), press
   **Start microphone** and speak. The text appears after each pause in
   your speech, a few seconds behind. Each start of the microphone writes
   into a new block below the previous one. **Open WAV file…** transcribes
   a recording instead.

If nothing appears, check the input level in the status line, and that
Windows lets desktop programs use the microphone (Settings → Privacy →
Microphone).

## Your own program

[source/](source/) has the interpreter's source (C++, a C interface), the
instructions to build it and to connect it to a program in C, C++ or any
other language, and short examples: load the network, give it audio, get
timed text. A ready build for Windows (`nai_whisper.dll`, the header, the
examples) is in the [latest release](../../releases/latest); the network
files are in the release's program archive, folder `models/`. Free for
non-commercial use.

## How it was done

[docs/METHODS.md](docs/METHODS.md): the format and the interpreter, how each
version was compressed (quantization by least squares, pruning with a
least-squares refit), where the gains in size and speed come from, and what
part NAI played.

**NAI** (No-Ansatz Inference) is my system that finds formulas in data
without being told their form. Here its principles shaped the work (own
format and runtime, least squares instead of retraining, every choice made
on development data and checked on held-out data), and a law it found on a
small Whisper - what compressing a layer costs the network follows the
disturbance of the layer's output - gave, for a deep network, the pruning
plan of whisper_large_v3_turbo_pp2: early layers can be cut harder.

## Contents

| path | what |
|---|---|
| [Releases](../../releases/latest) | two archives: the program with both networks (`models/whisper_large_v3_turbo_m5d6.nll`, `models/whisper_large_v3_turbo_pp2.nll`; the networks are not attached separately), and the interpreter as a ready library (DLL, header, examples) |
| `docs/RESULTS.md` | all measurements |
| `docs/TEST-SETS.md` | the test sets: sources, sizes, hours, speakers, words |
| `docs/METHODS.md` | how the networks were made; the role of NAI |
| `docs/test-sets/` | the exact recordings of each test set |
| `source/` | the interpreter's source: build your own program with the networks |

## License

Documents: CC BY 4.0. Network files: MIT, derived from OpenAI's Whisper.
The program: free for non-commercial use. See [LICENSE.md](LICENSE.md) and
[NOTICE.md](NOTICE.md).
