# Results

## In short

**What was achieved.** OpenAI's Whisper large-v3-turbo runs in my own
file format, by my own interpreter, on an ordinary laptop processor,
without PyTorch or any other engine. Two compressed versions:

- **whisper_large_v3_turbo_m5d6, 581 MB** (2.8× smaller than the original's 1.63 GB). On all three
  test sets its word error rate is the original's within measurement noise,
  lower on Russian FLEURS (7.45 % against 8.12 %). It runs 2.6× faster than
  the PyTorch reference on the same processor, 1.7× faster than the original
  network in my interpreter.
- **whisper_large_v3_turbo_pp2, 451 MB** (3.6× smaller). 3.2× faster than PyTorch. As accurate as
  the original on Russian speech (lower on FLEURS, equal within noise on
  Golos crowd), but measurably worse on English (+0.36 points, the whole
  95 % interval above zero).

**Why my networks are better than the original, and in what.**

- *Size and speed*: the same network in 2.8–3.6 times less space and 1.7–2×
  less time per recording on the same CPU, with no runtime dependencies (a
  1.6 MB Windows program).
- *Fewer invented words on Russian read speech*: the original inserts words
  that were not said in 3.32 % of reference words, whisper_large_v3_turbo_m5d6 in 2.62 %, with the
  same substitutions and deletions. That is the whole of its lower WER on
  Russian FLEURS. The difference is not proven beyond chance (95 % interval
  −2.26 to +0.21 points), and why the compressed network invents less is
  not established; I checked and rejected one explanation (the silence the
  original pads short recordings with).
- *Accuracy is otherwise the original's*: not better on English or on
  Golos crowd, and not worse beyond noise.

## The machine

Laptop with an AMD Ryzen 5 5500U (Zen 2, 6 cores / 12 threads, 2.1–4.0
GHz, AVX2 and FMA, no AVX-512), 16 GB of memory, Windows 11. All 12 threads
are used. Under long load the laptop lowers its clock, so times taken on
different days are not comparable; the speed comparison (table 4) was run
back to back, and the first variant measured again at the end gave the
same time.

## 1. Accuracy on the test sets

WER and CER, %, summed over each set; the sets are described in
[TEST-SETS.md](TEST-SETS.md). "Original" is Whisper large-v3-turbo as
published (16-bit weights, computed in 32-bit floats, the full 30 s
window).

| network | runs on | file | FLEURS ru (775) | FLEURS en (300) | Golos crowd (298) |
|---|---|---|---|---|---|
| original | PyTorch (reference) | 1.63 GB | 8.12 / 3.93 | 6.77 / 4.19 | 20.98 / 13.59 |
| original | my interpreter | 1.63 GB | 8.12 / 3.93 | 6.77 / 4.19 | 20.98 / 13.59 |
| **whisper_large_v3_turbo_m5d6** | my interpreter | **581 MB** | **7.45 / 3.37** | 6.86 / 4.20 | **20.64 / 13.82** |
| **whisper_large_v3_turbo_pp2** | my interpreter | **451 MB** | **7.59 / 3.42** | 7.13 / 4.35 | 21.25 / 13.99 |

Each cell: WER / CER.

My interpreter running the original network gives the same text as
PyTorch, recording for recording, on all three sets: 775 of 775, 300 of 300
and 298 of 298 texts identical. So the rows "original" of this report are
the published network itself, not an approximation of it.

## 2. Difference from the original

WER points (my network minus the original), with the 95 % interval by the
paired bootstrap over recordings; then how many recordings got fewer / more
word errors than with the original (the rest: the same number).

| set | whisper_large_v3_turbo_m5d6 | recordings better / worse | whisper_large_v3_turbo_pp2 | recordings better / worse |
|---|---|---|---|---|
| FLEURS ru | −0.67 [−2.26, +0.21] | 36 / 43 | −0.53 [−2.17, +0.39] | 38 / 62 |
| FLEURS en | +0.09 [−0.16, +0.37] | 20 / 20 | **+0.36 [+0.08, +0.65]** | 17 / 34 |
| Golos crowd | −0.34 [−1.50, +0.77] | 20 / 21 | +0.27 [−0.91, +1.53] | 18 / 23 |

An interval that contains zero means the sets are too small to tell the two
networks apart. Only whisper_large_v3_turbo_pp2 on English is outside: worse.

On Russian FLEURS more recordings come out worse than better for both
networks, yet the total is lower: the gain is concentrated in a few
recordings where the original adds many words, the losses are single words
spread over many recordings.

## 3. Kinds of errors

% of reference words.

| set | network | substitutions | deletions | insertions | text > 1.5× the reference |
|---|---|---|---|---|---|
| FLEURS ru | original | 4.41 | 0.39 | **3.32** | 15 recordings |
| | whisper_large_v3_turbo_m5d6 | 4.43 | 0.40 | **2.62** | 14 |
| | whisper_large_v3_turbo_pp2 | 4.53 | 0.43 | 2.64 | 14 |
| FLEURS en | original | 3.70 | 0.78 | 2.29 | 4 |
| | whisper_large_v3_turbo_m5d6 | 3.70 | 0.83 | 2.33 | 4 |
| | whisper_large_v3_turbo_pp2 | 3.95 | 0.72 | 2.46 | 4 |
| Golos crowd | original | 14.66 | 5.23 | 1.09 | 0 |
| | whisper_large_v3_turbo_m5d6 | 14.60 | 5.23 | 0.81 | 0 |
| | whisper_large_v3_turbo_pp2 | 15.34 | 5.02 | 0.88 | 0 |

## 4. Speed, back to back

The first 50 recordings of the Russian FLEURS test (0.143 h, 10.3 s each on
average), one after another, in this order.

| network | runs on | per recording | encoder | decoder | time / audio length | vs PyTorch | WER on these 50 |
|---|---|---|---|---|---|---|---|
| whisper_large_v3_turbo_m5d6 | my interpreter | 6.0 s | 5.5 s | 0.52 s | 0.58 | 2.6× faster | 3.36 % |
| whisper_large_v3_turbo_pp2 | my interpreter | 4.9 s | 4.3 s | 0.51 s | 0.47 | 3.2× faster | 3.69 % |
| original | my interpreter | 10.0 s | 8.7 s | 1.28 s | 0.97 | 1.55× faster | 4.14 % |
| original | PyTorch (reference) | 15.5 s | | | 1.50 | 1× | 4.14 % |
| whisper_large_v3_turbo_m5d6, again | my interpreter | 6.0 s | 5.4 s | 0.51 s | 0.58 | | 3.36 % |

"Time / audio length" below 1 means faster than real time: the network
keeps up with live speech.

## 5. Time on the whole test sets

Seconds per recording; each network's run of a set in one go (different
hours or days, so only roughly comparable; table 4 is the fair
comparison). In brackets: recordings done again with the full window.

| set (mean audio length) | original, my interpreter | whisper_large_v3_turbo_m5d6 | whisper_large_v3_turbo_pp2 |
|---|---|---|---|
| FLEURS ru (11.6 s) | 10.4 | 6.2 (3 of 775) | 5.4 (7 of 775) |
| FLEURS en (9.7 s) | 10.7 | 5.7 (0 of 300) | 5.0 (0 of 300) |
| Golos crowd (3.9 s) | 9.5 | 6.1 (15 of 298) | 5.3 (17 of 298) |

## 6. Files

| file | size | inside |
|---|---|---|
| `whisper_large_v3_turbo_m5d6.nll` (release archive, `models/`) | 581,165,601 bytes | encoder 5 bits, decoder 6 bits |
| `whisper_large_v3_turbo_pp2.nll` (release archive, `models/`) | 451,441,085 bytes | encoder partly pruned, 4 bits; decoder 6 bits |
| `NaiWhisper.exe` (release archive) | 1.6 MB | the program: microphone or WAV file to text; Windows 10/11, x86-64 with AVX2 |

Memory: the program's peak working set while transcribing a recording
(the network loaded, one piece done again with the full window) - 0.96 GB
with whisper_large_v3_turbo_m5d6, 0.81 GB with whisper_large_v3_turbo_pp2.
