# How the two networks were made

Both files are the same network - OpenAI's **Whisper large-v3-turbo** (open
weights, MIT license) - converted into my own file format and compressed.
Nothing was retrained: no gradient descent, no GPU. Every change was made
by fitting on the CPU and checked by numbers against the original network.

| | original | **whisper_large_v3_turbo_m5d6** | **whisper_large_v3_turbo_pp2** |
|---|---|---|---|
| file | 1.63 GB (16-bit floats) | **581 MB** | **451 MB** |
| encoder (32 layers) | 16-bit | 5 bits per weight | part of its units removed; 4 bits per weight |
| decoder (4 layers) and vocabulary | 16-bit | 6 bits | 6 bits |
| aim | - | accuracy of the original on all kinds of speech | smallest and fastest |

## 1. What both have in common

### Own format and own interpreter

- **Format (`.nll` files).** One file: the network's settings, the
  vocabulary, the mel filters and the weight arrays, each stored at its own
  precision - from 32-bit floats down to a few bits per weight with shared
  scales.
- **Interpreter.** A C++ library with a C interface (load a network,
  transcribe a recording, a stream interface for live audio), using nothing
  but the C++ standard library. No PyTorch, ONNX Runtime, whisper.cpp or any
  other engine runs the network. Its source is in [source/](../source/).
- **Checked against the original.** Before any compression, my interpreter
  and the reference (HuggingFace transformers on PyTorch) were compared
  array by array and on whole test sets: the same text in all 1,373 test
  recordings, the same WER (see [RESULTS.md](RESULTS.md)).

### Arithmetic tuned for a CPU

- Matrix products in packed blocks sized for the processor's registers and
  caches, with its vector instructions (AVX2, FMA).
- Compressed weights are expanded to floats just before use: the processor
  reads a few bits per weight from memory instead of 16, and computes in
  floats.

### A shorter look at the audio (the largest speed gain)

Whisper's encoder always reads a 30-second window: a 4-second phrase is
padded with 26 seconds of silence, and the encoder - most of the work -
processes all of it. My interpreter lets the encoder see only the audio
plus a margin, but at least 15 seconds; if the result looks broken
(repeats, too many words for the audio), the recording is done again with
the full window, as the model was trained. The bounds were chosen on
development data, never on the tests.

### Live audio

For the microphone the audio is cut at pauses into pieces of a few seconds,
and each piece is transcribed as soon as it ends. So text appears after
each pause, a few seconds behind the speaker; Whisper does not produce text
word by word.

## 2. whisper_large_v3_turbo_m5d6 - accurate on every kind of speech

- **Quantization by least squares.** Each layer's weights are rounded to a
  few bits so that the layer's output on real speech stays as close as
  possible to the original - not just each weight to its own value.
- **Calibrated on several kinds of speech**, not on one: the speech used to
  fit the rounding decides where the network stays accurate.
- **5 bits in the encoder.** At 4 bits the network differed from the
  original several times more on English and on short voice requests than
  on the Russian read speech it had been fitted on; more varied calibration
  did not close the gap, one more bit did.
- **No pruning**: on varied speech, removing units cost as much accuracy as
  the quantization itself.

## 3. whisper_large_v3_turbo_pp2 - the smallest and fastest

- **Pruning with a refit.** Part of the encoder's units that matter least
  are removed, and the rest of the layer is refitted by least squares so
  that the layer's output stays close to the original - a smaller layer
  standing in for the bigger one. Early layers are cut harder than late
  ones (section 5).
- **4 bits in the encoder**, quantized by least squares as above.
- Calibrated on Russian read speech only: on Russian it is as accurate as
  the original or better; on English it is measurably worse (see
  RESULTS.md) - that is why whisper_large_v3_turbo_m5d6 was made.

## 4. Where the gains come from

### Size

| | bits per encoder weight | bits per decoder weight | file |
|---|---|---|---|
| original | 16 | 16 | 1.63 GB |
| whisper_large_v3_turbo_m5d6 | about 5 | about 6 | 581 MB (2.8× smaller) |
| whisper_large_v3_turbo_pp2 | about 4, fewer units | about 6 | 451 MB (3.6× smaller) |

### Speed

Measured back to back on the same 50 Russian recordings (10.3 s each on
average; [RESULTS.md](RESULTS.md), table 4):

| step | per recording | gain |
|---|---|---|
| PyTorch reference, float32 | 15.5 s | |
| my interpreter, float32, full 30 s window | 10.0 s | 1.55×: arithmetic tuned for the CPU, the same text |
| whisper_large_v3_turbo_m5d6 | 6.0 s | 1.66× more: the shorter window for the encoder; the decoder, limited by reading its weights from memory, runs 2.5× faster on compressed weights |
| whisper_large_v3_turbo_pp2 | 4.9 s | 1.23× more: fewer units to compute |

## 5. The role of NAI

NAI (No-Ansatz Inference) is my system that finds formulas - laws - in
data without being told their form. In this work:

- **Principles taken from NAI.** Own format and own interpreter instead of
  third-party engines; fitting by least squares instead of gradient
  descent; every decision made on development data and checked on held-out
  data, by numbers only.
- **A law found with NAI.** On a small Whisper, NAI found how the cost of
  compressing a layer depends on what the compression does to the layer's
  output. In the deep turbo network the same measurements showed that
  early layers' errors are damped by the layers after them - so early
  layers can be compressed harder. That set how whisper_large_v3_turbo_pp2
  was pruned.
- **Where NAI did not help.** Choosing a bit width for every matrix from
  that law gave a network smaller but more divergent: rejected.
  whisper_large_v3_turbo_m5d6 was chosen by direct measurement.

## 6. What is not mine

The network - its architecture and its training on hundreds of thousands of
hours of speech - is OpenAI's. Quantization by least squares follows the
published idea of GPTQ (Frantar et al., 2022); packed matrix products
follow the BLIS/GotoBLAS design; redoing a broken decoding with the full
window follows Whisper's own practice of re-decoding when the output repeats
itself. My contribution is the working combination without dependencies,
each step checked by numbers against the reference, and the measurements
on a specific processor, negative results included.
