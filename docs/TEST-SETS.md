# Test sets

Three public sets of recorded speech. They are used **only for the final
check**: nothing about either network was chosen on them. Choices
(calibration, thresholds, which variant to keep) were made on separate
development data, listed at the end.

The exact recordings are listed in [`test-sets/`](test-sets/) (file
names for FLEURS, row numbers for Golos), so anyone can rebuild the same sets.

| set | language | kind of speech | recordings | audio | length of one recording (mean / min / max) | reference words | speakers |
|---|---|---|---|---|---|---|---|
| FLEURS ru, test | Russian | read sentences, quiet room | 775 | 2.50 h | 11.6 / 4.5 / 33.8 s | 14,752 | 413 recordings by women, 362 by men; 344 different sentences |
| FLEURS en, test (first 300) | English | read sentences, quiet room | 300 | 0.81 h | 9.7 / 3.6 / 29.3 s | 6,382 | 187 recordings by women, 113 by men; 161 different sentences |
| Golos crowd, test (298) | Russian | short voice-assistant requests, spoken into the speaker's own phone or headset | 298 | 0.33 h | 3.9 / 1.3 / 13.9 s | 1,473 | not given by the dataset |
| **all three** | | | **1,373** | **3.64 h** | | **22,607** | |

Reference words are counted after the normalization described below, the
same count the word error rate is divided by.

## FLEURS (Google)

- **What it is.** Sentences from Wikipedia (the FLoRes benchmark) read aloud
  by native speakers; about 12 hours per language in 102 languages. Clean
  speech, one sentence per recording.
- **Source.** [huggingface.co/datasets/google/fleurs](https://huggingface.co/datasets/google/fleurs),
  configurations `ru_ru` and `en_us`, split `test`, as published (CC BY 4.0).
- **Russian:** the whole test split, 775 recordings.
- **English:** the first 300 of the 647 recordings of the test split, in the
  order of the split's `test.tsv`.
- **Speakers.** FLEURS gives no speaker identifiers, only the speaker's
  gender per recording; so the number of different voices is not known. Each
  sentence was read by one to three people: in the Russian test, 344
  sentences in 775 recordings (167 read twice, 132 three times, 45 once).

## Golos crowd (SberDevices)

- **What it is.** Golos is a Russian speech corpus of about 1,240 hours
  collected by SberDevices for voice assistants. Its *crowd* part was
  recorded by crowd workers on their own devices: commands and requests of a
  few words, close to the microphone, various phones and headsets, various
  rooms.
- **Source.** The copy of the corpus at
  [huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd](https://huggingface.co/datasets/bond005/sberdevices_golos_10h_crowd),
  split `test`, its first file (`test-00000-of-00003`, 3,332 recordings):
  every 11th recording from the first, skipping those with an empty
  reference - 298 recordings. Original corpus:
  [github.com/salute-developers/golos](https://github.com/salute-developers/golos).
- **Speakers.** The copy has no speaker identifiers or gender.
- The corpus also has a *farfield* part (a smart speaker's microphones across
  a room). It is not among the tests: neither network is meant or tuned for
  far-field audio.

## How a recording is scored

- **Decoding.** Every model, including the PyTorch reference, gets the same
  task: the first 30 s of the recording (all recordings but a few long
  FLEURS ones are shorter), the language given (Russian or English), greedy
  decoding, no timestamps.
- **Normalization.** Both the reference and the model's text go through
  Whisper's basic normalizer: lower case; text in brackets dropped;
  punctuation, symbols and marks replaced by spaces; spaces collapsed. The
  same for both languages (Whisper's own English normalizer, which also
  rewrites numbers and spellings, is not used; so English figures here are
  comparable with each other, not with figures published elsewhere).
- **WER** - word edits (substitutions + deletions + insertions) over
  reference words; **CER** - character edits (spaces excluded) over
  reference characters. Both are summed over the whole set, not averaged per
  recording.
- **Intervals.** "95 % interval" of a difference between two models: the
  recordings drawn again with replacement 2,000 times, both models' errors
  together (paired bootstrap).
- **Numbers only.** Nobody listened to the recordings or read the
  transcriptions while this work was done: the programs compare texts and
  report counts.

## Development data (used for choices, not for these results)

Every choice - fitting the compression, the interpreter's bounds, which
variant to keep - was made on these, kept apart from the tests and never
mixed with them:

| set | recordings |
|---|---|
| FLEURS ru, `dev` split | 356 |
| FLEURS en, `dev` split | 394 |
| Golos crowd, `validation` split (200 drawn) | 200 |
| Golos farfield, `validation` split (200 drawn) | 200 |
