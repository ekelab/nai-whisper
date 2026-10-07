// Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
// use requires the written permission of the author (lab@mindscan.org).
// See LICENSE.md in the repository.

#pragma once
// Audio cut into pieces at pauses, so that a long recording, or a live
// stream, can be transcribed piece by piece.
//
// Frames of 20 ms; a frame is speech when its energy (dB) is above a
// threshold that follows the audio heard so far: the noise floor (a low
// percentile) plus a share of the way to the speech level (a high
// percentile). A piece ends at the middle of the first pause (≥ pauseSec of
// non-speech) once it is minSec long; if none comes by maxSec, at the
// quietest frame of its last seconds. The decisions use only audio already
// heard: the same cutter serves a file and a live stream.

#include <vector>

namespace wsp {

struct CutOptions {
    // Chosen by measurement: pieces of one phrase or so - longer
    // ones lose words (no timestamps), shorter ones break more often.
    double minSec = 5, maxSec = 15, pauseSec = 0.6;
};

class Cutter {
public:
    explicit Cutter(CutOptions o = {}) : o_(o) {}
    /** More audio (16 kHz mono). */
    void push(const float* pcm, int n);
    /** The end of the next piece (sample index from the start), or −1 if
     *  none is decided yet; final: the audio is over - the rest is a piece. */
    long long next(bool final);

private:
    CutOptions o_;
    std::vector<float> energy_;   // dB per frame, all frames heard
    std::vector<float> partial_;  // the samples of an unfinished frame
    long long start_ = 0;         // the current piece's first sample
    long long heard_ = 0;         // samples heard
    double threshold() const;
};

}  // namespace wsp
