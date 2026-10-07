# Copyright (c) 2026 Mikhail Smirnov. Free for non-commercial use; commercial
# use requires the written permission of the author (lab@mindscan.org).
# See LICENSE.md in the repository.

"""The network from Python through the shared library (ctypes), no other
package needed:

    python transcribe.py nai_whisper.dll model.nll audio.wav [language|auto]

(libnai_whisper.so on Linux). The WAV must be 16 kHz, 16-bit PCM, mono or
stereo; convert other audio first, e.g. ffmpeg -i in.mp3 -ar 16000 -ac 1 out.wav
"""
import ctypes as C
import sys
import wave


class Params(C.Structure):   # wsp_params in wsp_runtime.h
    _fields_ = [("language", C.c_char_p), ("detect_each_window", C.c_int),
                ("progress", C.c_void_p), ("segment", C.c_void_p), ("user", C.c_void_p)]


def main():
    lib_path, model_path, wav_path = sys.argv[1:4]
    language = sys.argv[4] if len(sys.argv) > 4 else "auto"
    lib = C.CDLL(lib_path)
    lib.wsp_load.restype = C.c_void_p
    lib.wsp_load.argtypes = [C.c_char_p]
    lib.wsp_default_params.restype = Params
    lib.wsp_run.restype = C.c_void_p
    lib.wsp_run.argtypes = [C.c_void_p, C.POINTER(C.c_float), C.c_longlong, C.POINTER(Params)]
    for name in ("wsp_result_t0", "wsp_result_t1"):
        getattr(lib, name).restype = C.c_double
        getattr(lib, name).argtypes = [C.c_void_p, C.c_int]
    lib.wsp_result_text.restype = C.c_char_p
    lib.wsp_result_text.argtypes = [C.c_void_p, C.c_int]
    lib.wsp_result_count.argtypes = [C.c_void_p]
    lib.wsp_result_language.restype = C.c_char_p
    lib.wsp_result_language.argtypes = [C.c_void_p]
    lib.wsp_result_free.argtypes = [C.c_void_p]
    lib.wsp_free.argtypes = [C.c_void_p]

    with wave.open(wav_path, "rb") as w:
        assert w.getframerate() == 16000 and w.getsampwidth() == 2, "a 16 kHz 16-bit WAV"
        ch = w.getnchannels()
        raw = w.readframes(w.getnframes())
    ints = memoryview(raw).cast("h")
    pcm = [sum(ints[i:i + ch]) / (32768.0 * ch) for i in range(0, len(ints), ch)]
    audio = (C.c_float * len(pcm))(*pcm)

    model = lib.wsp_load(model_path.encode())                 # 1. the network
    if not model:
        sys.exit("cannot load " + model_path)
    params = lib.wsp_default_params()                         # 2. the settings
    params.language = language.encode()
    result = lib.wsp_run(model, audio, len(pcm), C.byref(params))   # 3. audio in, result out
    if not result:
        sys.exit("unknown language " + language)
    print("language:", lib.wsp_result_language(result).decode())
    for i in range(lib.wsp_result_count(result)):
        print("[%8.2f --> %8.2f] %s" % (lib.wsp_result_t0(result, i), lib.wsp_result_t1(result, i),
                                        lib.wsp_result_text(result, i).decode()))
    lib.wsp_result_free(result)
    lib.wsp_free(model)


if __name__ == "__main__":
    main()
