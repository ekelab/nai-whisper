# License

Copyright © 2026 Mikhail Smirnov.

This repository contains four kinds of material, under different terms.

## 1. Reports, documentation and results - CC BY 4.0

All Markdown documents and the lists of test recordings in `docs/test-sets/` are
licensed under the Creative Commons Attribution 4.0 International license
(CC BY 4.0): https://creativecommons.org/licenses/by/4.0/

## 2. The network files - MIT

The files `whisper_large_v3_turbo_m5d6.nll` and `whisper_large_v3_turbo_pp2.nll`,
published in this repository's release (the `models/` folder of
`NaiWhisper-windows-x64.zip`), are derived
from OpenAI's Whisper large-v3-turbo weights (MIT license, see
[NOTICE.md](NOTICE.md)): converted into another file format, quantized and,
for `whisper_large_v3_turbo_pp2.nll`, pruned and refitted. They are licensed under the MIT license:

> Copyright (c) 2022 OpenAI (the original weights)
> Copyright (c) 2026 Mikhail Smirnov (the conversion, quantization and pruning)
>
> Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
>
> The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
>
> THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

## 3. The program - non-commercial use only

The file `NaiWhisper.exe` (in the release archive `NaiWhisper-windows-x64.zip`) is proprietary and is not open source.

You may, free of charge and for **non-commercial purposes only**:
- download and run it to check the results published in this repository;
- run it on your own audio, to evaluate it, for research or for teaching;
- publish the results you obtain, with attribution.

**Commercial use of any kind requires the prior written permission of the
author, Mikhail Smirnov (lab@mindscan.org).** This includes:
- use in or for a commercial product or service;
- use for a company's business or on a client's behalf;
- paid work.

You may not:
- redistribute the program outside a copy or fork of this repository;
- incorporate it into software or services offered to others;
- modify, decompile or reverse-engineer it.

The program is provided "as is", without warranty of any kind.

## 4. The interpreter source - non-commercial use only

The files in `source/` (the interpreter of the network files and its
examples) and the library built from them (`NaiWhisper-library-windows-x64.zip`
in the releases) are not open source; their source is published so that anyone can
build their own non-commercial programs with the networks.

You may, free of charge and for **non-commercial purposes only**:
- read, build and modify the source;
- use it, modified or not, in your own programs;
- distribute it and programs built with it, with this license and the
  copyright notice kept in every source file.

**Commercial use of any kind requires the prior written permission of the
author, Mikhail Smirnov (lab@mindscan.org)**, as for the program (section
3).

The source is provided "as is", without warranty of any kind.

## Third-party material

Third-party data and software keep their own licenses; see
[NOTICE.md](NOTICE.md).
