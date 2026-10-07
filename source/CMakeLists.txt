cmake_minimum_required(VERSION 3.15)
project(NaiWhisperInterpreter C CXX)

# The interpreter of the .nll network files, and a small example program.
#
#     cmake -S . -B build && cmake --build build -j
#
# nai_whisper         static library, for C and C++ programs; its interface
#                     is include/wsp_runtime.h (C). Inside: the C++20
#                     standard library only.
# nai_whisper_shared  the same as a shared library (nai_whisper.dll,
#                     libnai_whisper.so, libnai_whisper.dylib), for any other language (Python,
#                     Go, C#, Java, ...): only the functions of
#                     wsp_runtime.h are exported.
# transcribe          example/transcribe.c: a WAV file to timed text.

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")
set(CMAKE_C_FLAGS_RELEASE "-O2 -DNDEBUG")
# On x86-64: AVX2 + FMA (x86-64-v3), Intel since 2013, AMD since 2015; OFF:
# any x86-64, slower. Other processors (Apple silicon, ARM) get the
# compiler's own vector instructions. The vector code uses GCC/Clang vector
# extensions: GCC or Clang is needed (not MSVC).
option(NAI_WHISPER_AVX2 "On x86-64, build for x86-64-v3 (AVX2, FMA)" ON)
set(NAI_WHISPER_X86 OFF)
if(CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|amd64)$" AND NOT CMAKE_OSX_ARCHITECTURES MATCHES "arm64")
    set(NAI_WHISPER_X86 ON)
endif()

find_package(Threads REQUIRED)
# GCC notes that 32-byte vectors change the calling convention when AVX is
# off; they are passed only between the library's own functions.
add_compile_options($<$<CXX_COMPILER_ID:GNU>:-Wno-psabi>)

option(NAI_WHISPER_SHARED "Also build the shared library" ON)

set(NAI_WHISPER_SOURCES
    src/Model.cpp src/Mel.cpp src/Ops.cpp src/Whisper.cpp src/Segment.cpp src/Transcribe.cpp src/wsp_runtime.cpp)

add_library(nai_whisper STATIC ${NAI_WHISPER_SOURCES})
target_include_directories(nai_whisper PUBLIC include PRIVATE src)
if(NAI_WHISPER_AVX2 AND NAI_WHISPER_X86)
    target_compile_options(nai_whisper PRIVATE -march=x86-64-v3)
endif()
target_link_libraries(nai_whisper PUBLIC Threads::Threads)
# A C program links the C++ standard library through the C++ linker.
set_target_properties(nai_whisper PROPERTIES LINKER_LANGUAGE CXX)

if(NAI_WHISPER_SHARED)
    add_library(nai_whisper_shared SHARED ${NAI_WHISPER_SOURCES})
    target_include_directories(nai_whisper_shared PUBLIC include PRIVATE src)
    target_compile_definitions(nai_whisper_shared PRIVATE WSP_BUILD_SHARED)
    if(NAI_WHISPER_AVX2 AND NAI_WHISPER_X86)
        target_compile_options(nai_whisper_shared PRIVATE -march=x86-64-v3)
    endif()
    target_link_libraries(nai_whisper_shared PRIVATE Threads::Threads)
    set_target_properties(nai_whisper_shared PROPERTIES OUTPUT_NAME nai_whisper
        CXX_VISIBILITY_PRESET hidden VISIBILITY_INLINES_HIDDEN ON)
    if(WIN32)
        set_target_properties(nai_whisper_shared PROPERTIES PREFIX "")   # nai_whisper.dll
    endif()
    if(MINGW)
        # The C++ runtime inside the DLL: nothing else to ship with it.
        target_link_options(nai_whisper_shared PRIVATE -static -s)   # -s: no symbol table
    endif()
endif()

add_executable(transcribe example/transcribe.c)
target_link_libraries(transcribe PRIVATE nai_whisper)
set_target_properties(transcribe PROPERTIES LINKER_LANGUAGE CXX)
if(MINGW)
    target_link_options(transcribe PRIVATE -static)
endif()
