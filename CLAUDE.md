# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

sherpa-onnx is a speech processing framework supporting ASR (streaming/non-streaming), TTS, speaker diarization, VAD, keyword spotting, audio tagging, and more. It provides bindings for 12 languages (C++, C, Python, JavaScript, Java, Kotlin, C#, Swift, Go, Dart, Rust, Pascal) and runs on Linux, Windows, macOS, Android, iOS, HarmonyOS, WebAssembly, and various NPUs (Rockchip, Qualcomm, Ascend, Axera).

## Build Commands

### Standard CMake Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Release ..
cmake --build . --config Release
```

### Common CMake Options
- `-DSHERPA_ONNX_ENABLE_PYTHON=ON` - Build Python bindings
- `-DSHERPA_ONNX_ENABLE_TESTS=ON` - Build tests
- `-DSHERPA_ONNX_ENABLE_JNI=ON` - Build JNI for Android/Java
- `-DSHERPA_ONNX_ENABLE_C_API=ON` - Build C API (default ON)
- `-DSHERPA_ONNX_ENABLE_GPU=ON` - Enable CUDA support
- `-DSHERPA_ONNX_ENABLE_DIRECTML=ON` - Enable DirectML (Windows)
- `-DBUILD_SHARED_LIBS=ON` - Build shared libraries

### Python Package
```bash
pip install -e .
# or with specific args
SHERPA_ONNX_CMAKE_ARGS="-DSHERPA_ONNX_ENABLE_GPU=ON" pip install -e .
```

### Platform-Specific Builds
Platform-specific build scripts are in the root directory:
- `build-android-*.sh` - Android builds
- `build-ios*.sh` - iOS builds
- `build-wasm-*.sh` - WebAssembly builds
- `build-*-linux-*.sh` - Cross-compilation for ARM/RISC-V

### Running Tests
Tests are GoogleTest, built only with `-DSHERPA_ONNX_ENABLE_TESTS=ON`. Each `*-test.cc` becomes its own executable named after the file, registered with ctest. Add new test files to `sherpa_onnx_test_srcs` in `sherpa-onnx/csrc/CMakeLists.txt`.
```bash
cd build && ctest --output-on-failure                            # all tests
cd build && ctest -R circular-buffer-test --output-on-failure    # single test
./bin/circular-buffer-test --gtest_filter='Suite.Case'           # single case

# Python tests (pytest style; some need downloaded models)
cd sherpa-onnx/python/tests
pytest
```

## Code Architecture

### Core Structure
```
sherpa-onnx/csrc/          # C++ core implementation
├── offline-*              # Non-streaming models and recognizers
├── online-*               # Streaming models and recognizers
├── speaker-*              # Speaker diarization/identification
├── keyword-spotter-*      # Keyword spotting
└── *-impl.h/cc            # Implementation details
```

### API Layers
Language bindings are thin layers over the C++ core, and a new config field/feature usually has to be mirrored in each: `sherpa-onnx/c-api` (C API + `cxx-api`), `sherpa-onnx/python/csrc` (pybind11) and `python/sherpa_onnx`, `sherpa-onnx/jni` + `java-api` + `kotlin-api`, `rust`, `pascal-api`, and the wrappers in `swift-api-examples`, `dotnet-examples`, `go-api-examples`, `nodejs-addon-examples`, `flutter`, `dart-api-examples`, `harmony-os`, `wasm`. Each language has a `*-examples` dir that CI tests.

Many bindings (Go, C#, Dart, Swift, Pascal, JS) mirror C API struct layouts. When adding fields to a C API struct in this fork, append them at the end so upstream bindings stay layout-compatible.

### Model-Dispatch Pattern
Public facades (`OnlineRecognizer`, `OfflineRecognizer`, `OfflineTts`, ...) hold a `*Impl` created by `*Impl::Create(config)` in `*-impl.cc`, which inspects the config / model metadata to pick a concrete implementation (`offline-recognizer-<model>-impl.h`, `online-recognizer-{transducer,ctc,paraformer}-impl.h`). Each model family has its own `*-model.{h,cc}`, `*-model-config.{h,cc}` and `*-model-meta-data.h`. Adding a model = new model + config + impl + registration in the dispatcher + `sources` in `csrc/CMakeLists.txt` + `Config` plumbing in every binding.

### Model Types
- **Transducer models**: Zipformer, Conformer (streaming and non-streaming)
- **CTC models**: Paraformer, NeMo, WeNet
- **Whisper**: OpenAI Whisper variants
- **TTS**: Piper, Matcha, Kokoro, VITS, MeloTTS

### Configuration Pattern
Models are configured via config structs rather than hardcoded values. Configs are structs with `Register(ParseOptions*)`, `Validate()`, `ToString()`; CLI flags come from `Register`, and bindings expose the same fields. See `*-model-config.h` files.

## Code Style

### C++
- Google C++ style (configured in `.clang-format`)
- Pointer alignment: Right (`int *p`)
- Static analysis: clang-tidy with warnings as errors
- Lint: cpplint 2.0.2 via `./scripts/check_style_cpplint.sh` (no arg = last commit, `1` = uncommitted changes, `2` = whole repo). Config in `CPPLINT.cfg`; CI is `.github/workflows/style_check.yaml`.
- Headers use `SHERPA_ONNX_CSRC_<NAME>_H_` guards and a `// sherpa-onnx/csrc/<file>` + copyright header comment.

### Python
- flake8 with max line length 120
- Excludes: `.git`, `./cmake`

## Key Directories

- `scripts/` - Model export/conversion scripts (not part of the build)
- `cmake/` - Per-dependency fetch scripts (onnxruntime, kaldi-native-fbank, etc.), downloaded at configure time
- `*-api-examples/` - Usage examples for each language
- `android/` - Android applications and libraries
- `flutter/` - Flutter packages
- `wasm/` - WebAssembly builds for browser/Node.js
- `.github/workflows/` - 130+ CI workflows for all platforms

## Version Management

Version is defined in `CMakeLists.txt` line 17 (`SHERPA_ONNX_VERSION`). When updating, also update:
- `CHANGELOG.md`
- `new-release.sh`

## Build and Run C++ Examples

The `cxx-api-examples/` directory contains C++ demo examples for various features (ASR, TTS, VAD, etc.).

### Build Examples
```bash
mkdir build && cd build

# Build with shared ONNX runtime (faster builds)
cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
cmake --build . --config Release -j 8
```

### Run Examples
After building, executables are in `build/bin/` (Linux/macOS), `build-ninja\bin\` (Windows, Ninja script) or `build/bin/Release/` (Windows, VS generator):
```bash
# Example: Run SenseVoice ASR
./bin/sense-voice-cxx-api

# Example: Run streaming zipformer
./bin/streaming-zipformer-cxx-api

# Example: Run VAD
./bin/vad-cxx-api
```

Note: Most examples require downloading models first. Check the source file for model paths and download instructions.

### Available Examples
- **ASR**: `sense-voice-cxx-api`, `whisper-cxx-api`, `streaming-zipformer-cxx-api`, `funasr-nano-cxx-api`, etc.
- **TTS**: `matcha-tts-*-cxx-api`, `kokoro-tts-*-cxx-api` (require `-DSHERPA_ONNX_ENABLE_TTS=ON`)
- **VAD**: `vad-cxx-api`
- **Audio Tagging**: `audio-tagging-ced-cxx-api`, `audio-tagging-zipformer-cxx-api`
- **Microphone**: `*-microphone-cxx-api` (require `-DSHERPA_ONNX_ENABLE_PORTAUDIO=ON`)

## Build Fast

- Don't `rm -rf build/` every time - incremental builds are much faster
- Use parallel builds: `cmake --build . --config Release -j N` (N = number of CPU cores)
- Use shared ONNX runtime for faster link times:
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=ON ..
  ```
- Static builds (`BUILD_SHARED_LIBS=OFF`, default) produce standalone binaries but take longer to link

### IMPORTANT: Windows MSVC Static Build Issue

**DO NOT use static onnxruntime on Windows** (`BUILD_SHARED_LIBS=OFF`). The pre-compiled static onnxruntime library has MSVC ABI compatibility issues with newer Visual Studio versions (19.43+). You will get linker errors like:
```
error LNK2019: unresolved external symbol __std_find_first_of_trivial_pos_1
error LNK2019: unresolved external symbol __std_find_first_of_trivial_pos_2
```

**Solution**: Always use `-DBUILD_SHARED_LIBS=ON` for Windows MSVC builds:
```bash
cmake -G "Visual Studio 17 2022" -A x64 -DBUILD_SHARED_LIBS=ON ..
```

### Windows: build with Ninja (default for day-to-day work)

Use the Ninja script. It finds Visual Studio via `vswhere`, enters the x64
developer environment, uses VS's bundled Ninja (not `depot_tools`' wrapper on
PATH), and builds into `build-ninja\` with `BUILD_SHARED_LIBS=ON`:
```bash
build-windows-ninja.bat                                  # configure + build
build-windows-ninja.bat -DSHERPA_ONNX_ENABLE_TESTS=ON    # extra CMake args
```
From Git Bash call it as `cmd.exe //c "D:\\project\\sherpa-onnx\\build-windows-ninja.bat ..."`.
Executables are in `build-ninja\bin\`; run tests with `ctest --test-dir build-ninja`.
All tests pass on Windows except `lfr-test`, whose death tests do not capture
the expected stderr message under MSVC.

The onnxruntime DLLs must sit next to the executables. Otherwise Windows loads
the older `C:\Windows\System32\onnxruntime.dll` (1.17) that ships with the OS,
and every program crashes with access violations (0xc0000005).
`cmake/onnxruntime.cmake` copies the DLLs to `bin\` for single-config
generators such as Ninja.

ccache is enabled automatically if found (`D:\soft\ccache`, or set `CCACHE_DIR`).
With Ninja it is used as `CMAKE_<LANG>_COMPILER_LAUNCHER`, and header
dependencies come from the compiler (`/showIncludes`), so incremental builds
are exact.

### Windows: Visual Studio generator (IDE only)

`build\` uses the Visual Studio generator. With ccache, CMakeLists.txt copies
ccache as `cl.exe` and enables MSBuild `UseMultiToolTask`. **MultiToolTask's
incremental tracking can miss header changes**
([msbuild#15229](https://github.com/dotnet/msbuild/issues/15229), not planned to
be fixed). In this repo that once left stale `.obj` files after a struct
layout change, and the binaries segfaulted. If you build with the VS generator,
use `cmake --build build --config Release --clean-first` after editing headers
(ccache keeps this fast).

Verify ccache is working:
```bash
ccache -s  # Check statistics before/after build
```
