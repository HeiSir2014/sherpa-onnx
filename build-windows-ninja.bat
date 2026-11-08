@echo off
rem Build sherpa-onnx on Windows with MSVC + Ninja (+ ccache if found).
rem
rem Ninja takes header dependencies from the compiler (/showIncludes), so
rem incremental builds after a header change are reliable. The Visual Studio
rem generator with ccache relies on MSBuild MultiToolTask, which can miss them.
rem
rem Usage (from the repo root, in any shell):
rem   build-windows-ninja.bat                       configure + build
rem   build-windows-ninja.bat -DSHERPA_ONNX_ENABLE_TESTS=ON   extra CMake args
rem
rem The build directory is build-ninja\. Executables go to build-ninja\bin\.

setlocal

set "VSWHERE=%ProgramFiles(x86)%\Microsoft Visual Studio\Installer\vswhere.exe"
if not exist "%VSWHERE%" (
  echo vswhere.exe not found; install Visual Studio 2022 with C++ tools.
  exit /b 1
)
for /f "usebackq tokens=*" %%i in (`"%VSWHERE%" -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath`) do set "VSDIR=%%i"
if not defined VSDIR (
  echo Visual Studio with the C++ x64 toolset not found.
  exit /b 1
)

call "%VSDIR%\VC\Auxiliary\Build\vcvars64.bat" >nul || exit /b 1

rem Use the Ninja that ships with Visual Studio, not another one on PATH.
set "NINJA=%VSDIR%\Common7\IDE\CommonExtensions\Microsoft\CMake\Ninja\ninja.exe"
if not exist "%NINJA%" (
  echo Ninja not found at "%NINJA%"; install the "C++ CMake tools" component.
  exit /b 1
)

rem Shared libs: the prebuilt static onnxruntime has MSVC ABI issues (CLAUDE.md).
cmake -S . -B build-ninja -G Ninja ^
  -DCMAKE_MAKE_PROGRAM="%NINJA%" ^
  -DCMAKE_BUILD_TYPE=Release ^
  -DBUILD_SHARED_LIBS=ON ^
  %* || exit /b 1

cmake --build build-ninja || exit /b 1
