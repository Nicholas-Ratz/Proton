cmake_minimum_required(VERSION 3.21)

set(CMAKE_SYSTEM_NAME      Windows)
set(CMAKE_SYSTEM_PROCESSOR x86_64)

set(LLVM_ROOT "C:/Program Files/LLVM")

set(CMAKE_C_COMPILER   "${LLVM_ROOT}/bin/clang-cl.exe")
set(CMAKE_CXX_COMPILER "${LLVM_ROOT}/bin/clang-cl.exe")
set(CMAKE_RC_COMPILER  "${LLVM_ROOT}/bin/llvm-rc.exe")
set(CMAKE_AR     "${LLVM_ROOT}/bin/llvm-ar.exe")
set(CMAKE_LINKER "${LLVM_ROOT}/bin/lld-link.exe")

set(CMAKE_C_COMPILER_TARGET   "x86_64-pc-windows-msvc")
set(CMAKE_CXX_COMPILER_TARGET "x86_64-pc-windows-msvc")

set(MSVC_VERSION 1940)
