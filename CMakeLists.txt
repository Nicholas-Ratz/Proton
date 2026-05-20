cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD          98)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(TO_CMAKE_PATH "$ENV{HOME}" _HOME_PATH)
file(TO_CMAKE_PATH "$ENV{USERPROFILE}" _USERPROFILE_PATH)
list(APPEND CMAKE_MODULE_PATH
    "${_HOME_PATH}/.atom/CMake"
    "${_USERPROFILE_PATH}/.atom/CMake"
)
include(AtomFunctions REQUIRED)
Atom_RegisterNativeBuildType()

find_package(Quark REQUIRED)

project(Proton LANGUAGES CXX VERSION 1.0.0)
add_library(Proton STATIC
    PRIVATE
    Source/test.cpp
)
target_link_libraries(Proton PUBLIC Quark::Quark)
Atom_DetectCompiler()
Atom_SetupProject(Proton TYPE PLUGIN LIBRARY_TYPE STATIC)
Atom_SetupWarnings(Proton)
Atom_SetupOptimization(Proton LTO)
Atom_SetupInstall(Proton TYPE PLUGIN)
Atom_SetupCompileCommands(SYMLINK)

Atom_SetupExampleProject(Proton)
Atom_SetupProject(Example TYPE PROJECT)
Atom_SetupWarnings(Example)
Atom_SetupOptimization(Example)
