cmake_minimum_required(VERSION 3.21)
set(CMAKE_CXX_STANDARD          17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(TO_CMAKE_PATH "$ENV{HOME}" _HOME_PATH)
file(TO_CMAKE_PATH "$ENV{USERPROFILE}" _USERPROFILE_PATH)
list(APPEND CMAKE_MODULE_PATH
    "${_HOME_PATH}/.axiom/CMake"
    "${_USERPROFILE_PATH}/.axiom/CMake"
)
include(AxiomFunctions REQUIRED)
Axiom_UseToolchain("${CMAKE_SOURCE_DIR}/clang-gnu_Toolchain.cmake")
Axiom_RegisterNativeBuildType()

find_package(Level REQUIRED)

project(Ruler LANGUAGES CXX VERSION 1.0.0)
add_library(Ruler STATIC
    Source/Ruler.cpp
)
target_link_libraries(Ruler Level::Level)
Axiom_DetectCompiler()
Axiom_SetupProject(Ruler TYPE PLUGIN LIBRARY_TYPE STATIC)
Axiom_SetupWarnings(Ruler)
Axiom_SetupOptimization(Ruler LTO)
Axiom_SetupCompileCommands(SYMLINK)
Axiom_SetupInstall(Ruler TYPE PLUGIN)

Axiom_SetupExampleProject(Ruler)
Axiom_SetupProject(Example TYPE PROJECT)
Axiom_SetupWarnings(Example)
Axiom_SetupOptimization(Example)
