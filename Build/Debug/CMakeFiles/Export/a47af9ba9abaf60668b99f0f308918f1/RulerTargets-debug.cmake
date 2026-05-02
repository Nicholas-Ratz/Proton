#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Ruler::Ruler" for configuration "Debug"
set_property(TARGET Ruler::Ruler APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Ruler::Ruler PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libRuler.a"
  )

list(APPEND _cmake_import_check_targets Ruler::Ruler )
list(APPEND _cmake_import_check_files_for_Ruler::Ruler "${_IMPORT_PREFIX}/lib/libRuler.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
