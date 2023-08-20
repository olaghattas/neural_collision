#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "collision_lib::collision_lib" for configuration "Debug"
set_property(TARGET collision_lib::collision_lib APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(collision_lib::collision_lib PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/libcollision_lib.a"
  )

list(APPEND _cmake_import_check_targets collision_lib::collision_lib )
list(APPEND _cmake_import_check_files_for_collision_lib::collision_lib "${_IMPORT_PREFIX}/lib/libcollision_lib.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
