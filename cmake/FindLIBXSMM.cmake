# Distributed under the MIT License.
# See LICENSE.txt for details.

# Find LIBXSMM: https://github.com/hfp/libxsmm
# If not in one of the default paths specify -D LIBXSMM_ROOT=/path/to/LIBXSMM
# to search there as well.

if(NOT LIBXSMM_ROOT)
  # Need to set to empty to avoid warnings with --warn-uninitialized
  set(LIBXSMM_ROOT "")
  set(LIBXSMM_ROOT $ENV{LIBXSMM_ROOT})
endif()

# find the LIBXSMM include directory
find_path(LIBXSMM_INCLUDE_DIRS libxsmm.h
  PATH_SUFFIXES include
  HINTS ${LIBXSMM_ROOT})

find_library(LIBXSMM_LIBRARIES
  NAMES xsmm
  PATH_SUFFIXES lib64 lib
  HINTS ${LIBXSMM_ROOT})

set(LIBXSMM_VERSION "")

if(EXISTS "${LIBXSMM_INCLUDE_DIRS}/libxsmm.h")
  # Extract version info from header
  file(READ
    "${LIBXSMM_INCLUDE_DIRS}/libxsmm.h"
    LIBXSMM_FIND_HEADER_CONTENTS)

  string(REGEX MATCH "#define LIBXSMM_VERSION_MAJOR [0-9]+"
    LIBXSMM_MAJOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define LIBXSMM_VERSION_MAJOR " ""
    LIBXSMM_MAJOR_VERSION
    "${LIBXSMM_MAJOR_VERSION}")

  string(REGEX MATCH "#define LIBXSMM_VERSION_MINOR [0-9]+"
    LIBXSMM_MINOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define LIBXSMM_VERSION_MINOR " ""
    LIBXSMM_MINOR_VERSION
    "${LIBXSMM_MINOR_VERSION}")

  string(REGEX MATCH "#define LIBXSMM_VERSION_UPDATE [0-9]+"
    LIBXSMM_SUBMINOR_VERSION "${LIBXSMM_FIND_HEADER_CONTENTS}")
  string(REPLACE "#define LIBXSMM_VERSION_UPDATE " ""
    LIBXSMM_SUBMINOR_VERSION
    "${LIBXSMM_SUBMINOR_VERSION}")

  set(LIBXSMM_VERSION
    "${LIBXSMM_MAJOR_VERSION}.${LIBXSMM_MINOR_VERSION}.${LIBXSMM_SUBMINOR_VERSION}"
    )
else()
  message(WARNING "Failed to find file "
    "'${LIBXSMM_INCLUDE_DIRS}/libxsmm.h' "
    "while detecting the LIBXSMM version.")
endif(EXISTS "${LIBXSMM_INCLUDE_DIRS}/libxsmm.h")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  LIBXSMM
  FOUND_VAR LIBXSMM_FOUND
  REQUIRED_VARS LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES
  VERSION_VAR LIBXSMM_VERSION)
mark_as_advanced(LIBXSMM_INCLUDE_DIRS LIBXSMM_LIBRARIES
  LIBXSMM_MAJOR_VERSION LIBXSMM_MINOR_VERSION LIBXSMM_SUBMINOR_VERSION
  LIBXSMM_VERSION)
