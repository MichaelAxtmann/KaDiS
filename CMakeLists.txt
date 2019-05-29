cmake_minimum_required(VERSION 3.9.4)

project(KaDiS)

set(KaDiS_VERSION "0.1.0")

# prohibit in-source builds
if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif()

# set default build type
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

###########################
# cpplint and uncrustify  #
###########################

# test for target to avoid conflicts with parent projects
if (NOT TARGET AnalyzeModifiedSources)
  add_custom_target(AnalyzeModifiedSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-w")
endif()

if (NOT TARGET AnalyzeAllSources)
add_custom_target(AnalyzeAllSources  perl "${PROJECT_SOURCE_DIR}/codestyle/analyze-source.pl" "-aw")
endif()

########################################################
## External libraries
########################################################

# RBC
add_subdirectory(external/RBC)

# MPI
find_package(MPI REQUIRED)

########################################################
## Flags
########################################################

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

## enable warnings

if(NOT MSVC)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -W")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wextra")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
endif()

# enable extra warnings on gcc
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wcast-qual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Winit-self")
  # disabled to avoid warnings in ips4o
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wnoexcept")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Woverloaded-virtual")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wredundant-decls")
  # enable -Wshadow and -Wold-style-cast only for gcc >= 5
  if(CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 5.0)
    # disabled to avoid warnings in ips4o
    # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wshadow")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wold-style-cast")
  endif()
endif()

# enable extra warnings on clang
if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wdeprecated")
endif()

# Include interprocedural optimization for targets which are generated after this code.

if (CMAKE_BUILD_TYPE STREQUAL "Release")

  include(CheckIPOSupported)

  check_ipo_supported(RESULT ipo_supported OUTPUT error)
  if( ipo_supported )
    message(STATUS "IPO / LTO enabled")
    set(CMAKE_INTERPROCEDURAL_OPTIMIZATION TRUE)
  else()
    message(STATUS "IPO / LTO not supported: <${error}>")
  endif()
  
endif()

########################################################
## Executables and libraries
########################################################

# kadis library

file(GLOB KADIS_INCLUDES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  external/ips4o
  external/ips2pa
  include
  )

file(GLOB KADIS_SOURCES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  "src/AmsSort/GroupMsgToPeAssignment.cpp"
  "src/AmsSort/LocalSampleCount.cpp"
  "src/AmsSort/Overpartition/Overpartition.cpp"
  "src/AmsSort/Overpartition/EqualBucketBound.cpp"
  "src/AmsSort/Overpartition/NonEqualBucketBound.cpp"
  "src/Tools/CommonMpi.cpp"
  "src/Tools/Common.cpp"
  "src/Tools/DebugMpi.cpp"
  )

# Build shared library to inherit mpi wrapper from parent targets
add_library(kadis STATIC ${KADIS_SOURCES})
target_link_libraries(kadis PUBLIC rbc tlx MPI::MPI_CXX)

target_include_directories(kadis PUBLIC ${KADIS_INCLUDES})

# kadis examples
add_executable (kadisexample example/sorting.cpp)
target_link_libraries(kadisexample kadis)

###########################
# compile_commands.json   #
###########################

option(KADIS_COMPILE_DATABASE "Create compile commands file" ON)

if (KADIS_COMPILE_DATABASE)

  set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    # copy compile database to source directory
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
      )
    MESSAGE(STATUS "Copy file compile_commands_extended.json from binary dir to source dir.")
  ENDIF()
else()
  set(CMAKE_EXPORT_COMPILE_COMMANDS OFF)
endif()
