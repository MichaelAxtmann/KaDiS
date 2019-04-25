cmake_minimum_required(VERSION 3.4)

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

########################################################
## Flags
########################################################

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14")

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

########################################################
## Executables and libraries
########################################################

# rbcsorting library

add_library(rbcsorting INTERFACE)
target_link_libraries(rbcsorting INTERFACE rbc tlx)

file(GLOB RBCSORTING_INCLUDES RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
  external/ips4o
  include/JanusSort
  include/RQuick
  )

target_include_directories(rbcsorting INTERFACE ${RBCSORTING_INCLUDES})

# rbcsorting example
add_executable (sortingexample example/sorting_example.cpp)
target_link_libraries(sortingexample rbcsorting atomic)


###########################
# youcompleteme and rtags #
###########################

option(COMPILE_DATABASE "Creates a compile database for youcompleteme and adds the database to rtags" OFF)

if (COMPILE_DATABASE)
  SET(CMAKE_EXPORT_COMPILE_COMMANDS ON)

  # add header files to existing compile database
  list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")
  include(add_header_commands_module)
  add_header_commands("${CMAKE_CURRENT_SOURCE_DIR}/cmake" ${CMAKE_CURRENT_BINARY_DIR})

  # compile database created -- continue processing
  IF( EXISTS "${CMAKE_CURRENT_BINARY_DIR}/compile_commands.json" )
    # copy compile database to source directory
    EXECUTE_PROCESS( COMMAND ${CMAKE_COMMAND} -E copy_if_different
      ${CMAKE_CURRENT_BINARY_DIR}/compile_commands_extended.json
      ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
      )
    MESSAGE(STATUS "Copy file compile_commands_extended.json from binary dir to source dir.")

    # load compile database into rtags
    include(add_database_to_rtags_module)
    add_database_to_rtags(${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json)
  ENDIF()
endif()
