cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME "Raisr")
project(${PROJECT_NAME} CXX)

if("${CMAKE_CURRENT_SOURCE_DIR}" STREQUAL "${CMAKE_CURRENT_BINARY_DIR}")
    message(WARNING "Building in-source is highly not recommended\n"
                    "Please use the Build folder or create your own.")
endif()


set(CMAKE_BUILD_TYPE Release)

#COMPILE_AS_CPP
enable_language(CXX)
include(CheckCXXCompilerFlag)

set(RAISR_VERSION_MAJOR           "22")
set(RAISR_VERSION_MINOR           "9")
configure_file(${PROJECT_SOURCE_DIR}/Library/RaisrVersion.h.in ${PROJECT_SOURCE_DIR}/Library/RaisrVersion.h @ONLY)

if( UNIX )

set(flags_to_test
    -march=native
    -O3
    -std=c++17
    -DNDEBUG
    -ffast-math
    -Wno-narrowing
    -fPIC
)

foreach(cflag ${flags_to_test})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" cflag_var "${cflag}")
    set(test_cxx_flag "CXX_FLAG${cflag_var}")
    check_cxx_compiler_flag(${cflag} "${test_cxx_flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cflag}")
endforeach()

endif(UNIX)


# Intel Library for Video Super Resolution
add_subdirectory(Library)

# Raisr Sample app
include_directories(${PROJECT_SOURCE_DIR}/Library/)

