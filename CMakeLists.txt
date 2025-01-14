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

set(RAISR_VERSION_MAJOR           "23")
set(RAISR_VERSION_MINOR           "11")
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

# Raisr can use SVML for atan2 if Intel Compiler can be found, otherwise uses an atanApproximation
get_filename_component(CC_FILENAME ${CMAKE_CXX_COMPILER} NAME)
string(FIND "${CC_FILENAME}" "icp" CC_SUBSTR_INTEL)
if (${CC_SUBSTR_INTEL} LESS 0)
    message("Building with Atan Approximation")
    list(APPEND flags_to_test "-DUSE_ATAN2_APPROX")
else()
    message("Building with Intel Compiler, using SVML")
endif()

INCLUDE(CheckCXXSourceRuns)
SET(CMAKE_REQUIRED_FLAGS "-march=native")
check_cxx_source_runs("
    #include <immintrin.h>
    int main(int argc, char** argv) {
        _Float16 data[32] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31};
        _Float16 output[32];
        __m512h a = _mm512_loadu_ph(data); // avx512fp16
        __m512h b = _mm512_loadu_ph(data);
        __mmask32 mask = _mm512_cmp_ph_mask(a, b, _CMP_EQ_OQ); // avx512fp16
        __m512h c = _mm512_add_ph(a, b); // avx512fp16
        _mm512_storeu_ph(output, c); // avx512fp16
        return 0;
    }" HAVE_AVX512FP16)
check_cxx_source_runs("
    #include <immintrin.h>
    int main(int argc, char** argv) {
        float data[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
        float output[16];
        __m512 a = _mm512_loadu_ps(data); // avx512f
        __m512 b = _mm512_loadu_ps(data);
        __mmask8 mask = _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ); // avx512vl
        __m512 c = _mm512_add_ps(a, b); // avx512f
        _mm512_storeu_ps(output, c); // avx512f
        return 0;
    }" HAVE_AVX512)

foreach(cflag ${flags_to_test})
    string(REGEX REPLACE "[^A-Za-z0-9]" "_" cflag_var "${cflag}")
    set(test_cxx_flag "CXX_FLAG${cflag_var}")
    check_cxx_compiler_flag(${cflag} "${test_cxx_flag}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${cflag}")
endforeach()

endif(UNIX)

option( ENABLE_RAISR_OPENCL "Build with OpenCL?" OFF )
if (ENABLE_RAISR_OPENCL)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DENABLE_RAISR_OPENCL")
endif()


# Intel Library for Video Super Resolution
add_subdirectory(Library)

# Raisr Sample app
include_directories(${PROJECT_SOURCE_DIR}/Library/)

