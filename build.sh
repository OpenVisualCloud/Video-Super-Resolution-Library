#!/bin/bash

# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2024-2025 Intel Corporation

# Fails the script if any of the commands error (Other than if and some others)
set -e -o pipefail

SCRIPT_DIR="$(readlink -f "$(dirname -- "${BASH_SOURCE[0]}")")"
REPOSITORY_DIR="$(readlink -f "${SCRIPT_DIR}")"

. "${SCRIPT_DIR}/scripts/common.sh"
nproc="${nproc:-$(nproc)}"

# Env variable BUILD_TYPE can be one off: RelWithDebInfo, Release, Debug
BUILD_TYPE="${BUILD_TYPE:-Release}"

CMAKE_C_FLAGS=" -I/opt/intel/oneapi/ipp/latest/include -I/opt/intel/oneapi/ipp/latest/include/ipp ${CMAKE_C_FLAGS}"
CMAKE_CXX_FLAGS=" -I/opt/intel/oneapi/ipp/latest/include -I/opt/intel/oneapi/ipp/latest/include/ipp ${CMAKE_CXX_FLAGS}"
CMAKE_LIBRARY_PATH="/opt/intel/oneapi/ipp/latest/lib;${PREFIX}/lib;${CMAKE_LIBRARY_PATH}"
LDFLAGS="${LDFLAGS} -L/opt/intel/oneapi/ipp/latest/lib -L${PREFIX}/lib "

# Helpful when copying and pasting functions and debuging.
if printf '%s' "$0" | grep -q '\.sh'; then
    IN_SCRIPT=true
fi

function cd_safe() {
    if (cd "$1"); then
        cd "$1"
    else
        _dir="$1"
        shift
        die "${@:-Failed cd to $_dir.}"
    fi
}

# Usage: build [test]
function build()
(
    log_info "Create folder: build, build type: ${BUILD_TYPE}"

    if [[ -d "${REPOSITORY_DIR:?}/build" ]]; then
      rm -rf "${REPOSITORY_DIR:?}/build/"*
    fi

    mkdir -p "${REPOSITORY_DIR}/build" > /dev/null 2>&1

    cmake -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" "${CMAKE_EXTRA_FLAGS}" -B "${REPOSITORY_DIR}/build" -S "${REPOSITORY_DIR}" "$@"
    #cmake .. -DCMAKE_BUILD_TYPE="RelWithDebInfo" $CMAKE_EXTRA_FLAGS "$@"

    if [ -f "${REPOSITORY_DIR}/build/Makefile" ]; then
      make -j"${nproc}" -C "${REPOSITORY_DIR}/build"
      as_root make install -j"${nproc}" -C "${REPOSITORY_DIR}/build"
    fi
)

function check_executable()
{
    print_exec=(printf '\0')
    if [[ "$#" -ge "2" ]]; then
      if [[ "${1}" == "-p" ]]; then
        print_exec=(printf '%s\n')
      fi
      shift
    fi

    if [[ "$#" -ge "1" ]]; then
      command_to_check="${1}" && shift
    else
      log_error "Wrong number of parameters passed to check_executable()."
      return 1
    fi

    if [ -e "${command_to_check}" ]; then
      "${print_exec[@]}" "${command_to_check}"
      return 0
    fi

    for pt in "$@" $(echo "${PATH}" | tr ':' ' '); do
      if [ -e "${pt}/${command_to_check}" ]; then
        "${print_exec[@]}" "${pt}/${command_to_check}"
        return 0
      fi
    done

    return 127
}

if check_executable icpx; then
    CXX=$(check_executable -p icpx)
elif check_executable g++; then
    CXX=$(check_executable -p g++)
elif check_executable clang++; then
    CXX=$(check_executable -p clang++)
else
    log_error "No suitable cpp compiler found in path."
    log_error "Please either install one or set it via cxx=*"
    die "[Exiting due to error.]"
fi

export CXX
build "$@"
