#!/bin/bash

# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2024-2025 Intel Corporation

set -ex -o pipefail

# Helpful when copying and pasting functions and debuging.
if printf '%s' "$0" | grep -q '\.sh'; then
    IN_SCRIPT=true
fi

# Fails the script if any of the commands error (Other than if and some others)
set -e

cd_safe() {
    if (cd "$1"); then
        cd "$1"
    else
        _dir="$1"
        shift
        die "${@:-Failed cd to $_dir.}"
    fi
}

# Usage: build [test]
build() (
    build_type=Release
    echo "Create folder: build, build type: $build_type"
    mkdir -p build > /dev/null 2>&1
    cd_safe build

    for file in *; do
        rm -rf "$file"
    done

    cmake .. -DCMAKE_BUILD_TYPE="$build_type" $CMAKE_EXTRA_FLAGS "$@"
    #cmake .. -DCMAKE_BUILD_TYPE="RelWithDebInfo" $CMAKE_EXTRA_FLAGS "$@"

    if [ -f Makefile ]; then
        # make -j
        make install -j
    fi

    cd ..
)


check_executable() (
    print_exec=false
    while true; do
        case "$1" in
        -p) print_exec=true && shift ;;
        *) break ;;
        esac
    done
    [ -n "$1" ] && command_to_check="$1" || return 1
    shift
    if [ -e "$command_to_check" ]; then
        $print_exec && printf '%s\n' "$command_to_check"
        return 0
    fi
    for d in "$@" $(printf '%s ' "$PATH" | tr ':' ' '); do
        if [ -e "$d/$command_to_check" ]; then
            $print_exec && printf '%s\n' "$d/$command_to_check"
            return 0
        fi
    done
    return 127
)
if check_executable icpx; then
    CXX=$(check_executable -p icpx)
elif check_executable clang++; then
    CXX=$(check_executable -p clang++)
elif check_executable g++; then
    CXX=$(check_executable -p g++)
else
    die "No suitable cpp compiler found in path" \
        "Please either install one or set it via cxx=*"
fi
export CXX

build $@
