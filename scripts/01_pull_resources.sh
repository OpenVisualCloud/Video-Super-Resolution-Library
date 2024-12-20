#!/bin/bash

# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2024-2025 Intel Corporation

# This script is used to pull all resources(Cmake 3.14, nasm, x264, x265, IPP, Raisr and ffmpeg) used for build RAISR and ffmpeg. And by default, the script will generate a tarball raisr.tar.gz of these resources and you can use "no_package" option to disable generate the tarball.
set -eo pipefail

SCRIPT_DIR="$(readlink -f "$(dirname -- "${BASH_SOURCE[0]}")")"
REPOSITORY_DIR="$(readlink -f "${SCRIPT_DIR}/../")"
. "${SCRIPT_DIR}/common.sh"

prompt Starting script execution "${BASH_SOURCE[0]}"
raisr_folder="${raisr_folder:-raisr}"

mkdir -p "${raisr_folder}" "/tmp/Video-Super-Resolution-Library"
cp -r ${REPOSITORY_DIR}/* /tmp/Video-Super-Resolution-Library
pushd "${raisr_folder}"

package_flag=true
ipp_offline_uri='https://registrationcenter-download.intel.com/akdlm/IRC_NAS/7e07b203-af56-4b52-b69d-97680826a8df/l_ipp_oneapi_p_2021.12.1.16_offline.sh'

# Usage: 01_pull_resource <no_package>
while [ -n "$*" ]; do
    case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
        no_package) package_flag=false && shift ;;
        *) break ;;
    esac
done

# pull raisr code
# git clone https://github.com/OpenVisualCloud/Video-Super-Resolution-Library.git
mv /tmp/Video-Super-Resolution-Library .
if [ ! -d "Video-Super-Resolution-Library" ];then
    error "Failed to pull source code of Video-Super-Resolution-Library!"
    exit 1
fi

# pull cmake 3.14
wget --tries=5 --progress=dot:giga https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz
if [ ! -f "cmake-3.14.0.tar.gz" ];then
    error "Failed to download Cmake 3.14!"
    exit 1
fi


# pull ffmpeg
git clone https://github.com/FFmpeg/FFmpeg ffmpeg
if [ ! -d "ffmpeg" ];then
    error "Failed to pull source code of ffmpeg!"
    exit 1
fi

pushd ffmpeg
git checkout -b n6.0 n6.0
git am ../Video-Super-Resolution-Library/ffmpeg/0001-ffmpeg-raisr-filter.patch
popd

# pull nasm used for build x264
wget --tries=5 --progress=dot:giga https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2
if [ ! -f "nasm-2.15.05.tar.bz2" ];then
    error "Failed to download nasm!"
    exit 1
fi

# pull x264
git clone https://github.com/mirror/x264 -b stable --depth 1
if [ ! -d "x264" ];then
    error "Failed to pull source code of x264!"
    exit 1
fi

# pull x265
wget --tries=5 --progress=dot:giga https://github.com/videolan/x265/archive/3.4.tar.gz
if [ ! -f "3.4.tar.gz" ];then
    error "Failed to download source code of x265!"
    exit 1
fi

# pull IPP
wget --tries=5 --progress=dot:giga "${ipp_offline_uri}"
if [ ! -f "${ipp_offline_uri##*/}" ];then
    error "Failed to download IPP package!"
    exit 1
fi

prompt "Successfully downloaded all these resources!"

if [ "$package_flag" == "true" ]; then
    cd ..
    tar -zcvf ./$raisr_folder.tar.gz ./$raisr_folder
    if [ ! -f "$raisr_folder.tar.gz" ];then
        error "Failed to package these resources to $raisr_folder.tar.gz!"
        exit 1
    else
        echo "Successfully packaged these resources to $raisr_folder.tar.gz!"
        rm -rf ./$raisr_folder
    fi
fi

popd
prompt Finished script execution "${BASH_SOURCE[0]}"
