#!/bin/bash

# This script is used to pull all resources(Cmake 3.14, nasm, x264, x265, IPP, Raisr and ffmpeg) used for build RAISR and ffmpeg. And by default, the script will generate a tarball raisr.tar.gz of these resources and you can use "no_package" option to disable generate the tarball.

package_flag=true

# Usage: 01_pull_resource <no_package>
while [ -n "$*" ]; do
    case "$(printf %s "$1" | tr '[:upper:]' '[:lower:]')" in
        no_package) package_flag=false && shift ;;
        *) break ;;
    esac
done

raisr_folder=raisr
mkdir $raisr_folder
cd $raisr_folder

# pull cmake 3.14
wget  https://cmake.org/files/v3.14/cmake-3.14.0.tar.gz
if [ ! -f "cmake-3.14.0.tar.gz" ];then
    echo "Failed to download Cmake 3.14!"
    exit 1
fi

# pull raisr code
git clone https://github.com/OpenVisualCloud/Video-Super-Resolution-Library.git
if [ ! -d "Video-Super-Resolution-Library" ];then
    echo "Failed to pull source code of Video-Super-Resolution-Library!"
    exit 1
fi

# pull ffmpeg
git clone https://github.com/FFmpeg/FFmpeg ffmpeg
if [ ! -d "ffmpeg" ];then
    echo "Failed to pull source code of ffmpeg!"
    exit 1
fi

cd ffmpeg
git checkout -b n4.4 n4.4
git am ../Video-Super-Resolution-Library/ffmpeg/0001-ffmpeg-raisr-filter.patch
cd -

# pull nasm used for build x264
wget https://www.nasm.us/pub/nasm/releasebuilds/2.15.05/nasm-2.15.05.tar.bz2
if [ ! -f "nasm-2.15.05.tar.bz2" ];then
    echo "Failed to download nasm!"
    exit 1
fi

# pull x264 and x265
git clone https://github.com/mirror/x264 -b stable --depth 1
if [ ! -d "x264" ];then
    echo "Failed to pull source code of x264!"
    exit 1
fi

wget  https://github.com/videolan/x265/archive/3.4.tar.gz
if [ ! -f "3.4.tar.gz" ];then
    echo "Failed to download source code of x265!"
    exit 1
fi

# pull IPP
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19007/l_ipp_oneapi_p_2021.6.2.16995_offline.sh
if [ ! -f "l_ipp_oneapi_p_2021.6.2.16995_offline.sh" ];then
    echo "Failed to download IPP package!"
    exit 1
fi

echo "Successfully downloaded all these resources!"

if [ "$package_flag" == "true" ]; then
    cd ..
    tar -zcvf ./$raisr_folder.tar.gz ./$raisr_folder
    if [ ! -f "$raisr_folder.tar.gz" ];then
        echo "Failed to package these resources to $raisr_folder.tar.gz!"
        exit 1
    else
        echo "Successfully packaged these resources to $raisr_folder.tar.gz!"
        rm -rf ./$raisr_folder
    fi
fi
