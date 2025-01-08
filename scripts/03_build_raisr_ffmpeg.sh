#!/bin/bash

# This scrpit is used to build raisr and ffmpeg

# Usage: 03_build_raisr_ffmpeg.sh /xxx/raisr/Video-Super-Resolution-Library

raisr_path=$1

if [ -z "$raisr_path" ];then
    echo "Usage: 03_build_raisr_ffmpeg.sh /xxx/raisr/Video-Super-Resolution-Library"
    exit 1
fi

# set IPP, x264 and x265 env
export PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:$PKG_CONFIG_PATH"
source /opt/intel/oneapi/ipp/latest/env/vars.sh

# build raisr
cd $raisr_path
sudo -E ./build.sh

# build ffmpeg
cd ../ffmpeg
cp ../Video-Super-Resolution-Library/ffmpeg/vf_raisr.c libavfilter/

./configure \
 --enable-libipp \
 --extra-cflags="-fopenmp -I/opt/intel/oneapi/ipp/latest/include/ipp" \
 --extra-ldflags=-fopenmp \
 --enable-gpl \
 --enable-libx264 \
 --enable-libx265 \
 --extra-libs='-lraisr -lstdc++ -lippcore -lippvm -lipps -lippi' \
 --enable-cross-compile
make clean
make -j $(nproc)

cp -r ../Video-Super-Resolution-Library/filters* .

echo -e 'To run the library you can do the following: \n
      cd raisr/ffmpeg \n
      ./ffmpeg -y -i /input_files/input.mp4 -vf raisr=threadcount=20 -pix_fmt yuv420p /output_files/out.yuv \n
And you can see more use cases in README file of Video-Super-Resolution-Library.\n\n
Notice: If you get "ffmpeg: error while loading shared libraries", try first doing:
export LD_LIBRARY_PATH="/opt/intel/oneapi/ipp/latest/lib/intel64:/usr/local/lib:${LD_LIBRARY_PATH}"'

