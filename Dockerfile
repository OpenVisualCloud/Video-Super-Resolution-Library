# INTEL CONFIDENTIAL
#
# Copyright (C) 2022 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and your use of them is governed by the express license under which they were provided to you ("License").
# Unless the License provides otherwise, you may not use, modify, copy, publish, distribute, disclose or transmit this software or the related documents without Intel's prior written
# permission.
#
# This software and the related documents are provided as is, with no express or implied warranties, other than those that are expressly stated in the License.

# use Ubuntu 18.04 with Intel IPP 
FROM intel/oneapi-basekit:devel-ubuntu18.04

#
# Use bash shell
RUN rm /bin/sh && ln -s /bin/bash /bin/sh
#
# Update apt and install dependances
RUN apt-get update && apt-get install  -y \
build-essential \
git \
libx265-dev \
libx264-dev \
nasm \
software-properties-common \
zlib1g-dev
#
# Install and configure gcc 9
RUN add-apt-repository -y ppa:ubuntu-toolchain-r/test
RUN apt-get -y install gcc-9 g++-9
RUN update-alternatives \
--install /usr/bin/gcc gcc /usr/bin/gcc-9 90 \
--slave /usr/bin/g++ g++ /usr/bin/g++-9 \
--slave /usr/bin/gcov gcov /usr/bin/gcov-9
#
# Setup raisrfolder as the working folder
WORKDIR /raisrfolder
#
# Copy raisr library and include files to usr/local
COPY ./libraisr.a /usr/local/lib/
COPY Library/*.h /usr/local/include/raisr/
#
# Clone ffmpeg and checkout branch n4.4
RUN git clone https://github.com/FFmpeg/FFmpeg ffmpeg
WORKDIR /raisrfolder/ffmpeg
RUN git checkout -b mybranch n4.4
#
# Apply the raisr filter patch
COPY ./ffmpeg/*.patch .
COPY ./ffmpeg/vf_raisr.c libavfilter/
RUN git config --global user.email "you@example.com"
RUN git apply 0001-ffmpeg-raisr-filter.patch
#
# Source the IPP library
#
ENV CPATH "/opt/intel/oneapi/ipp/latest/include"
ENV IPPROOT "/opt/intel/oneapi/ipp/latest"
ENV IPP_TARGET_ARCH "intel64"
ENV LD_LIBRARY_PATH "/opt/intel/oneapi/ipp/latest/lib/intel64"
ENV LIBRARY_PATH "/opt/intel/oneapi/ipp/latest/lib/intel64"

# Configure and build ffmpeg
RUN ./configure \
--enable-libipp \
--enable-zlib \
--extra-cflags="-fopenmp" \
--extra-ldflags=-fopenmp \
--enable-gpl \
--enable-libx264 \
--enable-libx265 \
--extra-libs='-lraisr -lstdc++ -lippcore -lippvm -lipps -lippi' \
--enable-cross-compile
RUN make clean
RUN make -j $(nproc)
#
# Copy the raisr filters from the raisr library
COPY ./filters1/* ./filters1/
COPY ./filters2/* ./filters2/
COPY ./filters3/* ./filters3/
COPY ./filters4/* ./filters4/
COPY ./filters5/* ./filters5/
#
# Run ffmpeg and verify that the raisr filter is supported
RUN ./ffmpeg -h filter=raisr
