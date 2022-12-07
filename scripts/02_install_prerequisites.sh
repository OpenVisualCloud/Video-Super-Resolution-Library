#!/bin/bash

# This scrpit is used to extract the tarball raisr.tar.gz of resources and build and install the libraries required by building raisr and ffmpeg
# It requires Linux based OS(Tested and validated on Ubuntu 18.04 LTS), gcc/g++ 7.5 or later, make and pkg-config to run this script.

# Usage: 02_install_prerequisites.sh /xxx/raisr.tar.gz

package_path=$1
if [ -z "$package_path" ];then
    echo "Usage: 02_install_prerequisites.sh /xxx/raisr.tar.gz"
    exit 1
fi

tar -zxf $package_path ./
cd raisr

# install IPP
chmod +x ./l_ipp_oneapi_p_2021.6.2.16995_offline.sh
sudo ./l_ipp_oneapi_p_2021.6.2.16995_offline.sh -a -s --eula accept
echo "source /opt/intel/oneapi/ipp/latest/env/vars.sh" | tee -a ~/.bash_profile

# build and install CMake 3.14
tar zxf ./cmake-3.14.0.tar.gz
cd cmake-3.14.0 && \
    ./bootstrap --prefix=/usr/local && \
    make -j $(nproc) && \
    sudo make install
cd -

# build and install nasm
tar xjf ./nasm-2.15.05.tar.bz2 && \
    cd nasm-2.15.05 && \
    ./autogen.sh && \
    ./configure --prefix=/usr/local --libdir=/usr/local/lib && \
    make -j $(nproc) && \
    sudo make install
cd -

# build and install x264
cd x264 && \
    ./configure --prefix=/usr/local --libdir=/usr/local/lib \
    --enable-shared && \
    make -j $(nproc) && \
    sudo make install
cd -

# build and install x265
tar xzf ./3.4.tar.gz
cd x265-3.4/build/linux && \
    cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DLIB_INSTALL_DIR=/usr/local/lib ../../source && \
    make -j $(nproc) && \
    sudo make install
cd -

# remove the resources except Raisr and ffmpeg
rm l_ipp_oneapi_p_2021.6.2.16995_offline.sh
rm 3.4.tar.gz
rm cmake-3.14.0.tar.gz
rm nasm-2.15.05.tar.bz2
rm -rf ./x264
rm -rf ./x265-3.4
rm -rf ./cmake-3.14.0
rm -rf ./nasm-2.15.05
