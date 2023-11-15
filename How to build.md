# Prerequisites
To build this project you will need:
- Linux based OS
    - For CPU tested and validated on Ubuntu 18.04 LTS, Ubuntu 22.04 and CentOS 7.9
    - For GPU tested and validated on Intel® Data Center GPU Flex 170 with Ubuntu 22.04 LTS(5.15 LTS kernel)
- [Docker](https://www.docker.com/) 
- Intel Xeon hardware which supports Intel AVX512 (Skylake generation or later)
- Compiler (clang++, g++, icc), and enabling AVX512-FP16 on Sapphire Rapiads needs:
    - clang++ version 14.0.0 or later
    - g++ version 12.1.0 with binutils 2.38 or later
    - icc version 2021.2 or later
- Cmake version 3.14 or later 
- Intel® Integrated Performance Primitives (Intel® IPP) (Stand-Alone Version is the minimum requirement)
- zlib1g-dev, pkg-config (The pkg-config is used to find x264.pc/x265.pc in specific pkgconfig path.)

We provide 3 ways to build the Intel VSR with FFmpeg environment:
- build docker images with dockerfiles(only for CPU).
- build via [scripts](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/tree/master/scripts)(only for CPU).
- build manually(both CPU and GPU).

# Build Docker Images.

We provide 3 Dockerfile for Intel Xeon platforms: Ubuntu18.04, Ubuntu22.04 and CentOS7.9. You can refer to below steps to build docker images.
## Setup docker proxy as follows if you are behind a firewall:
```
sudo mkdir -p /etc/systemd/system/docker.service.d
printf "[Service]\nEnvironment=\"HTTPS_PROXY=$https_proxy\" \"NO_PROXY=$no_proxy\"\n" | sudo tee /etc/systemd/system/docker.service.d/proxy.conf
sudo systemctl daemon-reload
sudo systemctl restart docker
```

## build docker image via docker_build.sh
The usage of the script is as follows
```
./docker_build.sh <OS> <OS_VERSION>

./docker_build.sh ubuntu 22.04 #for building Ubuntu22.04
./docker_build.sh ubuntu 18.04 #for building Ubuntu18.04
./docker_build.sh centos 7.9 #for building CentOS7.9
```
If the image is built successfully, you can find a docker image named `raisr:ubuntu22.04` or `raisr:ubuntu18.04` or `raisr:centos7.9` with command `docker images`

# Build via [scripts](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/tree/master/scripts)
If the user would prefer not to use Docker, you can follow the steps below to setup enviroment: \
    `cd Video-Super-Resolution-Library/scripts` \
    `./01_pull_resources.sh` \
    `./02_install_prerequisites.sh /xxx/raisr.tar.gz` \
    `./03_build_raisr_ffmpeg.sh /xxx/raisr/Video-Super-Resolution-Library`
- [01_pull_resources.sh](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/master/scripts/01_pull_resources.sh): Download the resources used for build Intel Library for VSR and FFmpeg(cmake 3.14, nasm, x264, x265, ipp, Intel Library for VSR and FFmpeg) and package these resource to      raisr.tar.gz.
- [02_install_prerequisites.sh](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/master/scripts/02_install_prerequisites.sh): Extract the tarball raisr.tar.gz of resources and build and install the libraries required by building Intel Library for VSR and FFmpeg.
- [03_build_raisr_ffmpeg.sh](https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/blob/master/scripts/03_build_raisr_ffmpeg.sh): Build Intel Library for VSR and FFmpeg.

# Build manually following the steps below
## Install Intel IPP
Standalone version of IPP (minimum requirement): https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#ipp \
Alternatively, install IPP as part of oneAPI Base Toolkit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html \
Add below line to `~/.bash_profile` which sets IPP env. \
    `source /opt/intel/oneapi/ipp/latest/env/vars.sh` \
Then `source ~/.bash_profile`

## Install dependent libraries x264 and x265
The x264/x265 libraries can be installed via apt on Ubuntu OS or built and installed from source code.

### Install x264/x265 via apt on Ubuntu OS(Option-1)
`apt-get update && apt-get install -y libx264-dev libx265-dev nasm`

### Build and install x264/x265 from source code(Option-2)

#### Build and install x264 

`git clone https://github.com/mirror/x264 -b stable --depth 1` \
`cd x264` \
`./configure --prefix=/usr/local --libdir=/usr/local/lib --enable-shared` \
`make -j$(nproc)` \
`sudo make install`

#### Build and install x265

`wget -O - https://github.com/videolan/x265/archive/3.4.tar.gz | tar xz` \
`cd x265-3.4/build/linux` \
`cmake -DBUILD_SHARED_LIBS=ON -DCMAKE_INSTALL_PREFIX=/usr/local -DLIB_INSTALL_DIR=/usr/local/lib -DHIGH_BIT_DEPTH=ON ../../source` \
`make -j$(nproc)` \
`sudo make install`

#### Set PKG_CONFIG_PATH enviroment variable
The `.pc` files of x264 and x265 libraries are in `/usr/local/lib/pkgconfig`, add the path to the `PKG_CONFIG_PATH` environment variable. \
`export PKG_CONFIG_PATH=/usr/local/lib/pkgconfig/:$PKG_CONFIG_PATH`

## Install Opencl
Follow this guide to setup gpu driver [installation](https://dgpu-docs.intel.com/driver/installation.html) \
Install OpenCL: \
`sudo apt update` \
`sudo apt install intel-opencl-icd opencl-headers ocl-icd-opencl-dev` \

### Install Intel latest OpenCL driver
[intel/compute-runtime](https://github.com/intel/compute-runtime/releases)

## Use FFmpeg RAISR plugin

### Build and install the Intel Library for VSR Manually

If the user would prefer not to use Docker, these instructions may be utilized to setup the Intel Library for VSR.

To build the library without building the docker image, run \
`./build.sh -DCMAKE_INSTALL_PREFIX="$PWD/install"`

To build the library with OpenCL support, run \
`.  -DCMAKE_INSTALL_PREFIX="$PWD/install" -DENABLE_RAISR_OPENCL=ON`

#### Clone FFmpeg
`git clone https://github.com/FFmpeg/FFmpeg ffmpeg` \
`cd ffmpeg`

#### Checkout FFmpeg version 6.0 tag
`git checkout -b n6.0 n6.0`

#### Copy vf_raisr.c to ffmpeg libavfilter folder
`cp ../Video-Super-Resolution-Library/ffmpeg/vf_raisr.c libavfilter/` \
To use raisr_opencl you need to copy vf_raisr_opencl.c as well \
`cp ../Video-Super-Resolution-Library/ffmpeg/vf_raisr_opencl.c libavfilter/`

#### Apply patch
`git am ../Video-Super-Resolution-Library/ffmpeg/0001-ffmpeg-raisr-filter.patch` \
To use raisr_opencl you need to apply patch 0002 as well \
`git am ../Video-Super-Resolution-Library/ffmpeg/0002-libavfilter-raisr_opencl-Add-raisr_opencl-filter.patch`

#### Configure FFmpeg
When `DCMAKE_INSTALL_PREFIX` isn't used, the ffmpeg configure command is as: \
`./configure --enable-libipp --extra-cflags="-fopenmp" --extra-ldflags=-fopenmp --enable-gpl --enable-libx264 --enable-libx265 --extra-libs='-lraisr -lstdc++ -lippcore -lippvm -lipps -lippi' --enable-cross-compile`

When `DCMAKE_INSTALL_PREFIX` is used, please add the below line to the ffmpeg configure command: \
`--extra-cflags=="-fopenmp -I../Video-Super-Resolution-Library/install/include/" --extra-ldflags="-fopenmp -L../Video-Super-Resolution-Library/install/lib/"` \
The ffmmpeg confiure command is as: \
`./configure --enable-libipp --extra-cflags="-fopenmp -I../Video-Super-Resolution-Library/install/include/" --extra-ldflags="-fopenmp -L../Video-Super-Resolution-Library/install/lib/" --enable-gpl --enable-libx264 --enable-libx265 --extra-libs='-lraisr -lstdc++ -lippcore -lippvm -lipps -lippi' --enable-cross-compile`

Add option `--enable-opencl` and `-lm` to `--extra-libs` option to enable `raisr_opencl` filter.

#### Build FFmpeg
`make clean` \
`make -j $(nproc)`

#### Copy RAISR filter folder to FFmpeg folder
The folder contains filterbin_2_8/10, Qfactor_cohbin_2_8/10, and Qfactor_strbin_2_8/10 \
`cp -r ../Video-Super-Resolution-Library/filters* .`
