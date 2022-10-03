# Intel® Library for Video Super Resolution (Intel® Library for VSR) README
This project aims to provide a CPU based implementation of the RAISR (Rapid and Accurate Image Super Resolution) algorithm (https://arxiv.org/pdf/1606.01299.pdf) optimized to achieve beyond real-time performance for 2x upscaling on Intel® Xeon® platforms.  It can take a lower resolution image and upscale 2x (e.g 540p to 1080p) and provides better quality results than standard (bicubic) algorithms and a good performance vs quality trade-off as compared to DL-based algorithms like EDSR.  The Intel Library for VSR is provided as an FFmpeg plugin inside of a Docker container to help ease testing and deployment burdens.  This project is developed using C++ and takes advantage of Intel® Advanced Vector Extension 512 (Intel® AVX-512).

## Prerequisites
To build this project you will need:
- Linux based OS (Tested and validated on Ubuntu 18.04 LTS)
- [Docker](https://www.docker.com/) 
- Intel Xeon hardware which supports Intel AVX512 (Skylake generation or later)
- Compiler (clang++, g++) 
- Cmake version 3.12 or later 
- Intel® Integrated Performance Primitives (Intel® IPP) (Stand-Alone Version is the minimum requirement)
- (optional) libx264, libx265, zlib1g-dev

## Install Intel IPP
Standalone version of IPP (minimum requirement): https://www.intel.com/content/www/us/en/developer/articles/tool/oneapi-standalone-components.html#ipp \
Alternatively, install IPP as part of oneAPI Base Toolkit: https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit-download.html \
Add below lines to `~/.bash_profile` which sets IPP env. \
    `source /opt/intel/oneapi/ipp/latest/env/vars.sh` \
    `. ~/.bashrc` \
Then `source ~/.bash_profile`

# Use FFmpeg RAISR plugin

## Build and install the Intel® Library for VSR with Docker
In the top level directory of the repository, run
``` 
./build.sh
```
The library and docker image will automatically be built within 'build.sh'.

If the user is not root and does not have sudo permission, it is assumed that the user has been added to the Docker cgroup and the user must provide the path to the install directory when building. 
```
./build.sh -DCMAKE_INSTALL_PREFIX="/path/to/install"
```

This simple command will run the image in a docker container:
```
docker run -ti --rm raisr /bin/bash
```
Alternatively, it could be useful to create an input folder containing videos or images for testing the Intel Library for VSR.  That (and other) folders can then be binded as a volume between your host system and the docker container:
```
docker run -ti --rm -v $PWD/input_files:/input_files -v $PWD/output_files:/output_files raisr /bin/bash
```

## Build and install the Intel Library for VSR Manually

If the user would prefer not to use Docker, these instructions may be utilized to setup the Intel Library for VSR.

To build the library without building the docker image, run
`./build.sh no_docker -DCMAKE_INSTALL_PREFIX="/path/to/install"`

### Clone FFmpeg
`git clone https://github.com/FFmpeg/FFmpeg ffmpeg` \
`cd ffmpeg`

### Checkout n4.4 tag
`git checkout -b [your branch name] n4.4`

### Copy vf_raisr.c to ffmpeg libavfilter folder
`cp ../ilvsr-library/ffmpeg/vf_raisr.c libavfilter/`

### Apply patch
`git am ../ilvsr-library/ffmpeg/0001-ffmpeg-raisr-filter.patch`

### Configure FFmpeg
`./configure --enable-libipp --extra-cflags="-fopenmp" --extra-ldflags=-fopenmp --extra-libs='-lraisr -lstdc++ -lippcore -lippvm -lipps -lippi' --enable-cross-compile`

### Build FFmpeg
`make clean` \
`make -j $(nproc)`

### Copy RAISR filter folder to FFmpeg folder
The folder contains filterbin_partialCV2, Qfactor_cohbin_partialCV2, and Qfactor_strbin_partialCV2 \
`cp -r ../ilvsr-library/filters* .`

## Running the Intel Library for VSR 
One should be able to test with video files:
```
./ffmpeg -y -i /input_files/input.mp4 -vf raisr=threadcount=20 -pix_fmt yuv420p /output_files/out.yuv
```
Or folders of images:
```
./ffmpeg -y -start_number 000 -i '/input_files/img_%03d.png' -vf raisr=threadcount=20 -start_number 000 '/output_files/img_%03d.png'
```
Because saving raw uncompressed (.yuv) video can take up a lot of disk space, one could consider using the lossless (-crf 0) setting in x264/x265 to reduce the output file size by a substantial amount.

**x264 lossless encoding**
```
./ffmpeg -y -i /input_files/input.mp4 -vf raisr=threadcount=20 -pix_fmt yuv420p -c:v libx264 -crf 0 /output_files/out.mp4
```
**x265 lossless encoding**
```
./ffmpeg -y -i /input_files/input.mp4 -vf raisr=threadcount=20 -pix_fmt yuv420p -c:v libx265 -crf 0 /output_files/out_hevc.mp4
```
## Evaluating the Quality of RAISR Super Resolution
Evaluating the quality of the RAISR can be done in different ways.
1. A source video or image can be upscaled by 2x using different filter configurations. We suggest trying these 3 command lines based upon preference:

**Sharpest output**
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:passes=2:filterfolder=filters5" -pix_fmt yuv420p /output_files/out.yuv
```
**Fastest Performance ( second pass disabled )**
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:filterfolder=filters1" -pix_fmt yuv420p /output_files/out.yuv
```
**Medium sharpness**
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:passes=2:filterfolder=filters3" -pix_fmt yuv420p /output_files/out.yuv
```
2. A source video or image can be downscaled by 2x, then passed through the RAISR filter which upscales by 2x
```
./ffmpeg -y -i /input_files/input.mp4 -vf scale=iw/2:ih/2,raisr=threadcount=20 -pix_fmt yuv420p /output_files/out.yuv
```
At this point the source content is the same resolution as the output and the two can be compared to understand how well the super resolution is working.  RAISR can be compared against existing DL super resolution algorithms as well.  It is recommended to enable second pass in Intel Library for VSR to produce sharper images.  Please see the Advanced Usage section for guidance on enabling second pass as a feature.


## To see help on the RAISR filter
`./ffmpeg -h filter=raisr`

    raisr AVOptions:
      ratio             <int>        ..FV....... ratio (from 2 to 4) (default 2)
      bits              <int>        ..FV....... bit depth (from 8 to 10) (default 8)
      range             <string>     ..FV....... color range of the input. If you are working with images, you 
                                                 may want to set range to full (video/full) (default video)
      threadcount       <int>        ..FV....... thread count (from 1 to 120) (default 1)
      filterfolder      <string>     ..FV....... absolute filter folder path (default "filters1")
      blending          <int>        ..FV....... CT blending mode (1: Randomness, 2: CountOfBitsChanged) (from 1 to 2) (default 2)
      passes            <int>        ..FV....... passes to run (1: one pass, 2: two pass) (from 1 to 2) (default 1)
      mode              <int>        ..FV....... mode for two pass (1: upscale in 1st pass, 2: upscale in 2nd
                                                 pass) (from 1 to 2) (default 1)

## Advanced Usage ( through Exposed Parameters )
The FFmpeg plugin for Intel Library for VSR exposes a number of parameters that can be changed for advanced customization
### threadcount
Allowable values (1,120), default (20)

Changes the number of software threads used in the algorithm.  Values 1..120 will operate on segments of an image such that efficient threading can greatly increase the performance of the upscale.  The value itself is the number of threads allocated.
### filterfolder
Allowable values: (Any folder path containing the 4 required filter files: Qfactor_cohbin_partialCV2, Qfactor_strbin_partialCV2, filterbin_partialCV2, config), default (“filters1”)

Changing the way RAISR is trained (using different parameters and datasets) can alter the way RAISR's ML-based algorithms do upscale.  For the current release, 5 filters have been provided: filters1, filters2, filters3, filters4 and filters5.  Each filter has been generated with varying amounts of complexity and sharpness.

If doing a single pass ( passes=1 ), it is suggested to use filters1 or filters4.  With passes=2 the 5 filters are described as:
| filter name | filter complexity | sharpness |
| ----------- | ----------------- | ----------- |
| filters1    | 864x2             | high |
| filters2    | 864x2             | low |
| filters3    | 864x2             | medium |
| filters4    | 32000x2           | medium |
| filters5    | 32000x2           | high |

Please see the examples under the "Evaluating the Quality" section above where we suggest 3 command lines based upon preference.
Note that for second pass to work, the filter folder must contain 3 additional files: Qfactor_cohbin_partialCV2_2, Qfactor_strbin_partialCV2_2, filterbin_partialCV2_2
### bits
Allowable values (8: 8-bit depth, 10: 10-bit depth), default (8)

The library supports 8 and 10-bit depth input. Use HEVC encoder to encoder yuv420p10le format.
```
./ffmpeg -y -i [10bits video clip] -vf "raisr=threadcount=20:bits=10" -c:v libx265 -preset medium -crf 28 -pix_fmt yuv420p10le output_10bit.mp4
```
### range
Allowable values (video: video range, full: full range), default (video)

The library caps color within video/full range.
```
./ffmpeg -y -i [image/video file] -vf "raisr=threadcount=20:range=full" outputfile
```
### blending
Allowable values (1: Randomness, 2: CountOfBitsChanged), default (2 )

The library holds two different functions which blend the initial (cheap) upscaled image with the RAISR filtered image.  This can be a means of removing any aggressive or outlying artifacts that get introduced by the filtered image.
### passes
Allowable values (1,2), default(1)

`passes=2` enables a second pass.  Adding a second pass can further enhance the output image quality, but doubles the time to upscale.  Note that for second pass to work, the filter folder must contain 3 additional files: Qfactor_cohbin_partialCV2_2, Qfactor_strbin_partialCV2_2, filterbin_partialCV2_2
### mode
Allowable values (1,2), default(1).  Requires flag passes=2”

Dictates which pass the upscaling should occur in.  Some filters have the best results when it is applied on a high resolution image that was upscaled during a first pass by using mode=1.  Alternatively, the Intel Library for VSR can apply filters on low resolution images during the first pass THEN upscale the image in the second pass if mode=2, for a different outcome.
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:passes=2:mode=2" -pix_fmt yuv420p /output_files/out.yuv
```

# How to Contribute
We welcome community contributions to the Open Visual Cloud repositories. If you have any idea how to improve the project, please share it with us.

## Contribution process
Make sure you can build the project and run tests with your patch.
Submit a pull request at https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/pulls.
The Intel Library for VSR is licensed under the BSD 3-Clause "New" or "Revised" license. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## How to Report Bugs and Provide Feedback
Use the Issues tab on Github.

Intel, the Intel logo and Xeon are trademarks of Intel Corporation or its subsidiaries.