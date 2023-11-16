# Intel® Library for Video Super Resolution (Intel® Library for VSR) Release Notes

# New and Changed in Release v23.11

**v23.11**

## New and Changed in v23.11
- NEW:Enabled OpenCL acceleration, supported Intel® GPUs platforms.
- NEW:Enabled AVX512-FP16 for faster processing on 4th generation Intel® Xeon Scalable processor.
- NEW:Supported 1.5x upscaling for 8-bit.
- NEW:Added filters with denoising effect.
- NEW:Supported YUV444 format in ffmpeg plugin which can improve quality for input and output in RBG or YUV444 format.
- NEW:Added new dockerfiles with Ubuntu 22.04 and CentOS 7.9.
- Upgraded ffmpeg to n6.0.
- Improved performance for Intel® Xeon platforms.
- Optimized the filter size and enhanced filter file format to binary.
- Removed the filters2, filters3, filters5 and renamed filters1->filters_2x/filters_lowres and filters4->filters_2x/filters_highres, and improved the quality of filters_highres.

## Bug Fixes
- Fixed the segmentation fault issue with some specific threadcount.
- Fixed the black block issue with blending=1 cases.

## Known Limitations
- Only 2x and 1.5x upscaling supported.
- patchSize passed into RNLInit must be set to 11. Algorithms are currently tuned to work only with a patch size of 11. Values other than 11 will fail.
- For usage of Intel AVX2, system hardware must be run on Intel Haswell Processor or later. For usage of Intel AVX-512, system hardware must be run on Intel Xeon Scalable Processesors (1st Gen or later, Skylake or later ). For usage of Intel AVX512-FP16, system hardware must be run on 4th generation Intel Xeon Scalable Processesors or later.

# New and Changed in Release v22.12

**v22.12**

## New Features
- Support the use of Intel AVX2 instructions. See README for usage.
- Performance Optimizations: improved performance via using AVX2 instructions to enhance some functions.
- Add scripts that support setup Raisr without internet access. These scripts are in scripts folder.

## Bug Fixes
- Fixed ffmpeg compilation issue with low GCC 7.5.0.
- Fixed the issue that some video resolutions were not working correctly.
- Ehanced inspection of inputs.

## Known Limitations
- Only 2x upscaling supported. Ratio passed into RNLInit should be set to 2.
- patchSize passed into RNLInit must be set to 11. Algorithms are currently tuned to work only with a patch size of 11. Values other than 11 will fail.
- For usage of Intel AVX2, system hardware must be run on Intel Haswell Processor or later. For usage of Intel AVX-512, system hardware must be run on Intel Xeon Scalable Processesors (1st Gen or later, Skylake or later )

# Release Notes in Release v22.9

**v22.9**

This release is packaged as a docker container and should contain everything one needs to evaluate the super resolution upscaling of the Intel Library for VSR.  This project is under active development, but is well suited for evaluation.  Please refer to the included README for guidance surrounding building the plugin and running it.


## Package Contents
- filters1\, filters2\, filters3\, filters4\, filters5\ - 5 folders containing filters.  Each filter can produce different results.  See README
- license\ - license containing use, terms and limitations
- 0001-ffmpeg-raisr-filter.patch  - patch to create raisr plugin for ffmpeg branch n6.0
- Dockerfile – used to create a docker image
- Library\ - contains .cpp source code files and header files outlining project’s development APIs 

## New Features
- Docker support included
- FFmpeg raisr filter added, allowing upscaling of images and video.  See README for usage.
- Second pass added to allow for sharpening of raisr output.
- Multithreaded (threadcount=[1-120]) usage will segment data to multiple threads to increase fps with efficient parallelism
- 5 trained filters included in evaluation, trained with different parameters, producing results with varying sharpness
- Optimized for modern Intel® Xeon® Scalable servers (requires Intel® Advanced Vector Extensions 512 (Intel® AVX-512) instructions, used for performance efficiency)
## Known Limitations
- Ratio passed into RNLInit should be set to 2.  2x upscaling matches what has been validated (e.g. 1080p->4k).  Values other than 2 may produce unexpected results.
- patchSize passed into RNLInit must be set to 11.  Algorithms are currently tuned to work only with a patch size of 11.  Values other than 11 may produce unexpected results.
- Because of usage of Intel AVX-512, system hardware must be run on Intel Xeon Scalable Processesors (1st Gen or later, Skylake or later )
