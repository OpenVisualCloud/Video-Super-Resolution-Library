# Intel® Library for Video Super Resolution (Intel® Library for VSR) README
Video Super Resolution converts video from low resolution to high resolution using traditional image processing or AI-based methods. Intel Library for Video Super Resolution consist of a few different algorithms including machine learning and deep learning implementations to offer a balance between quality and performance.

We have enhanced the public RAISR (Rapid and Accurate Image Super Resolution), an AI based Super Resolution algorithm https://arxiv.org/pdf/1606.01299.pdf, to achieve better visual quality and beyond real-time performance for 2x and 1.5x upscaling on Intel® Xeon® platforms and Intel® GPUs. Enhanced RAISR provides better quality results than standard (bicubic) algorithms and a good performance vs quality trade-off as compared to compute intensive DL-based algorithms. 

Enhanced RAISR is provided as an FFmpeg plugin inside of a Docker container(Docker container only for CPU) to help ease testing and deployment burdens. This project is developed using C++ and takes advantage of Intel® Advanced Vector Extension 512 (Intel® AVX-512) on Intel® Xeon® Scalable Processor family and  OpenCL support on Intel® GPUs.

![image](https://github.com/user-attachments/assets/e28b52c2-67c7-44a9-a66f-df8b355735f9)

## Latest News 
- July 2024 : Release performance of the alogorithm and pipeline on Intel® Xeon® Scalable processor as well as EC2 Intel instances deployed on AWS Cloud. See details at [performance.md](./docs/performance.md).  

- April 2024: Intel Library for Video Super Resolution algorithm now available on AWS. See the repository for details on how video super resolution works on the AWS service at https://github.com/aws-samples/video-super-resolution-tool. Technical details including video quality comparisons and performance information are available in a joint Intel / AWS white paper available at https://www.intel.com/content/www/us/en/content-details/820769/aws-compute-video-super-resolution-powered-by-the-intel-library-for-video-super-resolution.html 

- Feb 2024 : AWS and Intel announced collaboration to release Intel Library for VSR on AWS Cloud at the Mile High Video 2024 conference, technical details available at  https://dl.acm.org/doi/10.1145/3638036.3640290  

We have enhanced the public RAISR algorithm to achieve better visual quality and beyond real-time performance for 2x and 1.5x upscaling on Intel® Xeon® platforms and Intel® GPUs. The Intel Library for VSR is provided as an FFmpeg plugin inside of a Docker container to help ease testing and deployment burdens. This project is developed using C++ and takes advantage of Intel® Advanced Vector Extension 512 (Intel® AVX-512) where available and newly added Intel® AVX-512FP16 support on Intel® Xeon® 4th Generation (Sapphire Rapids) and added OpenCL support on Intel® GPUs.

## How to build
Please see "How to build.md" to build via scripts or manually.

## Running the Intel Library for VSR 
One should be able to test with video files:
```
./ffmpeg -y -i /input_files/input.mp4 -vf raisr=threadcount=20 -pix_fmt yuv420p /output_files/out.yuv
```
Or folders of images:
```
./ffmpeg -y -start_number 000 -i '/input_files/img_%03d.png' -vf scale=out_range=full,raisr=threadcount=20 -start_number 000 '/output_files/img_%03d.png'
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
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:passes=2:filterfolder=filters_2x/filters_highres" -pix_fmt yuv420p /output_files/out.yuv
```
**Fastest Performance ( second pass disabled )**
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:filterfolder=filters_2x/filters_lowres" -pix_fmt yuv420p /output_files/out.yuv
```
**Denoised output**
```
./ffmpeg -i /input_files/input.mp4 -vf "raisr=threadcount=20:passes=2:mode=2:filterfolder=filters_2x/filters_denoise" -pix_fmt yuv420p /output_files/out.yuv
```
2. A source video or image can be downscaled by 2x, then passed through the RAISR filter which upscales by 2x
```
./ffmpeg -y -i /input_files/input.mp4 -vf scale=iw/2:ih/2,raisr=threadcount=20 -pix_fmt yuv420p /output_files/out.yuv
```
At this point the source content is the same resolution as the output and the two can be compared to understand how well the super resolution is working.  RAISR can be compared against existing DL super resolution algorithms as well.  It is recommended to enable second pass in Intel Library for VSR to produce sharper images.  Please see the Advanced Usage section for guidance on enabling second pass as a feature.

**OpenCL acceleration**
```
./ffmpeg -y -i /input_files/input.mp4 -vf raisr=asm=opencl -pix_fmt yuv420p /output_files/out.yuv
```
or user can use filter "raisr_opencl" to build full gpu pipeline. \
[ffmpeg-qsv](https://trac.ffmpeg.org/wiki/Hardware/QuickSync) \
[ffmpeg-vaapi](https://trac.ffmpeg.org/wiki/Hardware/VAAPI)
```
ffmpeg -init_hw_device vaapi=va -init_hw_device qsv=qs@va -init_hw_device opencl=ocl@va -hwaccel qsv -c:v h264_qsv -i input.264 -vf "hwmap=derive_device=opencl,format=opencl,raisr_opencl,hwmap=derive_device=qsv:reverse=1:extra_hw_frames=16" -c:v hevc_qsv output.mp4
```
```
ffmpeg -init_hw_device vaapi=va -init_hw_device opencl=ocl@va -hwaccel vaapi -hwaccel_output_format vaapi -i input.264 -vf "hwmap=derive_device=opencl,format=opencl,raisr_opencl,hwmap=derive_device=vaapi:reverse=1:extra_hw_frames=16" -c:v hevc_vaapi output.mp4
```

**Even output**

There are certain codecs that support only even resolution, the `evenoutput` parameter will support users to choose whether to make the output an even number

Set `evenoutput=1` to make output size as even number, the following command will get 632x632 output.
```
ffmpeg -i input.mp4 -an -vf scale=422x422,raisr=ratio=1.5:filterfolder=filters_1.5x/filters_highres:threadcount=1:evenoutput=1 output.mp4
```
It will keep the output resolution as the input resolution multiply by the upscaling ratio if set `evenoutput=0` or not set the parameter, will get 633x633 output with 422x422 input.

## To see help on the RAISR filter
`./ffmpeg -h filter=raisr`

    raisr AVOptions:
      ratio             <float>      ..FV....... ratio of the upscaling, between 1 and 2 (default 2)
      bits              <int>        ..FV....... bit depth (from 8 to 10) (default 8)
      range             <string>     ..FV....... color range of the input. If you are working with images, you may want to set range to full (video/full) (default video)
      threadcount       <int>        ..FV....... thread count (from 1 to 120) (default 20)
      filterfolder      <string>     ..FV....... absolute filter folder path (default "filters_2x/filters_lowres")
      blending          <int>        ..FV....... CT blending mode (1: Randomness, 2: CountOfBitsChanged) (from 1 to 2) (default 2)
      passes            <int>        ..FV....... passes to run (1: one pass, 2: two pass) (from 1 to 2) (default 1)
      mode              <int>        ..FV....... mode for two pass (1: upscale in 1st pass, 2: upscale in 2nd pass) (from 1 to 2) (default 1)
      asm               <string>     ..FV....... x86 asm type: (avx512fp16, avx512, avx2 or opencl) (default "avx512fp16")
      platform          <int>        ..FV....... select the platform (from 0 to INT_MAX) (default 0)
      device            <int>        ..FV....... select the device (from 0 to INT_MAX) (default 0)
      evenoutput        <int>        ..FV....... make output size as even number (0: ignore, 1: subtract 1px if needed) (from 0 to 1) (default 0)


# How to Contribute
We welcome community contributions to the Open Visual Cloud repositories. If you have any idea how to improve the project, please share it with us.

## Contribution process
Make sure you can build the project and run tests with your patch.
Submit a pull request at https://github.com/OpenVisualCloud/Video-Super-Resolution-Library/pulls.
The Intel Library for VSR is licensed under the BSD 3-Clause "New" or "Revised" license. By contributing to the project, you agree to the license and copyright terms therein and release your contribution under these terms.

## How to Report Bugs and Provide Feedback
Use the Issues tab on Github.

Intel, the Intel logo and Xeon are trademarks of Intel Corporation or its subsidiaries.

# License

Intel Library for VSR is licensed under the BSD 3-clause license.
