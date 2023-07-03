/**
* Intel Library for Video Super Resolution
*
* Copyright (c) 2022 Intel Corporation
* SPDX-License-Identifier: BSD-3-Clause
*/

#pragma once
#include <vector>
#include <string>
#include "RaisrDefaults.h"
#include "RaisrVersion.h"

RNLERRORTYPE RNLInit(std::string &modelPath,
                     float ratio,
                     unsigned int bitDepth = 8,
                     RangeType rangeType = VideoRange,
                     unsigned int threadCount = 20,
                     ASMType asmType = AVX512,
                     unsigned int passes = 1,
                     unsigned int twoPassMode = 1);

RNLERRORTYPE RNLSetRes(VideoDataType *inY, VideoDataType *inCr, VideoDataType *inCb,
                       VideoDataType *outY, VideoDataType *outCr, VideoDataType *outCb);

RNLERRORTYPE RNLProcess(VideoDataType *inY, VideoDataType *inCr, VideoDataType *inCb,
                        VideoDataType *outY, VideoDataType *outCr, VideoDataType *outCb,
                        BlendingMode blendingMode = CountOfBitsChanged);

RNLERRORTYPE RNLSetOpenCLContext(void *context, void *device_id,
                                 int platformIndex, int deviceIndex);

RNLERRORTYPE RNLDeinit();
