/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "RaisrHandler.h"
#include "Raisr.h"

RNLERRORTYPE RNLHandler_Init(
    const char *modelPath,
    float ratio,
    unsigned int bitDepth,
    RangeType    rangeType,
    unsigned int threadCount,
    ASMType      asmType,
    unsigned int passes,
    unsigned int twoPassMode)
{
    std::string model = modelPath;
    return RNLInit(model, ratio, bitDepth, rangeType, threadCount, asmType, passes, twoPassMode);
}

RNLERRORTYPE RNLHandler_SetRes(
    VideoDataType *inY,
    VideoDataType *inU,
    VideoDataType *inV,
    VideoDataType *outY,
    VideoDataType *outU,
    VideoDataType *outV)
{
    return RNLSetRes(inY, inU, inV, outY, outU, outV);
}

RNLERRORTYPE RNLHandler_Process(
    VideoDataType *inY,
    VideoDataType *inU,
    VideoDataType *inV,
    VideoDataType *outY,
    VideoDataType *outU,
    VideoDataType *outV,
    BlendingMode blendingMode)
{
    return RNLProcess(inY, inU, inV, outY, outU, outV, blendingMode);
}

RNLERRORTYPE RNLHandler_SetOpenCLContext(
        void *context,
        void *device_id,
        int platformIndex,
        int deviceIndex)
{
    return RNLSetOpenCLContext(context, device_id, platformIndex, deviceIndex);
}

RNLERRORTYPE RNLHandler_Deinit()
{
    return RNLDeinit();
}
