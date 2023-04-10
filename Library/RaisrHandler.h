/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#pragma once
#include "RaisrDefaults.h"
#ifdef __cplusplus
extern "C"
{
#endif

    extern RNLERRORTYPE RNLHandler_Init(
        const char *modelPath,
        unsigned int ratio,
        unsigned int bitDepth,
        RangeType    rangeType,
        unsigned int threadCount,
        ASMType      asmType,
        unsigned int passes,
        unsigned int twoPassMode);

    extern RNLERRORTYPE RNLHandler_SetRes(
        VideoDataType *inY,
        VideoDataType *inCr,
        VideoDataType *inCb,
        VideoDataType *outY,
        VideoDataType *outCr,
        VideoDataType *outCb);

    extern RNLERRORTYPE RNLHandler_Process(
        VideoDataType *inY,
        VideoDataType *inCr,
        VideoDataType *inCb,
        VideoDataType *outY,
        VideoDataType *outCr,
        VideoDataType *outCb,
        BlendingMode blendingMode);

    extern RNLERRORTYPE RNLHandler_SetOpenCLContext(
        void *context,
        void *device_id,
        int platformIndex,
        int deviceIndex);

    extern RNLERRORTYPE RNLHandler_Deinit();
#ifdef __cplusplus
}
#endif
