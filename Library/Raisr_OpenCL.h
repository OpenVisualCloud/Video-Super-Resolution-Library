/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once

#ifndef CL_TARGET_OPENCL_VERSION
#define CL_TARGET_OPENCL_VERSION 300
#endif

#include <CL/cl.h>
#include <vector>
#include "RaisrDefaults.h"

typedef struct RaisrOpenCLContext {
    void *priv;
    cl_context context;
    cl_device_id deviceID;
    int widthMax;
    int heightMax;
    float gRatio;
    ASMType gAsmType;
    int platformIndex;
    int deviceIndex;
    unsigned int gBitDepth;
    unsigned int gQuantizationAngle;
    unsigned int gQuantizationStrength;
    unsigned int gQuantizationCoherence;
    unsigned int gPatchSize;
    std::vector<float> gQStr;
    std::vector<float> gQCoh;
    std::vector<float> gQStr2;
    std::vector<float> gQCoh2;
    float *gFilterBuffer;
    float *gFilterBuffer2;
    int gPasses;
    int gTwoPassMode;
    unsigned char gMin8bit;
    unsigned char gMax8bit;
    unsigned short gMin16bit;
    unsigned short gMax16bit;
    int gUsePixelType;
} RaisrOpenCLContext;

RNLERRORTYPE RaisrOpenCLInit(RaisrOpenCLContext *raisrOpenCLContext);

RNLERRORTYPE RaisrOpenCLSetRes(RaisrOpenCLContext *raisrOpenCLContext,
                               int widthY, int heightY, int widthUV, int heightUV,
                               int nvComponent);

RNLERRORTYPE RaisrOpenCLProcessY(RaisrOpenCLContext *raisrOpenCLContext,
                                 uint8_t *inputY, int width, int height, int linesizeI,
                                 uint8_t *outputY, int linesizeO, int bitShift, BlendingMode blend);

RNLERRORTYPE RaisrOpenCLProcessUV(RaisrOpenCLContext *raisrOpenCLContext,
                                  uint8_t *inputUV, int width, int height,
                                  int linesizeI, uint8_t *outputUV, int linesizeO,
                                  int bitShift, int nbComponent);

RNLERRORTYPE RaisrOpenCLProcessImageY(RaisrOpenCLContext* raisrOpenCLContext,
                                      cl_mem inputImageY, int width, int height,
                                      cl_mem outputImageY, int bitShift, BlendingMode blend);

RNLERRORTYPE RaisrOpenCLProcessImageUV(RaisrOpenCLContext* raisrOpenCLContext,
                                       cl_mem inputImageUV, int width, int height, cl_mem outputImageUV,
                                       int bitShift, int nbComponent);

void RaisrOpenCLRelease(RaisrOpenCLContext *raisrOpenCLContext);
