/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */

#pragma once

#define defaultPatchSize (11)
static const unsigned int defaultPatchAreaSize = defaultPatchSize * defaultPatchSize;

typedef struct VideoDataType
{
    unsigned char *pData;
    unsigned int width;
    unsigned int height;
    unsigned int step; // distance(in bytes) between the starting points of lines in the image buffer
} VideoDataType;

typedef enum RNLERRORTYPE
{
    RNLErrorNone = 0,
    RNLErrorInsufficientResources = (int)0x80001000,
    RNLErrorUndefined = (int)0x80001001,
    RNLErrorBadParameter = (int)0x80001002,
    RNLErrorMax = (int)0x7FFFFFFF
} RNLERRORTYPE;

typedef enum BlendingMode
{
    Randomness = 1,
    CountOfBitsChanged = 2
} BlendingMode;

typedef enum ASMType
{
    AVX2 = 1,
    AVX512 = 2
} ASMType;

typedef enum RangeType
{
    VideoRange = 1,
    FullRange = 2
} RangeType;
