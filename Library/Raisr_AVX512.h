/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <immintrin.h>

void computeGTWG_Segment_AVX512_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[3][16], int pix, float *buf1, float *buf2);
int CTRandomness_AVX512_32f(float *inYUpscaled32f, int cols, int r, int c, int pix);
float DotProdPatch_AVX512_32f(const float *buf, const float *filter);
void GetHashValue_AVX512_32f_16Elements(float GTWG[3][16], int passIdx, int32_t *idx);
