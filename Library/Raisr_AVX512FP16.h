/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <immintrin.h>

void computeGTWG_Segment_AVX512FP16_16f(const _Float16 *img, const int nrows, const int ncols, const int r, const int col, _Float16 GTWG[3][32], int pix,  _Float16 *buf1, _Float16 *buf2, _Float16 *buf3, _Float16 *buf4);
int CTRandomness_AVX512FP16_16f(_Float16 *inYUpscaled32f, int cols, int r, int c, int pix);
_Float16 DotProdPatch_AVX512FP16_16f(const _Float16 *buf, const _Float16 *filter);
void CTCountOfBitsChangedSegment_AVX512FP16_16f(_Float16 *LRImage, _Float16 *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols);
void GetHashValue_AVX512FP16_16h_8Elements(_Float16 GTWG[3][32], int passIdx, int32_t *idx);
void GetHashValue_AVX512FP16_16h_32Elements(_Float16 GTWG[3][32], int passIdx, int32_t *idx);

