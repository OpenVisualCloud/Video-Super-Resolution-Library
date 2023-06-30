/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#pragma once
#include <immintrin.h>

void inline computeGTWG_Segment_AVX512FP16_16f(const _Float16 *img, const int nrows, const int ncols, const int r, const int col, float GTWG[][4], _Float16 *buf1, _Float16 *buf2);
int inline CTRandomness_AVX512FP16_16f(_Float16 *inYUpscaled32f, int cols, int r, int c, int pix);
_Float16 inline DotProdPatch_AVX512FP16_16f(const _Float16 *buf, const _Float16 *filter);
void inline CTCountOfBitsChangedSegment_AVX512FP16_16f(_Float16 *LRImage, _Float16 *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols);
