/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */
#pragma once
#include <immintrin.h>

void inline computeGTWG_Segment_AVX512_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[][4], float *buf1, float *buf2);
int inline CTRandomness_AVX512_32f(float *inYUpscaled32f, int cols, int r, int c, int pix);
float inline DotProdPatch_AVX512_32f(const float *buf, const float *filter);
