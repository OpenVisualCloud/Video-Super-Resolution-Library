/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */
#pragma once
#include <immintrin.h>

inline __mmask8 compare3x3_ps_AVX512(__m256 a, __m256 b);
inline float sumitup_ps_512(__m512 acc);
inline __m512 shiftL(__m512 r);
inline __m512 shiftR(__m512 r);
inline __m512 GetGx(__m512 r1, __m512 r3);
inline __m512 GetGy(__m512 r2);
inline __m512 GetGTWG(__m512 acc, __m512 a, __m512 w, __m512 b);

void inline computeGTWG_Segment_AVX512_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[][4], float *buf1, float *buf2);
int inline CTRandomness_AVX512_32f(float *inYUpscaled32f, int cols, int r, int c, int pix);
float inline DotProdPatch_AVX512_32f(const float *buf, const float *filter);
