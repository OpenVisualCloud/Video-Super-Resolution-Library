/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */
#include "Raisr_globals.h"
#include "Raisr_AVX256.h"
#include <immintrin.h>



inline __m256i compare3x3_ps(__m256 a, __m256 b, __m256i highbit_epi32)
{
    // compare if neighbors < centerpixel, toggle bit in mask if true
    // when cmp_ps is true, it returns 0x7fffff (-nan).  When we convert that to int, it is 0x8000 0000

    return _mm256_srli_epi32(_mm256_and_si256(_mm256_cvtps_epi32(
                                                  _mm256_cmp_ps(a, b, _CMP_LT_OS)),
                                              highbit_epi32),
                             31); // shift right by 31 such that the high bit (if set) moves to the low bit
}

inline int sumitup_256_epi32(__m256i acc)
{
    const __m128i r4 = _mm_add_epi32(_mm256_castsi256_si128(acc), _mm256_extractf128_si256(acc, 1));
    const __m128i r2 = _mm_hadd_epi32(r4, r4);
    const __m128i r1 = _mm_hadd_epi32(r2, r2);
    return _mm_cvtsi128_si32(r1);
}

int inline CTRandomness_AVX256_32f(float *inYUpscaled32f, int cols, int r, int c, int pix)
{
    int census_count = 0;

    __m128 zero_f = _mm_setzero_ps();
    __m256 row_f, center_f;

    load3x3_ps(inYUpscaled32f, c + pix, r, cols, &row_f, &center_f);

    // compare if neighbors < centerpixel, toggle bit in mask if true
    int highbit = 0x80000000;
    const __m256i highbit_epi32 = _mm256_setr_epi32(highbit, highbit, highbit, highbit, highbit, highbit, highbit, highbit);

    __m256i cmp_epi32 = compare3x3_ps(row_f, center_f, highbit_epi32);

    // count # of bits in mask
    census_count += sumitup_256_epi32(cmp_epi32);

    return census_count;
}

inline float sumitup_ps_256(__m256 acc)
{
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    return _mm_cvtss_f32(r1);
}
