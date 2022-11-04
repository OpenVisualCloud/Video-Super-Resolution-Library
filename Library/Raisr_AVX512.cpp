/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */
#include "Raisr_globals.h"
#include "Raisr_AVX512.h"
#include <immintrin.h>
#include <popcntintrin.h>

inline __mmask8 compare3x3_ps_AVX512(__m256 a, __m256 b)
{
    return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

int inline CTRandomness_AVX512_32f(float *inYUpscaled32f, int cols, int r, int c, int pix)
{
    int census_count = 0;

    __m128 zero_f = _mm_setzero_ps();
    __m256 row_f, center_f;

    load3x3_ps(inYUpscaled32f, c + pix, r, cols, &row_f, &center_f);

    // compare if neighbors < centerpixel, toggle bit in mask if true
    __mmask8 cmp_m8 = compare3x3_ps_AVX512(row_f, center_f);

    // count # of bits in mask
    census_count += _mm_popcnt_u32(cmp_m8);

    return census_count;
}

inline float sumitup_ps_512(__m512 acc)
{
    const __m256 r8 = _mm256_add_ps(_mm512_castps512_ps256(acc), _mm512_extractf32x8_ps(acc, 1));
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(r8), _mm256_extractf128_ps(r8, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    return _mm_cvtss_f32(r1);
}
inline __m512 shiftL(__m512 r)
{
    return _mm512_permutexvar_ps(_mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1), r);
}
inline __m512 shiftR(__m512 r)
{
    return _mm512_permutexvar_ps(_mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15), r);
}

inline __m512 GetGx(__m512 r1, __m512 r3)
{
    return _mm512_sub_ps(r3, r1);
}

inline __m512 GetGy(__m512 r2)
{
    return _mm512_sub_ps(shiftL(r2), shiftR(r2));
}

inline __m512 GetGTWG(__m512 acc, __m512 a, __m512 w, __m512 b)
{
    return _mm512_fmadd_ps(_mm512_mul_ps(a, w), b, acc);
}

void inline computeGTWG_Segment_AVX512_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[][4], float *buf1, float *buf2)
{
    // offset is the starting position(top left) of the block which centered by (r, c)
    int offset = (r - gLoopMargin) * ncols + col - gLoopMargin;
    const float *p1 = img + offset;

    __m512 gtwg0A = _mm512_setzero_ps(), gtwg1A = _mm512_setzero_ps(), gtwg3A = _mm512_setzero_ps();
    __m512 gtwg0B = _mm512_setzero_ps(), gtwg1B = _mm512_setzero_ps(), gtwg3B = _mm512_setzero_ps();

    // load 2 rows
    __m512 a = _mm512_loadu_ps(p1);
    p1 += ncols;
    __m512 b = _mm512_loadu_ps(p1);
#pragma unroll
    for (int i = 0; i < gPatchSize; i++)
    {
        // memcpy(buf1+gPatchSize*i, p1+1, sizeof(float)*gPatchSize);
        // memcpy(buf2+gPatchSize*i, p1+2, sizeof(float)*gPatchSize);

        // process patchSize rows
        // load next row
        p1 += ncols;
        __m512 c = _mm512_loadu_ps(p1);
        __m512 w;
        if (gBitDepth == 8)
        {
            w = _mm512_loadu_ps(gGaussian2D8bit[i]);
        }
        else if (gBitDepth == 10)
        {
            w = _mm512_loadu_ps(gGaussian2D10bit[i]);
        }
        else
        {
            w = _mm512_loadu_ps(gGaussian2D16bit[i]);
        }

        const __m512 gxi = GetGx(a, c);
        const __m512 gyi = GetGy(b);

        gtwg0A = GetGTWG(gtwg0A, gxi, w, gxi);
        gtwg1A = GetGTWG(gtwg1A, gxi, w, gyi);
        gtwg3A = GetGTWG(gtwg3A, gyi, w, gyi);

        w = shiftR(w);
        gtwg0B = GetGTWG(gtwg0B, gxi, w, gxi);
        gtwg1B = GetGTWG(gtwg1B, gxi, w, gyi);
        gtwg3B = GetGTWG(gtwg3B, gyi, w, gyi);

        _mm512_mask_storeu_ps(buf1 + gPatchSize * i - 1, 0x0ffe, b);
        _mm512_mask_storeu_ps(buf2 + gPatchSize * i - 2, 0x1ffc, b);
        a = b;
        b = c;
    }
    GTWG[0][0] = sumitup_ps_512(gtwg0A);
    GTWG[0][1] = sumitup_ps_512(gtwg1A);
    GTWG[0][3] = sumitup_ps_512(gtwg3A);
    GTWG[0][2] = GTWG[0][1];

    GTWG[1][0] = sumitup_ps_512(gtwg0B);
    GTWG[1][1] = sumitup_ps_512(gtwg1B);
    GTWG[1][3] = sumitup_ps_512(gtwg3B);
    GTWG[1][2] = GTWG[1][1];

    return;
}

// AVX512 version: for now, gPatchSize must be <= 16 because we can work with up to 16 float32s in one AVX512 register.
float inline DotProdPatch_AVX512_32f(const float *buf, const float *filter)
{
    __m512 a_ps = _mm512_load_ps(buf);
    __m512 b_ps = _mm512_load_ps(filter);
    __m512 sum = _mm512_mul_ps(a_ps, b_ps);
#pragma unroll
    for (int i = 1; i < 8; i++)
    {
        a_ps = _mm512_load_ps(buf + i * 16);
        b_ps = _mm512_load_ps(filter + i * 16);
        // compute dot prod using fmadd
        sum = _mm512_fmadd_ps(a_ps, b_ps, sum);
    }
    // sumitup adds all 16 float values in sum(zmm) and returns a single float value
    return sumitup_ps_512(sum);
}
