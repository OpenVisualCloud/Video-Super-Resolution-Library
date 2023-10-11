/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "Raisr_globals.h"
#include "Raisr_AVX512.h"
#include "Raisr_AVX256.h"
#include <immintrin.h>
#include <popcntintrin.h>
#include <cmath>

inline __mmask8 compare3x3_ps_AVX512(__m256 a, __m256 b)
{
    return _mm256_cmp_ps_mask(a, b, _CMP_LT_OS);
}

int CTRandomness_AVX512_32f(float *inYUpscaled32f, int cols, int r, int c, int pix)
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
inline __m512 shiftL_AVX512(__m512 r)
{
    return _mm512_permutexvar_ps(_mm512_set_epi32(0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1), r);
}
inline __m512 shiftR_AVX512(__m512 r)
{
    return _mm512_permutexvar_ps(_mm512_set_epi32(14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15), r);
}

inline __m512 GetGx_AVX512(__m512 r1, __m512 r3)
{
    return _mm512_sub_ps(r3, r1);
}

inline __m512 GetGy_AVX512(__m512 r2)
{
    return _mm512_sub_ps(shiftL_AVX512(r2), shiftR_AVX512(r2));
}

inline __m512 GetGTWG_AVX512(__m512 acc, __m512 a, __m512 w, __m512 b)
{
    return _mm512_fmadd_ps(_mm512_mul_ps(a, w), b, acc);
}

void computeGTWG_Segment_AVX512_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[3][16], int pix, float *buf1, float *buf2)
{
    // offset is the starting position(top left) of the block which centered by (r, c)
    int gtwgIdx = pix * 2;
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

        const __m512 gxi = GetGx_AVX512(a, c);
        const __m512 gyi = GetGy_AVX512(b);

        gtwg0A = GetGTWG_AVX512(gtwg0A, gxi, w, gxi);
        gtwg1A = GetGTWG_AVX512(gtwg1A, gxi, w, gyi);
        gtwg3A = GetGTWG_AVX512(gtwg3A, gyi, w, gyi);

        w = shiftR_AVX512(w);
        gtwg0B = GetGTWG_AVX512(gtwg0B, gxi, w, gxi);
        gtwg1B = GetGTWG_AVX512(gtwg1B, gxi, w, gyi);
        gtwg3B = GetGTWG_AVX512(gtwg3B, gyi, w, gyi);

        _mm512_mask_storeu_ps(buf1 + gPatchSize * i - 1, 0x0ffe, b);
        _mm512_mask_storeu_ps(buf2 + gPatchSize * i - 2, 0x1ffc, b);
        a = b;
        b = c;
    }

    GTWG[0][gtwgIdx] = sumitup_ps_512(gtwg0A);
    GTWG[1][gtwgIdx] = sumitup_ps_512(gtwg1A);
    GTWG[2][gtwgIdx] = sumitup_ps_512(gtwg3A);

    GTWG[0][gtwgIdx+1] = sumitup_ps_512(gtwg0B);
    GTWG[1][gtwgIdx+1] = sumitup_ps_512(gtwg1B);
    GTWG[2][gtwgIdx+1] = sumitup_ps_512(gtwg3B);

    return;
}

// AVX512 version: for now, gPatchSize must be <= 16 because we can work with up to 16 float32s in one AVX512 register.
float DotProdPatch_AVX512_32f(const float *buf, const float *filter)
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

inline __m512 atan2Approximation_AVX512_32f_16Elements(__m512 y_ps, __m512 x_ps)
{
    const float ONEQTR_PI = M_PI / 4.0;
    const float THRQTR_PI = 3.0 * M_PI / 4.0;
    const __m512 zero_ps = _mm512_set1_ps(0.0);
    const __m512 oneqtr_pi_ps = _mm512_set1_ps(ONEQTR_PI);
    const __m512 thrqtr_pi_ps = _mm512_set1_ps(THRQTR_PI);

    __m512 abs_y_ps = _mm512_add_ps( _mm512_abs_ps(y_ps), _mm512_set1_ps(1e-10f)); 

    __m512 r_cond1_ps = _mm512_div_ps( _mm512_add_ps(x_ps, abs_y_ps), _mm512_sub_ps(abs_y_ps, x_ps));
    __m512 r_cond2_ps = _mm512_div_ps( _mm512_sub_ps(x_ps, abs_y_ps), _mm512_add_ps(x_ps, abs_y_ps));
    __mmask16 r_cmp_m8 =  _mm512_cmp_ps_mask(x_ps, zero_ps, _CMP_LT_OQ);
    __m512 r_ps = _mm512_mask_blend_ps( r_cmp_m8, r_cond2_ps, r_cond1_ps);
    __m512 angle_ps = _mm512_mask_blend_ps( r_cmp_m8, oneqtr_pi_ps, thrqtr_pi_ps);

    angle_ps = _mm512_fmadd_ps(_mm512_fmadd_ps(_mm512_mul_ps(_mm512_set1_ps(0.1963f), r_ps),
                                                                                    r_ps, _mm512_set1_ps(-0.9817f)),
                                                                                    r_ps, angle_ps);

    __m512 neg_angle_ps = _mm512_mul_ps(_mm512_set1_ps(-1), angle_ps);
    return _mm512_mask_blend_ps(  _mm512_cmp_ps_mask(y_ps, zero_ps, _CMP_LT_OQ), angle_ps, neg_angle_ps );
}

void GetHashValue_AVX512_32f_16Elements(float GTWG[3][16], int passIdx, int32_t *idx) {
    const float one = 1.0;
    const float two = 2.0;
    const float four = 4.0;
    const float pi = PI;
    const float near_zero = 0.00000000000000001;

    const __m512 zero_ps = _mm512_setzero_ps();
    const __m512 one_ps = _mm512_set1_ps(1);
    const __m512i zero_epi32 = _mm512_setzero_si512();
    const __m512i one_epi32 = _mm512_set1_epi32(1);
    const __m512i two_epi32 = _mm512_set1_epi32(2);

    const int cmp_le = _CMP_LE_OQ;
    const int cmp_gt = _CMP_GT_OQ;

    __m512 m_a_ps = _mm512_load_ps( GTWG[0]);
    __m512 m_b_ps = _mm512_load_ps( GTWG[1]);
    __m512 m_d_ps = _mm512_load_ps( GTWG[2]);

    __m512 T_ps = _mm512_add_ps(m_a_ps, m_d_ps);
    __m512 D_ps = _mm512_sub_ps( _mm512_mul_ps( m_a_ps, m_d_ps),
                                _mm512_mul_ps( m_b_ps, m_b_ps));

    // 11 bit accuracy: fast sqr root
    __m512 sqr_ps = _mm512_rcp14_ps( _mm512_rsqrt14_ps( _mm512_sub_ps( _mm512_div_ps ( _mm512_mul_ps(T_ps, T_ps),
                                                    _mm512_set1_ps(four)), D_ps)));

    __m512 half_T_ps = _mm512_div_ps ( T_ps, _mm512_set1_ps(two) );
    __m512 L1_ps = _mm512_add_ps( half_T_ps, sqr_ps);
    __m512 L2_ps = _mm512_sub_ps( half_T_ps, sqr_ps);

    __m512 angle_ps = zero_ps;

    __m512 blend_ps = _mm512_mask_blend_ps( _mm512_cmp_ps_mask(m_b_ps, zero_ps, _CMP_NEQ_OQ),
                                            one_ps, _mm512_sub_ps(L1_ps, m_d_ps));

#ifdef USE_ATAN2_APPROX
    angle_ps = atan2Approximation_AVX512_32f_16Elements( m_b_ps, blend_ps);
#else
    angle_ps = _mm512_atan2_ps( m_b_ps, blend_ps);
#endif

    angle_ps = _mm512_add_ps ( angle_ps, _mm512_mask_blend_ps( _mm512_cmp_ps_mask(angle_ps, zero_ps, _CMP_LT_OQ), zero_ps, _mm512_set1_ps(pi)));

    // fast sqrt 
    __m512 sqrtL1_ps = _mm512_rcp14_ps( _mm512_rsqrt14_ps( L1_ps ));
    __m512 sqrtL2_ps = _mm512_rcp14_ps( _mm512_rsqrt14_ps( L2_ps ));

    __m512 coherence_ps = _mm512_div_ps( _mm512_sub_ps( sqrtL1_ps, sqrtL2_ps ),
                                        _mm512_add_ps( _mm512_add_ps(sqrtL1_ps, sqrtL2_ps), _mm512_set1_ps(near_zero) ) );
    __m512 strength_ps = L1_ps;

    __m512i angleIdx_epi32 = _mm512_cvtps_epi32( _mm512_floor_ps(_mm512_mul_ps (angle_ps, _mm512_set1_ps(gQAngle))));
    __m512i quantAngle_lessone_epi32 = _mm512_sub_epi32(_mm512_set1_epi32(gQuantizationAngle), one_epi32);
    angleIdx_epi32 = _mm512_min_epi32( _mm512_sub_epi32( _mm512_set1_epi32(gQuantizationAngle), _mm512_set1_epi32(1)),
                                       _mm512_max_epi32(angleIdx_epi32, zero_epi32 ) );

   // AFAIK, today QStr & QCoh are vectors of size 2.  I think searchsorted can return an index of 0,1, or 2
    float *gQStr_data, *gQCoh_data;
    if (passIdx == 0) gQStr_data = gQStr.data(); else gQStr_data = gQStr2.data();
    if (passIdx == 0) gQCoh_data = gQCoh.data(); else gQCoh_data = gQCoh2.data();
    __m512 gQStr1_ps = _mm512_set1_ps(gQStr_data[0]);
    __m512 gQStr2_ps = _mm512_set1_ps(gQStr_data[1]);
    __m512 gQCoh1_ps = _mm512_set1_ps(gQCoh_data[0]);
    __m512 gQCoh2_ps = _mm512_set1_ps(gQCoh_data[1]);

    __m512i strengthIdx_epi32 = 
                                    _mm512_add_epi32(
                                        _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(gQStr1_ps, strength_ps, _MM_CMPINT_LE),zero_epi32, one_epi32),
                                        _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(gQStr2_ps, strength_ps, _MM_CMPINT_LE),zero_epi32, one_epi32));
    __m512i coherenceIdx_epi32 = 
                                    _mm512_add_epi32(
                                        _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(gQCoh1_ps, coherence_ps, _MM_CMPINT_LE),zero_epi32, one_epi32),
                                        _mm512_mask_blend_epi32(_mm512_cmp_ps_mask(gQCoh2_ps, coherence_ps, _MM_CMPINT_LE),zero_epi32, one_epi32));

    const __m512i gQuantizationCoherence_epi32 = _mm512_set1_epi32(gQuantizationCoherence);
    __m512i idx_epi32 = _mm512_mullo_epi32(gQuantizationCoherence_epi32,
                                            _mm512_mullo_epi32(angleIdx_epi32, _mm512_set1_epi32(gQuantizationStrength)));
    idx_epi32 = _mm512_add_epi32(coherenceIdx_epi32,
                                _mm512_add_epi32(idx_epi32, _mm512_mullo_epi32(strengthIdx_epi32, gQuantizationCoherence_epi32)));

    _mm512_storeu_si512((__m512i *)idx, idx_epi32);
}

