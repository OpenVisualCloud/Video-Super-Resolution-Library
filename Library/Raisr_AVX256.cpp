/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "Raisr_globals.h"
#include "Raisr_AVX256.h"
#include <immintrin.h>
#include <string.h>
#include <cmath>



inline __m256i compare3x3_AVX256_32f(__m256 a, __m256 b, __m256i highbit_epi32)
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

    __m256i cmp_epi32 = compare3x3_AVX256_32f(row_f, center_f, highbit_epi32);

    // count # of bits in mask
    census_count += sumitup_256_epi32(cmp_epi32);

    return census_count;
}

inline __m256i HAMMING_DISTANCE_EPI32( __m256i hammDist, __m256 neigh_LR, __m256 center_LR, __m256 neigh_HR, __m256 center_HR) {
    const __m256i one_epi32 = _mm256_set1_epi32(1);
    return _mm256_add_epi32( hammDist,
                             _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(neigh_LR, center_LR, _CMP_LT_OQ))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(neigh_HR, center_HR, _CMP_LT_OQ))))));

}

// LRImage: cheap up scaled. HRImage: RAISR refined. outImage: output buffer in 8u.
// rows: rows of LRImage/HRImage. startRow: seg start row. blendingZone: zone to run blending.
// cols: stride for buffers in DT type.
// outImageCols: stride for outImage buffer
static void CTCountOfBitsChangedSegment_AVX256_32f(float *LRImage, float *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols)
{
    int rowStartOffset = blendingZone.first - startRow;
    int rowEndOffset = blendingZone.second - startRow;

    const __m256 zero_ps = _mm256_setzero_ps();
    const __m256 one_ps = _mm256_set1_ps(1.0);
    const int cmp_le = _CMP_LT_OQ;
    const __m256i one_epi32 = _mm256_set1_epi32(1);

    for (auto r = rowStartOffset; r < rowEndOffset; r++)
    {
        const int c_limit = (cols - CTmargin);
        int c_limit_avx = c_limit - (c_limit%8)+1;
        for (auto c = CTmargin; c < c_limit_avx; c+=8)
        {
            __m256i hammingDistance_epi32 = _mm256_setzero_si256();

            __m256 center_LR_ps = _mm256_loadu_ps( &LRImage[(r) * cols + c]);
            __m256 n1_LR_ps = _mm256_loadu_ps( &LRImage[(r-1) * cols + (c-1)]);
            __m256 n2_LR_ps = _mm256_loadu_ps( &LRImage[(r-1) * cols + (c)]);
            __m256 n3_LR_ps = _mm256_loadu_ps( &LRImage[(r-1) * cols + (c+1)]);
            __m256 n4_LR_ps = _mm256_loadu_ps( &LRImage[(r) * cols + (c-1)]);
            __m256 n5_LR_ps = _mm256_loadu_ps( &LRImage[(r) * cols + (c+1)]);
            __m256 n6_LR_ps = _mm256_loadu_ps( &LRImage[(r+1) * cols + (c-1)]);
            __m256 n7_LR_ps = _mm256_loadu_ps( &LRImage[(r+1) * cols + (c)]);
            __m256 n8_LR_ps = _mm256_loadu_ps( &LRImage[(r+1) * cols + (c+1)]);

            __m256 center_HR_ps = _mm256_loadu_ps( &HRImage[(r) * cols + c]);
            __m256 n1_HR_ps = _mm256_loadu_ps( &HRImage[(r-1) * cols + (c-1)]);
            __m256 n2_HR_ps = _mm256_loadu_ps( &HRImage[(r-1) * cols + (c)]);
            __m256 n3_HR_ps = _mm256_loadu_ps( &HRImage[(r-1) * cols + (c+1)]);
            __m256 n4_HR_ps = _mm256_loadu_ps( &HRImage[(r) * cols + (c-1)]);
            __m256 n5_HR_ps = _mm256_loadu_ps( &HRImage[(r) * cols + (c+1)]);
            __m256 n6_HR_ps = _mm256_loadu_ps( &HRImage[(r+1) * cols + (c-1)]);
            __m256 n7_HR_ps = _mm256_loadu_ps( &HRImage[(r+1) * cols + (c)]);
            __m256 n8_HR_ps = _mm256_loadu_ps( &HRImage[(r+1) * cols + (c+1)]);

            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n1_LR_ps, center_LR_ps, n1_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n2_LR_ps, center_LR_ps, n2_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n3_LR_ps, center_LR_ps, n3_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n4_LR_ps, center_LR_ps, n4_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n5_LR_ps, center_LR_ps, n5_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n6_LR_ps, center_LR_ps, n6_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n7_LR_ps, center_LR_ps, n7_HR_ps, center_HR_ps);
            hammingDistance_epi32 = HAMMING_DISTANCE_EPI32(hammingDistance_epi32, n8_LR_ps, center_LR_ps, n8_HR_ps, center_HR_ps);

            __m256 weight_ps = _mm256_div_ps( _mm256_cvtepi32_ps(hammingDistance_epi32), _mm256_set1_ps((float) CTnumberofPixel) );
            __m256 weight2_ps = _mm256_sub_ps(one_ps, weight_ps);
            __m256 val_ps = _mm256_add_ps( _mm256_mul_ps( weight_ps, center_LR_ps),
                                            _mm256_mul_ps(weight2_ps, center_HR_ps));
            val_ps = _mm256_add_ps( val_ps, _mm256_set1_ps(0.5));
            __m256i val_epi32 = _mm256_cvtps_epi32(_mm256_floor_ps(val_ps)), val_epi16, val_epu8, val_epu16, perm_epu;
            int64_t val_epu8_64_t;
            if (gBitDepth == 8) {
                val_epi32 = _mm256_max_epi32(_mm256_min_epi32( val_epi32, _mm256_set1_epi32(gMax8bit)), _mm256_set1_epi32(gMin8bit));
                val_epi16 = _mm256_packs_epi32(val_epi32,val_epi32);
                val_epu8 = _mm256_packus_epi16(val_epi16, val_epi16);
                perm_epu = _mm256_permutevar8x32_epi32(val_epu8, _mm256_setr_epi32(0,4,0,4,0,4,0,4));
                val_epu8_64_t = (_mm_cvtsi128_si64(_mm256_extractf128_si256(perm_epu, 0)));
                memcpy((void *) &outImage[(startRow + r) * outImageCols + c], (void *) &val_epu8_64_t, 8);
            }
            else {
                val_epi32 = _mm256_max_epi32(_mm256_min_epi32( val_epi32, _mm256_set1_epi32(gMax16bit)), _mm256_set1_epi32(gMin16bit));
                val_epu16 = _mm256_packus_epi32(val_epi32,val_epi32);
                perm_epu = _mm256_permute4x64_epi64(val_epu16, 0x88);
                unsigned short *out = (unsigned short *)outImage;
                _mm_storeu_si128((__m128i *) &out[(startRow + r) * outImageCols / sizeof(unsigned short) + c], _mm256_extractf128_si256(perm_epu, 0));
            }
        }

        for (auto c = c_limit_avx; c < c_limit; c++) // handle edge, too small for SIMD
        {
            int hammingDistance = 0;

            // Census transform
            for (int i = -CTmargin; i <= CTmargin; i++)
            {
                for (int j = -CTmargin; j <= CTmargin; j++)
                {
                    if (unlikely(i == 0 && j == 0))
                        continue;
                    hammingDistance += std::abs((LRImage[(r + i) * cols + (c + j)] < LRImage[r * cols + c] ? 1 : 0) - (HRImage[(r + i) * cols + (c + j)] < HRImage[r * cols + c] ? 1 : 0));
                }
            }
            float weight = (float)hammingDistance / (float)CTnumberofPixel;
            float val = weight * LRImage[r * cols + c] + (1 - weight) * HRImage[r * cols + c];
            val += 0.5; // to round the value
            //convert 32f to 8bit/10bit
            if (gBitDepth == 8) {
                outImage[(startRow + r) * outImageCols + c] = (unsigned char)(val < gMin8bit ? gMin8bit : (val > gMax8bit ? gMax8bit : val));
            }
            else {
                unsigned short *out = (unsigned short *)outImage;
                out[(startRow + r) * outImageCols / sizeof(unsigned short) + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
            }
        }
    }
}

inline float sumitup_ps_256(__m256 acc)
{
    const __m128 r4 = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps(acc, 1));
    const __m128 r2 = _mm_add_ps(r4, _mm_movehl_ps(r4, r4));
    const __m128 r1 = _mm_add_ss(r2, _mm_movehdup_ps(r2));
    return _mm_cvtss_f32(r1);
}

inline __m256 shiftL_AVX256(__m256 r)
{
    return _mm256_permutevar8x32_ps(r, _mm256_set_epi32(0, 7, 6, 5, 4, 3, 2, 1));
}

inline __m256 shiftR_AVX256(__m256 r)
{
    return _mm256_permutevar8x32_ps(r, _mm256_set_epi32(6, 5, 4, 3, 2, 1, 0, 7));
}

inline __m256 GetGx_AVX256(__m256 r1, __m256 r3)
{
    return _mm256_sub_ps(r3, r1);
}

inline __m256 GetGy_AVX256(__m256 r2)
{
    return _mm256_sub_ps(shiftL_AVX256(r2), shiftR_AVX256(r2));
}

inline __m128 GetFirstHalf(__m256 n)
{
    return _mm256_extractf128_ps(n, 0);
}

inline __m128 GetLastHalf(__m256 n)
{
    return _mm256_extractf128_ps(n, 1);
}

template <int halfIndex>
inline __m256 SetFirstVal(__m256 n, __m128 halfWithValue) {
    __m128 newHalf = _mm_insert_ps(_mm256_extractf128_ps(n, 0), halfWithValue, halfIndex);
    return _mm256_insertf128_ps(n, newHalf, 0);
}

template <int halfIndex>
inline __m256 SetLastVal(__m256 n, __m128 halfWithValue) {
    __m128 newHalf = _mm_insert_ps(_mm256_extractf128_ps(n, 1), halfWithValue, halfIndex);
    return _mm256_insertf128_ps(n, newHalf, 1);
}

inline __m256 GetGy_AVX256Hi(__m256 xlo, __m256 xhi)
{
    // ideally we do some cross lane permute, but one doesnt seem to exist.  Our approach instead is to save the original values,
    // do our in-lane permutes, then insert additional values on the ends to achieve correct behavior
    __m128 xlohi = GetLastHalf(xlo);
    __m128 xlolo = GetFirstHalf(xlo);

    __m256 newloLeft = SetLastVal<0x30>(shiftL_AVX256(xhi), xlolo);
    __m256 newloRight = SetFirstVal<0xC0>(shiftR_AVX256(xhi), xlohi);
    __m256 ret = _mm256_sub_ps(newloLeft, newloRight);
    return ret;
}

inline __m256 GetGy_AVX256Lo(__m256 xlo, __m256 xhi)
{
    // ideally we do some cross lane permute, but one doesnt seem to exist.  Our approach instead is to save the original values,
    // do our in-lane permutes, then insert additional values on the ends to achieve correct behavior
    __m128 xhilo = GetFirstHalf(xhi);
    __m128 xhihi = GetLastHalf(xhi);
    __m256 newloLeft = SetLastVal <0x30>(shiftL_AVX256(xlo), xhilo);
    __m256 newloRight = SetFirstVal<0xC0>(shiftR_AVX256(xlo), xhihi);

    __m256 ret = _mm256_sub_ps(newloLeft, newloRight);
    return ret;
}

inline __m256 GetGTWG_AVX256(__m256 acc, __m256 a, __m256 w, __m256 b)
{
    return _mm256_fmadd_ps(_mm256_mul_ps(a, w), b, acc);
}

void inline computeGTWG_Segment_AVX256_32f(const float *img, const int nrows, const int ncols, const int r, const int col, float GTWG[3][16], int pix, float *buf1, float *buf2)
{
    // offset is the starting position(top left) of the block which centered by (r, c)
    int gtwgIdx = pix *2;
    int offset = (r - gLoopMargin) * ncols + col - gLoopMargin;
    const float *p1 = img + offset;

    __m256 gtwg0A1 = _mm256_setzero_ps(), gtwg0A2 = _mm256_setzero_ps();
    __m256 gtwg0B1 = _mm256_setzero_ps(), gtwg0B2 = _mm256_setzero_ps();
    __m256 gtwg1A1 = _mm256_setzero_ps(), gtwg1A2 = _mm256_setzero_ps();
    __m256 gtwg1B1 = _mm256_setzero_ps(), gtwg1B2 = _mm256_setzero_ps();
    __m256 gtwg3A1 = _mm256_setzero_ps(), gtwg3A2 = _mm256_setzero_ps();
    __m256 gtwg3B1 = _mm256_setzero_ps(), gtwg3B2 = _mm256_setzero_ps();

    // load 2 rows
    __m256 a1 = _mm256_loadu_ps(p1);
    __m256 a2 = _mm256_loadu_ps(p1+8);
    p1 += ncols;
    __m256 b1 = _mm256_loadu_ps(p1);
    __m256 b2 = _mm256_loadu_ps(p1+8);
#pragma unroll
    for (int i = 0; i < gPatchSize; i++)
    {
        // process patchSize rows
        // load next row
        p1 += ncols;
        __m256 c1 = _mm256_loadu_ps(p1);
        __m256 c2 = _mm256_loadu_ps(p1+8);
        __m256 w1, w2;
       if(gBitDepth == 8) {
            w1 = _mm256_loadu_ps(gGaussian2D8bit[i]);
            w2 = _mm256_loadu_ps(gGaussian2D8bit[i]+8);
       } else if (gBitDepth == 10) {
            w1 = _mm256_loadu_ps(gGaussian2D10bit[i]);
            w2 = _mm256_loadu_ps(gGaussian2D10bit[i]+8);
       } else {
            w1 = _mm256_loadu_ps(gGaussian2D16bit[i]);
            w2 = _mm256_loadu_ps(gGaussian2D16bit[i]+8);
       }

        const __m256 gxi1 = GetGx_AVX256(a1, c1);
        const __m256 gxi2 = GetGx_AVX256(a2, c2);

        const __m256 gyi1 = GetGy_AVX256Lo(b1,b2);
        const __m256 gyi2 = GetGy_AVX256Hi(b1,b2);

        gtwg0A1 = GetGTWG_AVX256(gtwg0A1, gxi1, w1, gxi1);
        gtwg0A2 = GetGTWG_AVX256(gtwg0A2, gxi2, w2, gxi2);
        gtwg1A1 = GetGTWG_AVX256(gtwg1A1, gxi1, w1, gyi1);
        gtwg1A2 = GetGTWG_AVX256(gtwg1A2, gxi2, w2, gyi2);
        gtwg3A1 = GetGTWG_AVX256(gtwg3A1, gyi1, w1, gyi1);
        gtwg3A2 = GetGTWG_AVX256(gtwg3A2, gyi2, w2, gyi2);

        // Store last bit for shiftR and mask
        __m128 xlohi = GetLastHalf(w1);
        __m128 xhihi = GetLastHalf(w2);
        w1 = SetFirstVal<0xC0>(shiftR_AVX256(w1), xhihi);
        w2 = SetFirstVal<0xC0>(shiftR_AVX256(w2), xlohi);

        gtwg0B1 = GetGTWG_AVX256(gtwg0B1, gxi1, w1, gxi1);
        gtwg0B2 = GetGTWG_AVX256(gtwg0B2, gxi2, w2, gxi2);
        gtwg1B1 = GetGTWG_AVX256(gtwg1B1, gxi1, w1, gyi1);
        gtwg1B2 = GetGTWG_AVX256(gtwg1B2, gxi2, w2, gyi2);
        gtwg3B1 = GetGTWG_AVX256(gtwg3B1, gyi1, w1, gyi1);
        gtwg3B2 = GetGTWG_AVX256(gtwg3B2, gyi2, w2, gyi2);

        // skip one, store next 11 bits.  The two masks are 0xfe, 0x0f
        int lastbit = 0x80000000;
        _mm256_maskstore_ps(buf1 + gPatchSize * i - 1, _mm256_setr_epi32(0, lastbit, lastbit, lastbit, lastbit, lastbit, lastbit, lastbit), b1);
        _mm256_maskstore_ps(buf1 + gPatchSize * i - 1 + 8, _mm256_setr_epi32(lastbit, lastbit, lastbit, lastbit, 0,0,0,0), b2);
        // skip two, store next 11 bits.  The two masks are 0xfc, 0x1f
        _mm256_maskstore_ps(buf2 + gPatchSize * i - 2, _mm256_setr_epi32(0,0,lastbit,lastbit,lastbit,lastbit,lastbit,lastbit), b1);
        _mm256_maskstore_ps(buf2 + gPatchSize * i - 2 + 8, _mm256_setr_epi32(lastbit,lastbit,lastbit,lastbit,lastbit,0,0,0), b2);
        a1 = b1;
        a2 = b2;
        b1 = c1;
        b2 = c2;
    }

    GTWG[0][gtwgIdx] = sumitup_ps_256(_mm256_add_ps(gtwg0A1, gtwg0A2));
    GTWG[1][gtwgIdx] = sumitup_ps_256(_mm256_add_ps(gtwg1A1, gtwg1A2));
    GTWG[2][gtwgIdx] = sumitup_ps_256(_mm256_add_ps(gtwg3A1, gtwg3A2));

    GTWG[0][gtwgIdx+1] = sumitup_ps_256(_mm256_add_ps(gtwg0B1, gtwg0B2));
    GTWG[1][gtwgIdx+1] = sumitup_ps_256(_mm256_add_ps(gtwg1B1, gtwg1B2));
    GTWG[2][gtwgIdx+1] = sumitup_ps_256(_mm256_add_ps(gtwg3B1, gtwg3B2));

    return;
}


// AVX2 version: for now, gPatchSize must be <= 16 because we can work with up to 16 float32s in two AVX256 registers.
float inline DotProdPatch_AVX256_32f(const float *buf, const float *filter)
{
    __m256 a1_ps = _mm256_load_ps(buf);
    __m256 b1_ps = _mm256_load_ps(filter);
    __m256 a2_ps = _mm256_load_ps(buf+8);
    __m256 b2_ps = _mm256_load_ps(filter+8);

    __m256 sum1 = _mm256_mul_ps(a1_ps, b1_ps);
    __m256 sum2 = _mm256_mul_ps(a2_ps, b2_ps);

#pragma unroll
    for (int i = 1; i < 8; i++)
    {
        a1_ps = _mm256_load_ps(buf + i * 16);
        a2_ps = _mm256_load_ps(buf + i * 16 + 8);
        b1_ps = _mm256_load_ps(filter + i * 16);
        b2_ps = _mm256_load_ps(filter + i * 16 + 8);

        // compute dot prod using fmadd
        sum1 = _mm256_fmadd_ps(a1_ps, b1_ps, sum1);
        sum2 = _mm256_fmadd_ps(a2_ps, b2_ps, sum2);
    }

    // sumitup adds all 16 float values in sum(zmm) and returns a single float value
    return  sumitup_ps_256(_mm256_add_ps(sum1, sum2));
}

inline __m256 atan2Approximation_AVX256_32f(__m256 y_ps, __m256 x_ps)
{
    const float ONEQTR_PI = M_PI / 4.0;
    const float THRQTR_PI = 3.0 * M_PI / 4.0;
    const __m256 zero_ps = _mm256_set1_ps(0.0);
    const __m256 oneqtr_pi_ps = _mm256_set1_ps(ONEQTR_PI);
    const __m256 thrqtr_pi_ps = _mm256_set1_ps(THRQTR_PI);

    __m256 abs_y_ps = _mm256_add_ps( _mm256_andnot_ps( _mm256_set1_ps(-0.0f), y_ps),
                                     _mm256_set1_ps(1e-10f));

    __m256 r_cond1_ps = _mm256_div_ps( _mm256_add_ps(x_ps, abs_y_ps), _mm256_sub_ps(abs_y_ps, x_ps));
    __m256 r_cond2_ps = _mm256_div_ps( _mm256_sub_ps(x_ps, abs_y_ps), _mm256_add_ps(x_ps, abs_y_ps));
    __m256 r_cmp_ps =  _mm256_cmp_ps(x_ps, zero_ps, _CMP_LT_OQ);
    __m256 r_ps = _mm256_blendv_ps( r_cond2_ps, r_cond1_ps, r_cmp_ps);
    __m256 angle_ps = _mm256_blendv_ps( oneqtr_pi_ps, thrqtr_pi_ps, r_cmp_ps );

    angle_ps = _mm256_fmadd_ps(_mm256_fmadd_ps(_mm256_mul_ps(_mm256_set1_ps(0.1963f), r_ps),
                                                                                    r_ps, _mm256_set1_ps(-0.9817f)),
                                                                                    r_ps, angle_ps);

    __m256 neg_angle_ps = _mm256_mul_ps(_mm256_set1_ps(-1), angle_ps);
    return _mm256_blendv_ps( angle_ps, neg_angle_ps, _mm256_cmp_ps(y_ps, zero_ps, _CMP_LT_OQ));
}

void inline GetHashValue_AVX256_32f_8Elements(float GTWG[3][16], int passIdx, int32_t *idx) {
    const float one = 1.0;
    const float two = 2.0;
    const float four = 4.0;
    const float pi = PI;
    const float near_zero = 0.00000000000000001;

    const __m256 zero_ps = _mm256_setzero_ps();
    const __m256i zero_epi32 = _mm256_setzero_si256();
    const __m256i one_epi32 = _mm256_set1_epi32(1);
    const __m256i two_epi32 = _mm256_set1_epi32(2);

    const int cmp_le = _CMP_LE_OQ;
    const int cmp_gt = _CMP_GT_OQ;

    __m256 m_a_ps = _mm256_load_ps( GTWG[0]);
    __m256 m_b_ps = _mm256_load_ps( GTWG[1]);
    __m256 m_d_ps = _mm256_load_ps( GTWG[2]);

    __m256 T_ps = _mm256_add_ps(m_a_ps, m_d_ps);
    __m256 D_ps = _mm256_sub_ps( _mm256_mul_ps( m_a_ps, m_d_ps),
                                _mm256_mul_ps( m_b_ps, m_b_ps));

    __m256 sqr_ps = _mm256_rcp_ps( _mm256_rsqrt_ps( _mm256_sub_ps( _mm256_div_ps ( _mm256_mul_ps(T_ps, T_ps),
                                                    _mm256_broadcast_ss(&four)), D_ps)));

    __m256 half_T_ps = _mm256_div_ps ( T_ps, _mm256_broadcast_ss(&two) );
    __m256 L1_ps = _mm256_add_ps( half_T_ps, sqr_ps);
    __m256 L2_ps = _mm256_sub_ps( half_T_ps, sqr_ps);

    __m256 angle_ps = zero_ps;

    __m256 blend_ps = _mm256_blendv_ps( _mm256_broadcast_ss(&one), _mm256_sub_ps(L1_ps, m_d_ps),
                                    _mm256_cmp_ps(m_b_ps, zero_ps, _CMP_NEQ_OQ) );

#ifdef USE_ATAN2_APPROX
    angle_ps = atan2Approximation_AVX256_32f( m_b_ps, blend_ps);
#else
    angle_ps = _mm256_atan2_ps( m_b_ps, blend_ps);
#endif

    angle_ps = _mm256_add_ps ( angle_ps, _mm256_blendv_ps( zero_ps, _mm256_broadcast_ss(&pi),
                                    _mm256_cmp_ps(angle_ps, zero_ps, _CMP_LT_OQ) ) );

    __m256 sqrtL1_ps = _mm256_rcp_ps( _mm256_rsqrt_ps( L1_ps ));
    __m256 sqrtL2_ps = _mm256_rcp_ps( _mm256_rsqrt_ps( L2_ps ));
    __m256 coherence_ps = _mm256_div_ps( _mm256_sub_ps( sqrtL1_ps, sqrtL2_ps ),
                                        _mm256_add_ps( _mm256_add_ps(sqrtL1_ps, sqrtL2_ps), _mm256_broadcast_ss(&near_zero) ) );
    __m256 strength_ps = L1_ps;

    __m256i angleIdx_epi32 = _mm256_cvtps_epi32( _mm256_floor_ps(_mm256_mul_ps (angle_ps, _mm256_broadcast_ss(&gQAngle))));

    angleIdx_epi32 = _mm256_min_epi32( _mm256_sub_epi32( _mm256_set1_epi32(gQuantizationAngle), _mm256_set1_epi32(1)),
                                       _mm256_max_epi32(angleIdx_epi32, zero_epi32 ) );

   // AFAIK, today QStr & QCoh are vectors of size 2.  I think searchsorted can return an index of 0,1, or 2
    float *gQStr_data, *gQCoh_data;
    if (passIdx == 0) gQStr_data = gQStr.data(); else gQStr_data = gQStr2.data();
    if (passIdx == 0) gQCoh_data = gQCoh.data(); else gQCoh_data = gQCoh2.data();
    __m256 gQStr1_ps = _mm256_broadcast_ss(gQStr_data);
    __m256 gQStr2_ps = _mm256_broadcast_ss(gQStr_data + 1);
    __m256 gQCoh1_ps = _mm256_broadcast_ss(gQCoh_data);
    __m256 gQCoh2_ps = _mm256_broadcast_ss(gQCoh_data + 1);

   __m256i strengthIdx_epi32 = _mm256_sub_epi32(two_epi32,
                                    _mm256_add_epi32(
                                    _mm256_and_si256(one_epi32, _mm256_castps_si256( _mm256_cmp_ps(strength_ps, gQStr1_ps, cmp_le))),
                                    _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(strength_ps, gQStr2_ps, cmp_le)))));
    __m256i coherenceIdx_epi32 = _mm256_sub_epi32(two_epi32,
                                    _mm256_add_epi32(
                                    _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(coherence_ps, gQCoh1_ps, cmp_le))),
                                    _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(coherence_ps, gQCoh2_ps, cmp_le)))));

    const __m256i gQuantizationCoherence_epi32 = _mm256_set1_epi32(gQuantizationCoherence);
    __m256i idx_epi32 = _mm256_mullo_epi32(gQuantizationCoherence_epi32,
                                            _mm256_mullo_epi32(angleIdx_epi32, _mm256_set1_epi32(gQuantizationStrength)));
    idx_epi32 = _mm256_add_epi32(coherenceIdx_epi32,
                                _mm256_add_epi32(idx_epi32, _mm256_mullo_epi32(strengthIdx_epi32, gQuantizationCoherence_epi32)));
    _mm256_storeu_si256((__m256i *)idx, idx_epi32);
}

