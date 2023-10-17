/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "Raisr_globals.h"
#include "Raisr_AVX512FP16.h"
#include <immintrin.h>
#include <popcntintrin.h>
#include <cmath>
#include<string.h>

inline __m512i floor_ph_512(__m512h val_ph)
{ 
    __m512i ret_epi16;
    ret_epi16 = _mm512_cvt_roundph_epi16(val_ph, _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC);
    return ret_epi16;
}

inline __m128i floor_ph_128(__m128h val_ph)
{ 
    __m128i ret_epi16;
    ret_epi16 = _mm512_castsi512_si128(_mm512_cvt_roundph_epi16(_mm512_castph128_ph512(val_ph), _MM_FROUND_TO_NEG_INF | _MM_FROUND_NO_EXC));
    return ret_epi16;
}

inline void load3x3_ph(_Float16 *img, unsigned int width, unsigned int height, unsigned int stride, __m128h *out_8neighbors_ph, __m128h *out_center_ph)
{
    int index = (height - 1) * stride + (width - 1);
    // load 3x3 grid for lr image, including center pixel plus 8 neighbors
    *out_8neighbors_ph = _mm_permutex2var_ph( _mm_loadu_ph(img + index - 0), // get pixels 0,1,2 from a, the rest from b
                        _mm_setr_epi16(0x0|0x8,0x1|0x8,0x2|0x8,0x3,0x4,0x5,0x6,0x7), *out_8neighbors_ph);
    index += stride;
    _Float16 center2x[2] = {*(img + index + 1), *(img + index +1)};
    _Float16 neighbors[2] = {*(img + index), *(img + index + 2)};
    *out_8neighbors_ph = _mm_permutex2var_ph( _mm_loadu_ph(neighbors - 3), // get pixels 3,4 from a, the rest from b
                        _mm_setr_epi16(0x0,0x1,0x2,0x3|0x8,0x4|0x8,0x5,0x6,0x7), *out_8neighbors_ph);
    index += stride;
    *out_8neighbors_ph = _mm_permutex2var_ph( _mm_loadu_ph(img + index - 5), // get pixels 5,6,7 from a, the rest from b
                        _mm_setr_epi16(0x0,0x1,0x2,0x3,0x4,0x5|0x8,0x6|0x8,0x7|0x8), *out_8neighbors_ph);

    *out_center_ph = _mm_castps_ph(_mm_broadcast_ss((float *) center2x)); // dont see a broadcast_sh
}

inline __mmask8 compare3x3_ph(__m128h a, __m128h b)
{
    return _mm_cmp_ph_mask(a, b, _CMP_LT_OS);
}

int CTRandomness_AVX512FP16_16f(_Float16 *inYUpscaled16f, int cols, int r, int c, int pix) {
    int census_count = 0;

    __m128h row_ph=_mm_setzero_ph(), center_ph=_mm_setzero_ph();

    load3x3_ph(inYUpscaled16f, c + pix, r, cols, &row_ph, &center_ph);

    // compare if neighbors < centerpixel, toggle bit in mask if true
    __mmask8 cmp_m8 = compare3x3_ph(row_ph, center_ph);

    // count # of bits in mask
    census_count += _mm_popcnt_u32(cmp_m8);

    return census_count;
}

inline void sumitup2lane_AVX512FP16_16f(__m512h acc, _Float16 *out_lower, _Float16 *out_upper) {
    __m512h r8_2 = _mm512_add_ph( acc, _mm512_castsi512_ph(_mm512_permutex_epi64( _mm512_castph_si512(acc), 14 ))); // move 2 and 3 to 0 and 1 positions
    __m512h r4_2 = _mm512_add_ph( r8_2, _mm512_castpd_ph(_mm512_movedup_pd( _mm512_castph_pd(r8_2) )));
    __m512h r2_2 = _mm512_add_ph( r4_2, _mm512_castps_ph(_mm512_movehdup_ps( _mm512_castph_ps(r4_2) )));
    __m512h r1_2 = _mm512_add_ph( r2_2, _mm512_castsi512_ph(_mm512_bsrli_epi128(_mm512_castph_si512(r2_2), 2)));
    r1_2 = _mm512_castsi512_ph(_mm512_permutex_epi64( _mm512_castph_si512(r1_2), 5 )); // move 1 to 0 and 1 positions
    (*out_lower) = _mm_cvtsh_h(_mm512_castph512_ph128(r1_2));
    (*out_upper) = _mm_cvtsh_h(_mm_castps_ph(_mm512_extractf32x4_ps(_mm512_castph_ps(r1_2), 2)));
}

inline _Float16 sumituphalf_AVX512FP16_16f(__m256h acc)
{
    // donts see extract instructions for ph, so we cast and use the ps version
    const __m128h r8 = _mm_add_ph(_mm256_castph256_ph128(acc),
                                    _mm_castps_ph(_mm256_extractf32x4_ps(_mm256_castph_ps(acc), 1)));
    const __m128h r4 = _mm_add_ph(r8,
                                _mm_castps_ph(_mm_movehl_ps( _mm_castph_ps(r8), _mm_castph_ps(r8))));
    const __m128h r2 = _mm_add_ph(r4,
                                _mm_castps_ph(_mm_movehdup_ps( _mm_castph_ps(r4))));
    // cant spot fp16 move/shift instr to add final two entries, so doing it in c
    float sum_f = (_mm_cvtss_f32( _mm_castph_ps(r2)));
    //_Float16 *sum = (_Float16*) &sum_f;
    _Float16 sum[2] = {0};
    memcpy(sum,&sum_f,sizeof(float));
    return sum[0] + sum[1];
/*
 * not any faster...
    const __m128h r4 = _mm_add_ph(r8,
                                    _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(r8), 8)));
    const __m128h r2 = _mm_add_ph(r4,
                                    _mm_castps_ph(_mm_movehdup_ps(_mm_castph_ps(r4))));
    const __m128h r1 = _mm_add_ph(r2,
                                    _mm_castsi128_ph(_mm_bsrli_si128(_mm_castph_si128(r2), 2)));
    return _mm_cvtsh_h(r1);
*/
}

inline _Float16 sumitup_AVX512FP16_16f(__m512h acc)
{
    const __m256h r16 = _mm256_add_ph(_mm512_castph512_ph256(acc),
                                        _mm256_castps_ph(_mm512_extractf32x8_ps(_mm512_castph_ps(acc), 1)));
    return sumituphalf_AVX512FP16_16f(r16);
}

inline __m512h shiftL_AVX512FP16(__m512h r)
{
    return _mm512_permutexvar_ph(_mm512_set_epi16(  16, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,
                                                    0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1 ), r);
}

inline __m512h shiftR_AVX512FP16(__m512h r)
{
    return _mm512_permutexvar_ph(_mm512_set_epi16(  30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17,16,31,
                                                    14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15), r);
}

inline __m512h GetGx_AVX512FP16(__m512h r1, __m512h r3)
{
    return _mm512_sub_ph(r3, r1);
}

inline __m512h GetGy_AVX512FP16(__m512h r2)
{
    return _mm512_sub_ph(shiftL_AVX512FP16(r2), shiftR_AVX512FP16(r2));
}

inline __m512h GetGTWG_AVX512FP16(__m512h acc, __m512h a, __m512h w, __m512h b)
{
    return _mm512_fmadd_ph(_mm512_mul_ph(a, w), b, acc);
}

void computeGTWG_Segment_AVX512FP16_16f(const _Float16 *img, const int nrows, const int ncols, const int r, const int col, _Float16 GTWG[3][32], int pix, _Float16 *buf1, _Float16 *buf2, _Float16 *buf3, _Float16 *buf4)
{
    // pix is the # of time calling computeGTWG_Segment. Because we compute 4 pixels in one function call, if unrollSizePatchBased == 8, then this function is called twice. First with pix==0 then pix==1. This is to help provide an index of where within GTWG** to store output data.
    // offset is the starting position(top left) of the block which centered by (r, c)
    int gtwgIdx = pix * 4;
    int offset = (r - gLoopMargin) * ncols + col - gLoopMargin;
    const _Float16 *p1 = img + offset;
    float normal = 0.0;
    if (gBitDepth == 8)
        normal = NF_8;
    else if (gBitDepth == 10)
        normal = NF_10;
    else
        normal = NF_16;

    __m512h gtwg0A = _mm512_setzero_ph(), gtwg1A = _mm512_setzero_ph(), gtwg3A = _mm512_setzero_ph();
    __m512h gtwg0B = _mm512_setzero_ph(), gtwg1B = _mm512_setzero_ph(), gtwg3B = _mm512_setzero_ph();

    // load 2 rows
    __m512h a = _mm512_zextph256_ph512(_mm256_loadu_ph(p1));
    a = _mm512_castps_ph(_mm512_insertf32x8(_mm512_castph_ps(a), _mm256_castph_ps(_mm512_castph512_ph256(a)), 1));  // duplicate high & low to compute GTWG for 2 pixels
    p1 += ncols;
    __m512h b = _mm512_zextph256_ph512(_mm256_loadu_ph(p1));
    b = _mm512_castps_ph(_mm512_insertf32x8(_mm512_castph_ps(b), _mm256_castph_ps(_mm512_castph512_ph256(b)), 1));  // duplicate high & low to compute GTWG for 2 pixels
#pragma unroll
    for (unsigned int i = 0; i < gPatchSize; i++)
    {
        // process patchSize rows
        // load next row
        p1 += ncols;
        __m512h c = _mm512_zextph256_ph512(_mm256_loadu_ph(p1));
        c = _mm512_castps_ph(_mm512_insertf32x8(_mm512_castph_ps(c), _mm256_castph_ps(_mm512_castph512_ph256(c)), 1));  // duplicate high & low to compute GTWG for 2 pixels
        __m512h w = _mm512_loadu_ph(gGaussian2DOriginal_fp16_doubled_w1w3[i]); // pixels 1,3

        const __m512h gxi = GetGx_AVX512FP16(a, c);
        const __m512h gyi = GetGy_AVX512FP16(b);

        gtwg0A = GetGTWG_AVX512FP16(gtwg0A, gxi, w, gxi);
        gtwg1A = GetGTWG_AVX512FP16(gtwg1A, gxi, w, gyi);
        gtwg3A = GetGTWG_AVX512FP16(gtwg3A, gyi, w, gyi);

        w = shiftR_AVX512FP16(w); // pixels 2,4

        gtwg0B = GetGTWG_AVX512FP16(gtwg0B, gxi, w, gxi);
        gtwg1B = GetGTWG_AVX512FP16(gtwg1B, gxi, w, gyi);
        gtwg3B = GetGTWG_AVX512FP16(gtwg3B, gyi, w, gyi);

        _mm256_mask_storeu_epi16(buf1 + gPatchSize * i - 1, 0x0ffe,_mm256_castph_si256(_mm512_castph512_ph256(b)));
        _mm256_mask_storeu_epi16(buf2 + gPatchSize * i - 2, 0x1ffc,_mm256_castph_si256(_mm512_castph512_ph256(b)));
        _mm256_mask_storeu_epi16(buf3 + gPatchSize * i - 3, 0x3ff8,_mm256_castph_si256(_mm512_castph512_ph256(b)));
        _mm256_mask_storeu_epi16(buf4 + gPatchSize * i - 4, 0x7ff0,_mm256_castph_si256(_mm512_castph512_ph256(b)));

        a = b;
        b = c;
    }
    // gtwg0 for pixels 0,2
    int gtwgIdxA = gtwgIdx+0;
    int gtwgIdxB = gtwgIdx+2;
    sumitup2lane_AVX512FP16_16f(gtwg0A, &GTWG[0][gtwgIdxA], &GTWG[0][gtwgIdxB]);
    GTWG[0][gtwgIdxA] *= normal;
    GTWG[0][gtwgIdxB] *= normal;
    // gtwg1 for pixels 0,2
    sumitup2lane_AVX512FP16_16f(gtwg1A, &GTWG[1][gtwgIdxA], &GTWG[1][gtwgIdxB]);
    GTWG[1][gtwgIdxA] *= normal;
    GTWG[1][gtwgIdxB] *= normal;
    // gtwg3 for pixels 0,2
    sumitup2lane_AVX512FP16_16f(gtwg3A, &GTWG[2][gtwgIdxA], &GTWG[2][gtwgIdxB]);
    GTWG[2][gtwgIdxA] *= normal;
    GTWG[2][gtwgIdxB] *= normal;

    gtwgIdxA = gtwgIdx+1;
    gtwgIdxB = gtwgIdx+3;
    // gtwg0 for pixels 1,3
    sumitup2lane_AVX512FP16_16f(gtwg0B, &GTWG[0][gtwgIdxA], &GTWG[0][gtwgIdxB]);
    GTWG[0][gtwgIdxA] *= normal;
    GTWG[0][gtwgIdxB] *= normal;
    // gtwg1 for pixels 1,3
    sumitup2lane_AVX512FP16_16f(gtwg1B, &GTWG[1][gtwgIdxA], &GTWG[1][gtwgIdxB]);
    GTWG[1][gtwgIdxA] *= normal;
    GTWG[1][gtwgIdxB] *= normal;
    // gtwg3 for pixels 1,3
    sumitup2lane_AVX512FP16_16f(gtwg3B, &GTWG[2][gtwgIdxA], &GTWG[2][gtwgIdxB]);
    GTWG[2][gtwgIdxA] *= normal;
    GTWG[2][gtwgIdxB] *= normal;

    return;
}


_Float16 DotProdPatch_AVX512FP16_16f(const _Float16 *buf, const _Float16 *filter)
{
    __m512h a_ph = _mm512_load_ph(buf);
    __m512h b_ph = _mm512_loadu_ph(filter);
    __m512h sum_ph = _mm512_mul_ph(a_ph, b_ph);
#pragma unroll
    for (int i = 1; i < 4; i++)
    {
        a_ph = _mm512_load_ph(buf + i * 32);
        b_ph = _mm512_loadu_ph(filter + i * 32);
        // compute dot prod using fmadd
        sum_ph = _mm512_fmadd_ph(a_ph, b_ph, sum_ph);
    }
    // sumitup adds all 32 float16 values in sum(zmm) and returns a single float16 value
    return sumitup_AVX512FP16_16f(sum_ph);
}

inline __m512i HAMMING_DISTANCE (__m512i hammDist, __m512h neigh_LR, __m512h center_LR, __m512h neigh_HR, __m512h center_HR)
{

    __mmask32 maskLR =  _mm512_cmp_ph_mask(neigh_LR,center_LR, _CMP_LT_OQ);
    __mmask32 maskHR =  _mm512_cmp_ph_mask(neigh_HR,center_HR, _CMP_LT_OQ);
    __m512i one_epi16 = _mm512_set1_epi16(1);
    __m512i zero_epi16 = _mm512_set1_epi16(0);

    return _mm512_add_epi16( hammDist,
                               _mm512_abs_epi16(_mm512_sub_epi16(
                                               _mm512_mask_blend_epi16(maskLR, zero_epi16, one_epi16),
                                               _mm512_mask_blend_epi16(maskHR, zero_epi16, one_epi16))));
}

void CTCountOfBitsChangedSegment_AVX512FP16_16f(_Float16 *LRImage, _Float16 *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols)
{
    int rowStartOffset = blendingZone.first - startRow;
    int rowEndOffset = blendingZone.second - startRow;

    const __m512h one_ph = _mm512_set1_ph(1.0);

    for (auto r = rowStartOffset; r < rowEndOffset; r++)
    {
        const int c_limit = (cols - CTmargin);
        int c_limit_avx = c_limit - (c_limit%32)+1;

        for (auto c = CTmargin; c < c_limit_avx; c+=32)
        {
            __m512i hammingDistance_epi16 = _mm512_setzero_si512();

            __m512h center_LR_ph = _mm512_loadu_ph( &LRImage[(r) * cols + c]);
            __m512h n1_LR_ph = _mm512_loadu_ph( &LRImage[(r-1) * cols + (c-1)]);
            __m512h n2_LR_ph = _mm512_loadu_ph( &LRImage[(r-1) * cols + (c)]);
            __m512h n3_LR_ph = _mm512_loadu_ph( &LRImage[(r-1) * cols + (c+1)]);
            __m512h n4_LR_ph = _mm512_loadu_ph( &LRImage[(r) * cols + (c-1)]);
            __m512h n5_LR_ph = _mm512_loadu_ph( &LRImage[(r) * cols + (c+1)]);
            __m512h n6_LR_ph = _mm512_loadu_ph( &LRImage[(r+1) * cols + (c-1)]);
            __m512h n7_LR_ph = _mm512_loadu_ph( &LRImage[(r+1) * cols + (c)]);
            __m512h n8_LR_ph = _mm512_loadu_ph( &LRImage[(r+1) * cols + (c+1)]);

            __m512h center_HR_ph = _mm512_loadu_ph( &HRImage[(r) * cols + c]);
            __m512h n1_HR_ph = _mm512_loadu_ph( &HRImage[(r-1) * cols + (c-1)]);
            __m512h n2_HR_ph = _mm512_loadu_ph( &HRImage[(r-1) * cols + (c)]);
            __m512h n3_HR_ph = _mm512_loadu_ph( &HRImage[(r-1) * cols + (c+1)]);
            __m512h n4_HR_ph = _mm512_loadu_ph( &HRImage[(r) * cols + (c-1)]);
            __m512h n5_HR_ph = _mm512_loadu_ph( &HRImage[(r) * cols + (c+1)]);
            __m512h n6_HR_ph = _mm512_loadu_ph( &HRImage[(r+1) * cols + (c-1)]);
            __m512h n7_HR_ph = _mm512_loadu_ph( &HRImage[(r+1) * cols + (c)]);
            __m512h n8_HR_ph = _mm512_loadu_ph( &HRImage[(r+1) * cols + (c+1)]);

            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n1_LR_ph, center_LR_ph, n1_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n2_LR_ph, center_LR_ph, n2_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n3_LR_ph, center_LR_ph, n3_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n4_LR_ph, center_LR_ph, n4_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n5_LR_ph, center_LR_ph, n5_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n6_LR_ph, center_LR_ph, n6_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n7_LR_ph, center_LR_ph, n7_HR_ph, center_HR_ph);
            hammingDistance_epi16 = HAMMING_DISTANCE(hammingDistance_epi16, n8_LR_ph, center_LR_ph, n8_HR_ph, center_HR_ph);

             __m512h weight_ph = _mm512_div_ph( _mm512_cvtepi16_ph(hammingDistance_epi16), _mm512_set1_ph((_Float16) CTnumberofPixel) );
            __m512h weight2_ph = _mm512_sub_ph(one_ph, weight_ph);

            __m512h val_ph = _mm512_add_ph( _mm512_mul_ph( weight_ph, center_LR_ph),
                                            _mm512_mul_ph(weight2_ph, center_HR_ph));
            val_ph = _mm512_add_ph( val_ph, _mm512_set1_ph(0.5));
            val_ph = _mm512_cvtepi16_ph(floor_ph_512(val_ph));

            // convert (float)val to (epu8/16)val
            __m512i val_epu16 = _mm512_cvtph_epu16(val_ph), val_epu8, perm_epu;
            if (gBitDepth == 8) {
                val_epu16 = _mm512_max_epu16(_mm512_min_epu16( val_epu16, _mm512_set1_epi16(gMax8bit)), _mm512_set1_epi16(gMin8bit));
                val_epu8 = _mm512_packus_epi16(val_epu16, val_epu16);
                perm_epu = _mm512_permutexvar_epi64(_mm512_setr_epi64(0,2,4,6,0,2,4,6), val_epu8);
                _mm256_storeu_si256((__m256i *) &outImage[(startRow + r) * outImageCols / sizeof(unsigned char) + c], _mm512_extracti64x4_epi64(perm_epu, 0));
            }
            else {
                val_epu16 = _mm512_max_epu16(_mm512_min_epu16( val_epu16, _mm512_set1_epi16(gMax16bit)), _mm512_set1_epi16(gMin16bit));
                unsigned short *out = (unsigned short *)outImage;
                _mm512_storeu_si512((__m512i *) &out[(startRow + r) * outImageCols / sizeof(unsigned short) + c], val_epu16 );
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

inline __m128h atan2Approximation_AVX512FP16_16h(__m128h y_ph, __m128h x_ph)
{
    const _Float16 ONEQTR_PI = M_PI / 4.0;
    const _Float16 THRQTR_PI = 3.0 * M_PI / 4.0;
    const __m128h zero_ph = _mm_set1_ph(0.0);
    const __m128h oneqtr_pi_ph = _mm_set1_ph(ONEQTR_PI);
    const __m128h thrqtr_pi_ph = _mm_set1_ph(THRQTR_PI);

    __m128h abs_y_ph = _mm_add_ph( _mm_abs_ph(y_ph), _mm_set1_ph(1e-10f));

    __m128h r_cond1_ph = _mm_div_ph( _mm_add_ph(x_ph, abs_y_ph), _mm_sub_ph(abs_y_ph, x_ph));
    __m128h r_cond2_ph = _mm_div_ph( _mm_sub_ph(x_ph, abs_y_ph), _mm_add_ph(x_ph, abs_y_ph));
    __mmask8 r_cmp_m8 =  _mm_cmp_ph_mask(x_ph, zero_ph, _CMP_LT_OQ);
    __m128h r_ph = _mm_mask_blend_ph( r_cmp_m8, r_cond2_ph, r_cond1_ph);
    __m128h angle_ph = _mm_mask_blend_ph( r_cmp_m8, oneqtr_pi_ph, thrqtr_pi_ph);

    angle_ph = _mm_fmadd_ph(_mm_fmadd_ph(_mm_mul_ph(_mm_set1_ph(0.1963f), r_ph),
                                                    r_ph, _mm_set1_ph(-0.9817f)),
                                                    r_ph, angle_ph);

    __m128h neg_angle_ph = _mm_mul_ph(_mm_set1_ph(-1), angle_ph);
    return _mm_mask_blend_ph( _mm_cmp_ph_mask(y_ph, zero_ph, _CMP_LT_OQ), angle_ph, neg_angle_ph );
}

// 8 elements, uses 128 bit regs
void GetHashValue_AVX512FP16_16h_8Elements(_Float16 GTWG[3][32], int passIdx, int32_t *idx) {
    const _Float16 one = 1.0;
    const _Float16 two = 2.0;
    const _Float16 four = 4.0;
    const _Float16 pi = PI;
    const _Float16 near_zero = 0.00000000000000001;

    const __m128h zero_ph = _mm_setzero_ph();
    const __m128h one_ph = _mm_set1_ph(1);
    const __m128i zero_epi16 = _mm_setzero_si128();
    const __m128i one_epi16 = _mm_set1_epi16(1);
    const __m128i two_epi16 = _mm_set1_epi16(2);

    const int cmp_le = _CMP_LE_OQ;
    const int cmp_gt = _CMP_GT_OQ;

    __m128h m_a_ph = _mm_load_ph(GTWG[0]);
    __m128h m_b_ph = _mm_load_ph(GTWG[1]);
    __m128h m_d_ph = _mm_load_ph(GTWG[2]);

    // GTWG values are very small. Let's multiply by 100 to make them bigger (scaled)
    __m128h c = _mm_set1_ph(100);
    m_a_ph = _mm_mul_ph(m_a_ph, c);
    m_b_ph = _mm_mul_ph(m_b_ph, c);
    m_d_ph = _mm_mul_ph(m_d_ph, c);

    __m128h T_ph = _mm_add_ph(m_a_ph, m_d_ph);
    __m128h D_ph = _mm_sub_ph( _mm_mul_ph( m_a_ph, m_d_ph),
                                _mm_mul_ph( m_b_ph, m_b_ph));

    // 11 bit accuracy:
    __m128h sqr_ph = _mm_rcp_ph( _mm_rsqrt_ph( _mm_sub_ph( _mm_div_ph ( _mm_mul_ph(T_ph, T_ph),
                                                           _mm_set1_ph(four)), D_ph)));

    __m128h half_T_ph = _mm_div_ph ( T_ph, _mm_set1_ph(two) );
    __m128h L1_ph = _mm_add_ph( half_T_ph, sqr_ph);
    __m128h L2_ph = _mm_sub_ph( half_T_ph, sqr_ph);

    __m128h angle_ph = zero_ph;

    __m128h blend_ph = _mm_mask_blend_ph( _mm_cmp_ph_mask(m_b_ph, zero_ph, _CMP_NEQ_OQ),
                                            one_ph, _mm_sub_ph(L1_ph, m_d_ph));

#ifdef USE_ATAN2_APPROX
    angle_ph = atan2Approximation_AVX512FP16_16h( m_b_ph, blend_ph);
#else
    angle_ph = _mm_atan2_ph( m_b_ph, blend_ph);
#endif

    angle_ph = _mm_add_ph ( angle_ph, _mm_mask_blend_ph( _mm_cmp_ph_mask(angle_ph, zero_ph, _CMP_LT_OQ), zero_ph, _mm_set1_ph(pi)));

    // fast sqrt with 11 bit accuracy
    __m128h sqrtL1_ph = _mm_rcp_ph( _mm_rsqrt_ph( L1_ph));
    __m128h sqrtL2_ph = _mm_rcp_ph( _mm_rsqrt_ph( L2_ph));

    __m128h coherence_ph = _mm_div_ph( _mm_sub_ph( sqrtL1_ph, sqrtL2_ph ),
                                        _mm_add_ph( _mm_add_ph(sqrtL1_ph, sqrtL2_ph), _mm_set1_ph(near_zero) ) );
    __m128h strength_ph = L1_ph;
    strength_ph = _mm_div_ph(strength_ph, c); // scale back because we scaled it up earlier

    __m128i angleIdx_epi16 = floor_ph_128( _mm_mul_ph (angle_ph, _mm_set1_ph(gQAngle)));
    __m128i quantAngle_lessone_epi16 = _mm_sub_epi16(_mm_set1_epi16(gQuantizationAngle), one_epi16);
    angleIdx_epi16 = _mm_min_epi16( _mm_sub_epi16(_mm_set1_epi16(gQuantizationAngle), _mm_set1_epi16(1)),
                    _mm_max_epi16(angleIdx_epi16, zero_epi16));

   // AFAIK, today QStr & QCoh are vectors of size 2.  I think searchsorted can return an index of 0,1, or 2
    _Float16 *gQStr_data, *gQCoh_data;
    if (passIdx == 0) gQStr_data = gQStr_fp16.data(); else gQStr_data = gQStr2_fp16.data();
    if (passIdx == 0) gQCoh_data = gQCoh_fp16.data(); else gQCoh_data = gQCoh2_fp16.data();
    __m128h gQStr1_ph = _mm_set1_ph(gQStr_data[0]);
    __m128h gQStr2_ph = _mm_set1_ph(gQStr_data[1]);
    __m128h gQCoh1_ph = _mm_set1_ph(gQCoh_data[0]);
    __m128h gQCoh2_ph = _mm_set1_ph(gQCoh_data[1]);

 __m128i strengthIdx_epi16 =
                                    _mm_add_epi16(
                                        _mm_mask_blend_epi16(_mm_cmp_ph_mask(gQStr1_ph, strength_ph, _MM_CMPINT_LE),zero_epi16, one_epi16),
                                        _mm_mask_blend_epi16(_mm_cmp_ph_mask(gQStr2_ph, strength_ph, _MM_CMPINT_LE),zero_epi16, one_epi16));
    __m128i coherenceIdx_epi16 =
                                    _mm_add_epi16(
                                        _mm_mask_blend_epi16(_mm_cmp_ph_mask(gQCoh1_ph, coherence_ph, _MM_CMPINT_LE),zero_epi16, one_epi16),
                                        _mm_mask_blend_epi16(_mm_cmp_ph_mask(gQCoh2_ph, coherence_ph, _MM_CMPINT_LE),zero_epi16, one_epi16));

   const __m128i gQuantizationCoherence_epi16 = _mm_set1_epi16(gQuantizationCoherence);
    __m128i idx_epi16 = _mm_mullo_epi16(gQuantizationCoherence_epi16,
                                            _mm_mullo_epi16( (angleIdx_epi16), _mm_set1_epi16(gQuantizationStrength)));
    idx_epi16 = _mm_add_epi16((coherenceIdx_epi16),
                                _mm_add_epi16(idx_epi16, _mm_mullo_epi16((strengthIdx_epi16), gQuantizationCoherence_epi16)));
    _mm256_storeu_si256((__m256i *)idx, _mm256_cvtepi16_epi32(idx_epi16));
}

inline __m512h atan2Approximation_AVX512FP16_16h_32Elements(__m512h y_ph, __m512h x_ph)
{
    const _Float16 ONEQTR_PI = M_PI / 4.0;
    const _Float16 THRQTR_PI = 3.0 * M_PI / 4.0;
    const __m512h zero_ph = _mm512_set1_ph(0.0);
    const __m512h oneqtr_pi_ph = _mm512_set1_ph(ONEQTR_PI);
    const __m512h thrqtr_pi_ph = _mm512_set1_ph(THRQTR_PI);

    __m512h abs_y_ph = _mm512_add_ph( _mm512_abs_ph(y_ph), _mm512_set1_ph(1e-10f));

    __m512h r_cond1_ph = _mm512_div_ph( _mm512_add_ph(x_ph, abs_y_ph), _mm512_sub_ph(abs_y_ph, x_ph));
    __m512h r_cond2_ph = _mm512_div_ph( _mm512_sub_ph(x_ph, abs_y_ph), _mm512_add_ph(x_ph, abs_y_ph));
    __mmask32 r_cmp_m8 =  _mm512_cmp_ph_mask(x_ph, zero_ph, _CMP_LT_OQ);
    __m512h r_ph = _mm512_mask_blend_ph( r_cmp_m8, r_cond2_ph, r_cond1_ph);
    __m512h angle_ph = _mm512_mask_blend_ph( r_cmp_m8, oneqtr_pi_ph, thrqtr_pi_ph);

    angle_ph = _mm512_fmadd_ph(_mm512_fmadd_ph(_mm512_mul_ph(_mm512_set1_ph(0.1963f), r_ph),
                                                    r_ph, _mm512_set1_ph(-0.9817f)),
                                                    r_ph, angle_ph);

    __m512h neg_angle_ph = _mm512_mul_ph(_mm512_set1_ph(-1), angle_ph);
    return _mm512_mask_blend_ph( _mm512_cmp_ph_mask(y_ph, zero_ph, _CMP_LT_OQ), angle_ph, neg_angle_ph );
}

void GetHashValue_AVX512FP16_16h_32Elements(_Float16 GTWG[3][32], int passIdx, int32_t *idx) {
    const _Float16 one = 1.0;
    const _Float16 two = 2.0;
    const _Float16 four = 4.0;
    const _Float16 pi = PI;
    const _Float16 near_zero = 0.00000000000000001;

    const __m512h zero_ph = _mm512_setzero_ph();
    const __m512h one_ph = _mm512_set1_ph(1);
    const __m512i zero_epi16 = _mm512_setzero_si512();
    const __m512i one_epi16 = _mm512_set1_epi16(1);
    const __m512i two_epi16 = _mm512_set1_epi16(2);

    const int cmp_le = _CMP_LE_OQ;
    const int cmp_gt = _CMP_GT_OQ;

    __m512h m_a_ph = _mm512_load_ph( GTWG[0]);
    __m512h m_b_ph = _mm512_load_ph( GTWG[1]); 
    __m512h m_d_ph = _mm512_load_ph( GTWG[2]); 

    // GTWG values are very small. Let's multiply by 100 to make them bigger (scaled)
    __m512h c = _mm512_set1_ph(100);
    m_a_ph = _mm512_mul_ph(m_a_ph, c);
    m_b_ph = _mm512_mul_ph(m_b_ph, c);
    m_d_ph = _mm512_mul_ph(m_d_ph, c);

    __m512h T_ph = _mm512_add_ph(m_a_ph, m_d_ph);

    __m512h D_ph = _mm512_sub_ph( _mm512_mul_ph( m_a_ph, m_d_ph),
                                _mm512_mul_ph( m_b_ph, m_b_ph));

    // 11 bit accuracy:
    __m512h sqr_ph = _mm512_rcp_ph( _mm512_rsqrt_ph( _mm512_sub_ph( _mm512_div_ph ( _mm512_mul_ph(T_ph, T_ph),
                                                           _mm512_set1_ph(four)), D_ph)));

    __m512h half_T_ph = _mm512_div_ph ( T_ph, _mm512_set1_ph(two) );
    __m512h L1_ph = _mm512_add_ph( half_T_ph, sqr_ph);
    __m512h L2_ph = _mm512_sub_ph( half_T_ph, sqr_ph);

    __m512h angle_ph = zero_ph;

    __m512h blend_ph = _mm512_mask_blend_ph( _mm512_cmp_ph_mask(m_b_ph, zero_ph, _CMP_NEQ_OQ),
                                            one_ph, _mm512_sub_ph(L1_ph, m_d_ph));

#ifdef USE_ATAN2_APPROX
    angle_ph = atan2Approximation_AVX512FP16_16h_32Elements( m_b_ph, blend_ph);
#else
    angle_ph = _mm512_atan2_ph( m_b_ph, blend_ph);
#endif

    angle_ph = _mm512_add_ph ( angle_ph, _mm512_mask_blend_ph( _mm512_cmp_ph_mask(angle_ph, zero_ph, _CMP_LT_OQ), zero_ph, _mm512_set1_ph(pi)));

    // fast sqrt with 11 bit accuracy
    __m512h sqrtL1_ph = _mm512_rcp_ph( _mm512_rsqrt_ph( L1_ph));
    __m512h sqrtL2_ph = _mm512_rcp_ph( _mm512_rsqrt_ph( L2_ph));

    __m512h coherence_ph = _mm512_div_ph( _mm512_sub_ph( sqrtL1_ph, sqrtL2_ph ),
                                        _mm512_add_ph( _mm512_add_ph(sqrtL1_ph, sqrtL2_ph), _mm512_set1_ph(near_zero) ) );
    __m512h strength_ph = L1_ph;
    strength_ph = _mm512_div_ph(strength_ph, c);

    __m512i angleIdx_epi16 = floor_ph_512(_mm512_mul_ph (angle_ph, _mm512_set1_ph(gQAngle)));

    __m512i quantAngle_lessone_epi16 = _mm512_sub_epi16(_mm512_set1_epi16(gQuantizationAngle), one_epi16);
    angleIdx_epi16 = _mm512_min_epi16(_mm512_sub_epi16(_mm512_set1_epi16(gQuantizationAngle),_mm512_set1_epi16(1)),
                    _mm512_max_epi16(angleIdx_epi16, zero_epi16));

   // AFAIK, today QStr & QCoh are vectors of size 2.  I think searchsorted can return an index of 0,1, or 2
    _Float16 *gQStr_data, *gQCoh_data;
    if (passIdx == 0) gQStr_data = gQStr_fp16.data(); else gQStr_data = gQStr2_fp16.data();
    if (passIdx == 0) gQCoh_data = gQCoh_fp16.data(); else gQCoh_data = gQCoh2_fp16.data();
    __m512h gQStr1_ph = _mm512_set1_ph(gQStr_data[0]);
    __m512h gQStr2_ph = _mm512_set1_ph(gQStr_data[1]);
    __m512h gQCoh1_ph = _mm512_set1_ph(gQCoh_data[0]);
    __m512h gQCoh2_ph = _mm512_set1_ph(gQCoh_data[1]);

    __m512i strengthIdx_epi16 =
                                    _mm512_add_epi16(
                                        _mm512_mask_blend_epi16(_mm512_cmp_ph_mask(gQStr1_ph, strength_ph, _MM_CMPINT_LE),zero_epi16, one_epi16),
                                        _mm512_mask_blend_epi16(_mm512_cmp_ph_mask(gQStr2_ph, strength_ph, _MM_CMPINT_LE),zero_epi16, one_epi16));
    __m512i coherenceIdx_epi16 =
                                    _mm512_add_epi16(
                                        _mm512_mask_blend_epi16(_mm512_cmp_ph_mask(gQCoh1_ph, coherence_ph, _MM_CMPINT_LE),zero_epi16, one_epi16),
                                        _mm512_mask_blend_epi16(_mm512_cmp_ph_mask(gQCoh2_ph, coherence_ph, _MM_CMPINT_LE),zero_epi16, one_epi16));

   const __m512i gQuantizationCoherence_epi16 = _mm512_set1_epi16(gQuantizationCoherence);
    __m512i idx_epi16 = _mm512_mullo_epi16(gQuantizationCoherence_epi16,
                                            _mm512_mullo_epi16( (angleIdx_epi16), _mm512_set1_epi16(gQuantizationStrength)));
    idx_epi16 = _mm512_add_epi16((coherenceIdx_epi16),
                                _mm512_add_epi16(idx_epi16, _mm512_mullo_epi16((strengthIdx_epi16), gQuantizationCoherence_epi16)));

    _mm512_storeu_si512((__m512i *)idx, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(idx_epi16)));
    _mm512_storeu_si512((__m512i *)(idx+16), _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(idx_epi16,1)));
}
