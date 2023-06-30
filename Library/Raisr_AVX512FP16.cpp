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

    __m128h row_ph, center_ph;

    load3x3_ph(inYUpscaled16f, c + pix, r, cols, &row_ph, &center_ph);

    // compare if neighbors < centerpixel, toggle bit in mask if true
    __mmask8 cmp_m8 = compare3x3_ph(row_ph, center_ph);

    // count # of bits in mask
    census_count += _mm_popcnt_u32(cmp_m8);

    return census_count;
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
    _Float16 *sum = (_Float16*) &sum_f;
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
    return _mm512_permutexvar_ph(_mm512_set_epi16(  0, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1,
                                                    16, 31, 30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17 ), r);
}

inline __m512h shiftR_AVX512FP16(__m512h r)
{
    return _mm512_permutexvar_ph(_mm512_set_epi16(  14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 15,
                                                    30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 31), r);
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

void computeGTWG_Segment_AVX512FP16_16f(const _Float16 *img, const int nrows, const int ncols, const int r, const int col, float GTWG[][4], _Float16 *buf1, _Float16 *buf2)
{
    // offset is the starting position(top left) of the block which centered by (r, c)
    int offset = (r - gLoopMargin) * ncols + col - gLoopMargin;
    const _Float16 *p1 = img + offset;
    float normal = 0.0;
    if (gBitDepth == 8)
        normal = NF_8;
    else if (gBitDepth == 10)
        normal = NF_10;
    else
        normal = NF_16;

    __m512h gtwg0 = _mm512_setzero_ph(), gtwg1 = _mm512_setzero_ph(), gtwg3 = _mm512_setzero_ph();

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
        __m512h w = _mm512_loadu_ph(gGaussian2DOriginal_fp16_doubled[i]);

        const __m512h gxi = GetGx_AVX512FP16(a, c);
        const __m512h gyi = GetGy_AVX512FP16(b);

        gtwg0 = GetGTWG_AVX512FP16(gtwg0, gxi, w, gxi);
        gtwg1 = GetGTWG_AVX512FP16(gtwg1, gxi, w, gyi);
        gtwg3 = GetGTWG_AVX512FP16(gtwg3, gyi, w, gyi);

        _mm256_mask_storeu_epi16(buf1 + gPatchSize * i - 1, 0x0ffe,_mm256_castph_si256(_mm512_castph512_ph256(b)));
        _mm256_mask_storeu_epi16(buf2 + gPatchSize * i - 2, 0x1ffc,_mm256_castph_si256(_mm512_castph512_ph256(b)));

        a = b;
        b = c;
    }

    GTWG[0][0] = sumituphalf_AVX512FP16_16f(_mm512_castph512_ph256(gtwg0)) * normal;
    GTWG[0][1] = sumituphalf_AVX512FP16_16f(_mm512_castph512_ph256(gtwg1)) * normal;
    GTWG[0][3] = sumituphalf_AVX512FP16_16f(_mm512_castph512_ph256(gtwg3)) * normal;
    GTWG[0][2] = GTWG[0][1];

    GTWG[1][0] = sumituphalf_AVX512FP16_16f(_mm256_castps_ph(_mm512_extractf32x8_ps(_mm512_castph_ps(gtwg0),1))) * normal;
    GTWG[1][1] = sumituphalf_AVX512FP16_16f(_mm256_castps_ph(_mm512_extractf32x8_ps(_mm512_castph_ps(gtwg1),1))) * normal;
    GTWG[1][3] = sumituphalf_AVX512FP16_16f(_mm256_castps_ph(_mm512_extractf32x8_ps(_mm512_castph_ps(gtwg3),1))) * normal;
    GTWG[1][2] = GTWG[1][1];

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

#ifndef USE_ATAN2_APPROX
            val_ph = _mm512_floor_ph(val_ph); // svml instruction. perhaps rounding would be better?
#endif

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

