/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * All rights reserved.
 */

#include "Raisr.h"
#include "Raisr_globals.h"
#include "Raisr_AVX256.h"
#include "Raisr_AVX512.h"
#include "Raisr_AVX256.cpp"
#include "Raisr_AVX512.cpp"
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <ipp.h>
#include <cstring>
#include <cmath>
#include <immintrin.h>
#include "ThreadPool.h"
#include "cpuid.h"
#include <chrono>

#ifndef WIN32
#include <unistd.h>
#endif

//#define MEASURE_TIME

/************************************************************
 *   helper functions
 ************************************************************/
#define ALIGNED_SIZE(size, align) (((size) + (align)-1) & ~((align)-1))

static bool is_machine_intel()
{
    bool ret = false;

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    __get_cpuid(0, &eax, &ebx, &ecx, &edx);

    char vendor_string[13];
    memcpy((vendor_string + 0), &ebx, 4);
    memcpy((vendor_string + 4), &edx, 4);
    memcpy((vendor_string + 8), &ecx, 4);
    vendor_string[12] = 0;

    if (!strcmp(vendor_string, "GenuineIntel"))
        ret = true;
    return ret;
}

template <typename DT>
static void createGaussianKernel(int n, double sigma, DT *result)
{
    DT sd_0_15 = 0.15;
    DT sd_0_35 = 0.35;
    DT sd_minus_0_125 = -0.5 * 0.25;

    DT sigmaX = DT(sigma);
    DT scale2X = sd_minus_0_125 / (sigmaX * sigmaX);

    int n2_ = (n - 1) / 2;
    std::vector<DT> values;
    DT sum = DT(0);
    for (int i = 0, x = 1 - n; i < n2_; i++, x += 2)
    {
        DT t = std::exp(DT(x * x) * scale2X);
        values.push_back(t);
        sum += t;
    }
    sum *= DT(2);
    sum += DT(1);
    if ((n & 1) == 0)
    {
        sum += DT(1);
    }

    DT mul1 = DT(1) / sum;

    DT sum2 = DT(0);
    for (int i = 0; i < n2_; i++)
    {
        DT t = values[i] * mul1;
        result[i] = t;
        result[n - 1 - i] = t;
        sum2 += t;
    }
    sum2 *= DT(2);
    result[n2_] = DT(1) * mul1;
    sum2 += result[n2_];
    if ((n & 1) == 0)
    {
        result[n2_ + 1] = result[n2_];
        sum2 += result[n2_];
    }
}

static RNLERRORTYPE VerifyTrainedData(std::string input, std::string file_type, std::string path)
{
    for (const char &c : input)
    {
        // c = '-' or '.' or [0:9]
        if (c < '-' || c > '9' || c == '/')
        {
            std::cout << "[RAISR ERROR] " << file_type << " corrupted: " << path << std::endl;
            return RNLErrorBadParameter;
        }
    }

    // multiple dots or dot in first place
    if (input.find_first_of('.') != input.find_last_of('.') || input.find_first_of('.') == 0)
    {
        std::cout << "[RAISR ERROR] " << file_type << " corrupted: " << path << std::endl;
        return RNLErrorBadParameter;
    }
    if (input.find_first_of('-') < 0xFFFF && input.find_first_of('.') < input.find_first_of('-'))
    {
        std::cout << "[RAISR ERROR] " << file_type << " corrupted: " << path << std::endl;
        return RNLErrorBadParameter;
    }
    return RNLErrorNone;
}

static RNLERRORTYPE ReadTrainedData(std::string hashtablePath, std::string QStrPath, std::string QCohPath, int pass)
{
    if (pass == 2)
    {
        hashtablePath += "_2";
        QStrPath += "_2";
        QCohPath += "_2";
    }
    auto &filterBuckets = (pass == 1) ? gFilterBuckets : gFilterBuckets2;
    auto &filterBuffer = (pass == 1) ? gFilterBuffer : gFilterBuffer2;
    auto &QStr = (pass == 1) ? gQStr : gQStr2;
    auto &QCoh = (pass == 1) ? gQCoh : gQCoh2;

    std::string line;
    // Read filter file
    std::ifstream filterFile(hashtablePath);
    if (!filterFile.is_open())
    {
        std::cout << "[RAISR ERROR] Unable to load model: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }
    std::getline(filterFile, line);
    std::istringstream filteriss(line);
    std::vector<std::string> filterTokens{std::istream_iterator<std::string>{filteriss},
                                          std::istream_iterator<std::string>{}};
    unsigned int hashkeySize = std::stoi(filterTokens[0].c_str());
    unsigned int pixelTypes = std::stoi(filterTokens[1].c_str());
    unsigned int rows = std::stoi(filterTokens[2].c_str());
    int aligned_rows = 16 * (int)((rows + 15) / 16);

    if (hashkeySize != gQuantizationAngle * gQuantizationStrength * gQuantizationCoherence)
    {
        std::cout << "[RAISR ERROR] HashTable format is not compatible in number of hash keys!\n";
        std::cout << hashkeySize << std::endl;
        return RNLErrorBadParameter;
    }

    if (pixelTypes != gRatio * gRatio)
    {
        std::cout << "[RAISR ERROR] HashTable format is not compatible in number of pixel types!\n";
        return RNLErrorBadParameter;
    }

    if (gPatchSize % 2 == 0 || rows != gPatchSize * gPatchSize)
    {
        std::cout << "[RAISR ERROR] HashTable format is not compatible in patch size!\n";
        return RNLErrorBadParameter;
    }

    // Allocate buffer to hold hashtable
    filterBuckets.clear();
    filterBuckets.resize(hashkeySize);
    for (auto i = 0; i < hashkeySize; i++)
    {
        filterBuckets[i].resize(pixelTypes);
    }

    // allocate contiguous memory to hold all filters
    filterBuffer = new float[aligned_rows * hashkeySize * pixelTypes + 16];
    uint64_t Aoffset = (uint64_t)filterBuffer & 0x3f;
    float *AFilters = &filterBuffer[16 - (int)(Aoffset / sizeof(float))];
    memset(AFilters, 0, sizeof(float) * aligned_rows * hashkeySize * pixelTypes);

    int num = 0;
    try
    {
        // Load hashtable
        while (getline(filterFile, line))
        {
            // too much lines in the file
            if (num > (hashkeySize * pixelTypes))
            {
                std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
                return RNLErrorBadParameter;
            }
            int k = 0;
            std::istringstream new_iss(line);
            std::vector<std::string> new_tokens{std::istream_iterator<std::string>{new_iss},
                                                std::istream_iterator<std::string>{}};
            float *currentfilter = &AFilters[num * aligned_rows];

            for (const auto &value : new_tokens)
            {
                currentfilter[k] = std::stod(value.c_str());
                k++;
            }

            // not enough value or too much values
            if (k < rows || k > rows)
            {
                std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
                return RNLErrorBadParameter;
            }
            filterBuckets[num / pixelTypes][num % pixelTypes] = currentfilter;
            num++;
        }
    }
    catch (const std::invalid_argument &)
    {
        std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }
    catch (const std::out_of_range &)
    {
        std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }

    // not enough lines in the file
    if (num < (hashkeySize * pixelTypes))
    {
        std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }
    filterFile.close();

    // Read QStr file
    std::ifstream Strfile(QStrPath);
    if (!Strfile.is_open())
    {
        std::cout << "[RAISR ERROR] Unable to load model: " << QStrPath << std::endl;
        return RNLErrorBadParameter;
    }
    QStr.clear();
    num = 0;
    try
    {
        while (Strfile >> line)
        {
            if (RNLErrorNone != VerifyTrainedData(line, "StrFile", QStrPath))
                return RNLErrorBadParameter;
            QStr.push_back(std::stod(line));
            num++;
        }
    }
    catch (const std::invalid_argument &)
    {
        std::cout << "[RAISR ERROR] StrFile corrupted: " << QStrPath << std::endl;
        return RNLErrorBadParameter;
    }
    catch (const std::out_of_range &)
    {
        std::cout << "[RAISR ERROR] StrFile corrupted: " << QStrPath << std::endl;
        return RNLErrorBadParameter;
    }
    if (num < (gQuantizationStrength - 1) || num > (gQuantizationStrength - 1))
    {
        std::cout << "[RAISR ERROR] StrFile corrupted: " << QStrPath << std::endl;
        return RNLErrorBadParameter;
    }
    Strfile.close();

    // Read QCoh file
    std::ifstream Cohfile(QCohPath);
    if (!Cohfile.is_open())
    {
        std::cout << "[RAISR ERROR] Unable to load model: " << QCohPath << std::endl;
        return RNLErrorBadParameter;
    }
    QCoh.clear();
    num = 0;
    try
    {
        while (Cohfile >> line)
        {
            if (RNLErrorNone != VerifyTrainedData(line, "CohFile", QCohPath))
                return RNLErrorBadParameter;
            QCoh.push_back(std::stod(line));
            num++;
        }
    }
    catch (const std::invalid_argument &)
    {
        std::cout << "[RAISR ERROR] CohFile corrupted: " << QCohPath << std::endl;
        return RNLErrorBadParameter;
    }
    catch (const std::out_of_range &)
    {
        std::cout << "[RAISR ERROR] CohFile corrupted: " << QCohPath << std::endl;
        return RNLErrorBadParameter;
    }
    if (num < (gQuantizationCoherence - 1) || num > (gQuantizationCoherence - 1))
    {
        std::cout << "[RAISR ERROR] CohFile corrupted: " << QCohPath << std::endl;
        return RNLErrorBadParameter;
    }
    Cohfile.close();

    return RNLErrorNone;
}

static IppStatus ippInit(IppiSize srcSize, Ipp32s srcStep, IppiSize dstSize, Ipp32s dstStep, IppiResizeSpec_32f **pSpec, Ipp8u **pBuffer)
{
    int specSize = 0, initSize = 0, bufSize = 0;
    Ipp8u *pInitBuf = 0;
    Ipp32u numChannels = 1;
    IppStatus status = ippStsNoErr;

    /* Spec and init buffer sizes */
    status = ippiResizeGetSize_8u(srcSize, dstSize, IPP_RESIZE_TYPE, 0, &specSize, &initSize);
    if (status != ippStsNoErr)
        return status;

    /* Memory allocation */
    pInitBuf = ippsMalloc_8u(initSize);
    *pSpec = (IppiResizeSpec_32f *)ippsMalloc_8u(specSize);
    if (pInitBuf == NULL || *pSpec == NULL)
    {
        ippsFree(pInitBuf);
        ippsFree(*pSpec);
        return ippStsNoMemErr;
    }

    /* Filter initialization */
    // 0, 0.75 is the value of OpenCV is using to config IPP cubic filter
    if (gBitDepth == 8)
    {
#ifdef USE_BICUBIC
        status = IPPRInit(8)(srcSize, dstSize, 0, 0.75, *pSpec, pInitBuf);
#elif USE_LANCZOS
        status = IPPRInit(8)(srcSize, dstSize, 3, *pSpec, pInitBuf);
#else
        status = IPPRInit(8)(srcSize, dstSize, *pSpec);
#endif
    }
    else
    {
#ifdef USE_BICUBIC
        status = IPPRInit(16)(srcSize, dstSize, 0, 0.75, *pSpec, pInitBuf);
#elif USE_LANCZOS
        status = IPPRInit(16)(srcSize, dstSize, 3, *pSpec, pInitBuf);
#else
        status = IPPRInit(16)(srcSize, dstSize, *pSpec);
#endif
    }

    ippsFree(pInitBuf);
    if (status != ippStsNoErr)
    {
        ippsFree(*pSpec);
        return status;
    }

    /* work buffer size */
    status = ippiResizeGetBufferSize_8u(*pSpec, dstSize, numChannels, &bufSize);
    if (status != ippStsNoErr)
    {
        ippsFree(*pSpec);
        return status;
    }

    *pBuffer = ippsMalloc_8u(bufSize);
    if (*pBuffer == NULL)
    {
        ippsFree(*pSpec);
        return ippStsNoMemErr;
    }

    return status;
}

enum class Axis
{
    NONE = 0,
    ROW,
    COL
};

// NumPy Reference: https://www.numpy.org/devdocs/reference/generated/numpy.gradient.html
template <typename DT>
RNLERRORTYPE gradient_for_reference(uint8_t *inArray, DT *outArray, unsigned int width /*cols*/, unsigned int height /*rows*/, unsigned int widthWithPadding, Axis inAxis = Axis::ROW)
{
    if (inAxis == Axis::ROW)
    {
        if (height < 2)
        {
            std::cout << "image size is too small to compute by ROW\n";
            return RNLErrorBadParameter;
        }

        // first do the first and last rows
        for (auto col = 0; col < width; ++col)
        {
            outArray[col] = static_cast<DT>(inArray[widthWithPadding + col] - inArray[col]);
        }
        for (auto col = 0; col < width; ++col)
        {
            outArray[(height - 1) * widthWithPadding + col] = static_cast<DT>(inArray[(height - 1) * widthWithPadding + col] - inArray[(height - 2) * widthWithPadding + col]);
        }

        // then rip through the rest of the array
        for (auto col = 0; col < width; ++col)
        {
            for (auto row = 1; row < height - 1; ++row)
            {
                outArray[row * widthWithPadding + col] = static_cast<DT>(inArray[(row + 1) * widthWithPadding + col] - inArray[(row - 1) * widthWithPadding + col]);
            }
        }
    }
    else if (inAxis == Axis::COL)
    {
        if (width < 2)
        {
            std::cout << "image size is too small to compute by COL\n";
            return RNLErrorBadParameter;
        }

        // first do the first and last columns
        for (auto row = 0; row < height; ++row)
        {
            outArray[row * widthWithPadding] = static_cast<DT>(inArray[row * widthWithPadding + 1] - inArray[row * widthWithPadding]);
            outArray[row * widthWithPadding + width - 1] = static_cast<DT>(inArray[row * widthWithPadding + width - 1] - inArray[row * widthWithPadding + width - 2]);
        }

        // then rip through the rest of the array
        for (auto row = 0; row < height; ++row)
        {
            for (auto col = 1; col < width - 1; ++col)
            {
                outArray[row * widthWithPadding + col] = static_cast<DT>(inArray[row * widthWithPadding + col + 1] - inArray[row * widthWithPadding + col - 1]);
            }
        }
    }
    else
    {
        std::cout << "gradient must be applied on either ROW or COL\n";
        return RNLErrorBadParameter;
    }
    return RNLErrorNone;
}

/************************************************************
 *   CT Blending functions
 ************************************************************/
template <typename DT>
static void CTRandomness_for_reference(DT *LRImage, DT *HRImage, DT *outImage, unsigned int width /*cols*/, unsigned int height /*rows*/, unsigned int widthWithPadding)
{
    for (auto r = CTmargin; r < height - CTmargin; r++)
    {
        for (auto c = CTmargin; c < width - CTmargin; c++)
        {
            int census = 0;
            // Census transform
            for (int i = -CTmargin; i <= CTmargin; i++)
            {
                for (int j = -CTmargin; j <= CTmargin; j++)
                {
                    if (unlikely(i == 0 && j == 0))
                        continue;
                    if (LRImage[(r + i) * widthWithPadding + (c + j)] < LRImage[r * widthWithPadding + c])
                        census++;
                }
            }

            double weight = (double)census / (double)CTnumberofPixel;
            outImage[r * widthWithPadding + c] = weight * HRImage[r * widthWithPadding + c] + (1 - weight) * LRImage[r * widthWithPadding + c];
        }
    }
}

template <typename DT>
static void CTCountOfBitsChanged_for_reference(DT *LRImage, DT *HRImage, DT *outImage, unsigned int width /*cols*/, unsigned int height /*rows*/, unsigned int widthWithPadding)
{
    for (auto r = CTmargin; r < height - CTmargin; r++)
    {
        for (auto c = CTmargin; c < width - CTmargin; c++)
        {
            int hammingDistance = 0;

            // Census transform
            for (int i = -CTmargin; i <= CTmargin; i++)
            {
                for (int j = -CTmargin; j <= CTmargin; j++)
                {
                    if (unlikely(i == 0 && j == 0))
                        continue;
                    hammingDistance += std::abs((LRImage[(r + i) * widthWithPadding + (c + j)] < LRImage[r * widthWithPadding + c] ? 1 : 0) - (HRImage[(r + i) * widthWithPadding + (c + j)] < HRImage[r * widthWithPadding + c] ? 1 : 0));
                }
            }
            DT weight = (DT)hammingDistance / (DT)CTnumberofPixel;
            outImage[r * widthWithPadding + c] = weight * LRImage[r * widthWithPadding + c] + (1 - weight) * HRImage[r * widthWithPadding + c];
        }
    }
}

static void CTCountOfBitsChanged_AVX2(float *LRImage, float *HRImage, float *outImage, unsigned int width /*cols*/, unsigned int height /*rows*/, unsigned int widthWithPadding)
{

    for (auto r = CTmargin; r < height - CTmargin; r++)
    {
        for (auto c = CTmargin; c < width - CTmargin; c++)
        {
            int hammingDistance = 0;
            __m256 row_lr_f, row_hr_f, center_lr_f, center_hr_f;

            load3x3_ps(LRImage, c, r, widthWithPadding, &row_lr_f, &center_lr_f);
            load3x3_ps(HRImage, c, r, widthWithPadding, &row_hr_f, &center_hr_f);

            // compare if neighbors < centerpixel, toggle bit in mask if true
            int highbit = 0x80000000;
            const __m256i highbit_epi32 = _mm256_setr_epi32(highbit, highbit, highbit, highbit, highbit, highbit, highbit, highbit);

            __m256i cmp_lr_epi32 = compare3x3_AVX256_32f(row_lr_f, center_lr_f, highbit_epi32);
            __m256i cmp_hr_epi32 = compare3x3_AVX256_32f(row_hr_f, center_hr_f, highbit_epi32);

            // hammingDistance = abs( lr_cmp - hr_cmp )
            __m256i cmp_epi32 = _mm256_abs_epi32(_mm256_sub_epi32(cmp_lr_epi32, cmp_hr_epi32));
            // count # of bits in mask
            hammingDistance += sumitup_256_epi32(cmp_epi32);

            float weight = (float)hammingDistance / (float)CTnumberofPixel;
            outImage[r * widthWithPadding + c] = weight * LRImage[r * widthWithPadding + c] + (1 - weight) * HRImage[r * widthWithPadding + c];
        }
    }
}

// LRImage: cheap up scaled. HRImage: RAISR refined. outImage: output buffer in 8u.
// rows: rows of LRImage/HRImage. startRow: seg start row. blendingZone: zone to run blending.
// cols: stride for buffers in DT type.
// outImageCols: stride for outImage buffer
static void CTCountOfBitsChangedSegment_AVX2(float *LRImage, float *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols)
{
    int rowStartOffset = blendingZone.first - startRow;
    int rowEndOffset = blendingZone.second - startRow;
    for (auto r = rowStartOffset; r < rowEndOffset; r++)
    {
        for (auto c = CTmargin; c < cols - CTmargin; c++)
        {
            int hammingDistance = 0;
            __m256 row_lr_f, row_hr_f, center_lr_f, center_hr_f;

            load3x3_ps(LRImage, c, r, cols, &row_lr_f, &center_lr_f);
            load3x3_ps(HRImage, c, r, cols, &row_hr_f, &center_hr_f);

            // compare if neighbors < centerpixel, toggle bit in mask if true
            int highbit = 0x80000000;
            const __m256i highbit_epi32 = _mm256_setr_epi32(highbit, highbit, highbit, highbit, highbit, highbit, highbit, highbit);

            __m256i cmp_lr_epi32 = compare3x3_AVX256_32f(row_lr_f, center_lr_f, highbit_epi32);
            __m256i cmp_hr_epi32 = compare3x3_AVX256_32f(row_hr_f, center_hr_f, highbit_epi32);

            // hammingDistance = abs( lr_cmp - hr_cmp )
            __m256i cmp_epi32 = _mm256_abs_epi32(_mm256_sub_epi32(cmp_lr_epi32, cmp_hr_epi32));
            // count # of bits in mask
            hammingDistance += sumitup_256_epi32(cmp_epi32);

            float weight = (float)hammingDistance / (float)CTnumberofPixel;
            float val = weight * LRImage[r * cols + c] + (1 - weight) * HRImage[r * cols + c];
            val += 0.5; // to round the value
            // convert 32f to 8bit/10bit
            if (gBitDepth == 8)
            {
                outImage[(startRow + r) * outImageCols + c] = (unsigned char)(val < gMin8bit ? gMin8bit : (val > gMax8bit ? gMax8bit : val));
            }
            else
            {
                unsigned short *out = (unsigned short *)outImage;
                out[(startRow + r) * outImageCols + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
            }
        }
    }
}

// LRImage: cheap up scaled. HRImage: RAISR refined. outImage: output buffer in 8u.
// rows: rows of LRImage/HRImage. startRow: seg start row. blendingZone: zone to run blending.
// cols: stride for buffers in DT type.
// outImageCols: stride for outImage buffer
template <typename DT>
static void CTCountOfBitsChangedSegment(DT *LRImage, DT *HRImage, const int rows, const int startRow, const std::pair<int, int> blendingZone, unsigned char *outImage, const int cols, const int outImageCols)
{
    // run census transform on a CTwindowSize * CTwindowSize block centered by [r, c]
    int rowStartOffset = blendingZone.first - startRow;
    int rowEndOffset = blendingZone.second - startRow;
    for (auto r = rowStartOffset; r < rowEndOffset; r++)
    {
        for (auto c = CTmargin; c < cols - CTmargin; c++)
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
            DT weight = (DT)hammingDistance / (DT)CTnumberofPixel;
            float val = weight * LRImage[r * cols + c] + (1 - weight) * HRImage[r * cols + c];
            val += 0.5; // to round the value
            // convert 32f to 8bit/10bit
            if (gBitDepth == 8)
            {
                outImage[(startRow + r) * outImageCols + c] = (unsigned char)(val < gMin8bit ? gMin8bit : (val > gMax8bit ? gMax8bit : val));
            }
            else
            {
                unsigned short *out = (unsigned short *)outImage;
                out[(startRow + r) * outImageCols + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
            }
        }
    }
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

            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n1_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n1_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n2_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n2_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n3_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n3_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n4_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n4_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n5_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n5_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n6_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n6_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n7_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n7_HR_ps, center_HR_ps, cmp_le))))));
            hammingDistance_epi32 = _mm256_add_epi32( hammingDistance_epi32,
                                        _mm256_abs_epi32(_mm256_sub_epi32(
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n8_LR_ps, center_LR_ps, cmp_le))),
                                            _mm256_and_si256(one_epi32, _mm256_castps_si256(_mm256_cmp_ps(n8_HR_ps, center_HR_ps, cmp_le))))));

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
                _mm_storeu_si128((__m128i *) &out[(startRow + r) * outImageCols + c], _mm256_extractf128_si256(perm_epu, 0));
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
                out[(startRow + r) * outImageCols + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
            }
        }
    }
}

int inline CTRandomness_C(float *inYUpscaled32f, int cols, int r, int c, int pix)
{
    // Census transform
    int census_count = 0;
    for (int i = -1; i <= CTmargin; i++)
    {
        for (int j = -1; j <= CTmargin; j++)
        {
            if (unlikely(i == 0 && j == 0))
                continue;
            if (inYUpscaled32f[(r + i) * cols + (c + j + pix)] < inYUpscaled32f[r * cols + c + pix])
                census_count++;
        }
    }
    return census_count;
}

/************************************************************
 *   Hashing functions
 ************************************************************/
#ifdef USE_ATAN2_APPROX
const float ONEQTR_PI = M_PI / 4.0;
const float THRQTR_PI = 3.0 * M_PI / 4.0;
inline float atan2Approximation(float y, float x)
{
    float r, angle;
    float abs_y = fabs(y) + 1e-10f; // kludge to prevent 0/0 condition
    if (unlikely(x < 0.0f))
    {
        r = (x + abs_y) / (abs_y - x);
        angle = THRQTR_PI;
    }
    else
    {
        r = (x - abs_y) / (x + abs_y);
        angle = ONEQTR_PI;
    }
    angle += (0.1963f * r * r - 0.9817f) * r;
    if (likely(y < 0.0f))
        return (-angle); // negate if in quad III or IV
    else
        return (angle);
}
#endif

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

void inline GetHashValue_AVX256_32f(float GTWG[8][4], int passIdx, int32_t *idx) {
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

    __m256 m_a_ps = _mm256_setr_ps (GTWG[0][0], GTWG[1][0], GTWG[2][0], GTWG[3][0],
                                   GTWG[4][0], GTWG[5][0], GTWG[6][0], GTWG[7][0]);
    __m256 m_b_ps = _mm256_setr_ps (GTWG[0][1], GTWG[1][1], GTWG[2][1], GTWG[3][1],
                                   GTWG[4][1], GTWG[5][1], GTWG[6][1], GTWG[7][1]);
    __m256 m_d_ps = _mm256_setr_ps (GTWG[0][3], GTWG[1][3], GTWG[2][3], GTWG[3][3],
                                   GTWG[4][3], GTWG[5][3], GTWG[6][3], GTWG[7][3]);
    __m256 T_ps = _mm256_add_ps(m_a_ps, m_d_ps);
    __m256 D_ps = _mm256_sub_ps( _mm256_mul_ps( m_a_ps, m_d_ps),
                                _mm256_mul_ps( m_b_ps, m_b_ps));

    __m256 sqr_ps = _mm256_sqrt_ps( _mm256_sub_ps( _mm256_div_ps ( _mm256_mul_ps(T_ps, T_ps),
                                                                   _mm256_broadcast_ss(&four)), D_ps));

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

    __m256 sqrtL1_ps = _mm256_sqrt_ps( L1_ps );
    __m256 sqrtL2_ps = _mm256_sqrt_ps( L2_ps );
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

inline int int_floor(float x)
{
    int i = (int)x;     /* truncate */
    return i - (i > x); /* convert trunc to floor */
}

template <class dType>
inline int searchsorted(std::vector<dType> &array, dType value)
{
    int i = 0;
    for (auto data : array)
    {
        if (value <= data)
            return i;
        i++;
    }
    return i;
}

int inline GetHashValue(float *GTWG, int pass)
{
    /*  Consider the eigenvalues and eigenvectors of
     *      | a   b |
     *      | c   d |
     * */
    // NOTE: m_b == m_c
    const float m_a = GTWG[0];
    const float m_b = GTWG[1];
    // const float m_c = GTWG[2];
    const float m_d = GTWG[3];
    const float T = m_a + m_d;
    const float D = m_a * m_d - m_b * m_b;
    const float sqr = std::sqrt((T * T) / 4 - D);
    const float half_T = T / 2;
    const float L1 = half_T + sqr; // largest eigenvalude
    const float L2 = half_T - sqr; // smaller eigenvalude

    float angle = 0;
    if (likely(m_b != 0))
    {
#ifdef USE_ATAN2_APPROX
        angle = atan2Approximation(m_b, L1 - m_d);
#else
        angle = std::atan2(m_b, L1 - m_d);
#endif
    }
    else
    { // m_b == 0
#ifdef USE_ATAN2_APPROX
        angle = atan2Approximation(0, 1);
#else
        angle = std::atan2(0, 1);
#endif
    }

    if (unlikely(angle < 0))
        angle += PI;
    const float sqrtL1 = std::sqrt(L1);
    const float sqrtL2 = std::sqrt(L2);
    const float coherence = (sqrtL1 - sqrtL2) / (sqrtL1 + sqrtL2 + 0.00000000000000001);
    const float strength = L1;
    int angleIdx = int_floor(angle * gQAngle);

    angleIdx = angleIdx > (gQuantizationAngle - 1) ? gQuantizationAngle - 1 : (angleIdx < 0 ? 0 : angleIdx);

    int strengthIdx;
    int coherenceIdx;
    if (pass == 0)
    {
        strengthIdx = searchsorted(gQStr, strength);
        coherenceIdx = searchsorted(gQCoh, coherence);
    }
    else
    {
        strengthIdx = searchsorted(gQStr2, strength);
        coherenceIdx = searchsorted(gQCoh2, coherence);
    }

    return angleIdx * gQuantizationStrength * gQuantizationCoherence +
           strengthIdx * gQuantizationCoherence +
           coherenceIdx;
}

RNLERRORTYPE processSegment(VideoDataType *srcY, VideoDataType *final_outY, BlendingMode blendingMode, int threadIdx)
{
    VideoDataType *inY;
    VideoDataType *outY;
    for (int passIdx = 0; passIdx < gPasses; passIdx++)
    {
#ifdef MEASURE_TIME
        auto start = std::chrono::system_clock::now();
#endif
        inY = srcY;
        if (passIdx == gPasses - 1) // the last pass
        {
            // because of the each threads share few lines (up and down) with 2 others threads, we need to wait that the first passs is done for the 2 others threads
            if (threadIdx != 0 && gPasses == 2) // in case of the second pass we need to wait for the upper segment to be done before to begin the second pass to avoid line between segment
            {
                while (threadStatus[threadIdx - 1] == 0)
                {
                }
            }
            if ((threadIdx != gThreadCount - 1) && (gPasses == 2)) // we also need to wait for the lower to be done before to begin the second pass
            {
                while (threadStatus[threadIdx + 1] == 0)
                {
                }
            }

            outY = final_outY; // change output to the final one
            if (gPasses == 2)
            {
                inY = gIntermediateY; // 2nd pass in 2pass, set inY as output of 1st pass
            }
        }
        else // 1st pass when 2 pass is enabled
        {
            outY = gIntermediateY; // change the output to VideoDataType gIntermediateY to save the output of 1st pass
        }

        // step is mean line size in a frame, for video the line size should be multiplies of the CPU alignment(16 or 32 bytes), 
        // the outY->step may greater than or equal to outY->width.
        // the step of gIppCtx.segZones[passIdx][threadIdx].inYUpscaled is equal to the outY->width 
        const int rows = outY->height;
        const int cols = outY->width;
        const int step = outY->width;

        // 1. Prepare cheap up-scaled 32f data
        IppStatus status = ippStsNoErr;
        int startRow = gIppCtx.segZones[passIdx][threadIdx].scaleStartRow;
        int endRow = gIppCtx.segZones[passIdx][threadIdx].scaleEndRow;
        int segRows = endRow - startRow;

        Ipp8u *pDst = gIppCtx.segZones[passIdx][threadIdx].inYUpscaled;
        float *pSeg32f = gIppCtx.segZones[passIdx][threadIdx].inYUpscaled32f;
        float *pRaisr32f = gIppCtx.segZones[passIdx][threadIdx].raisr32f;
        if ((passIdx + 1) == gTwoPassMode) // upscales in this pass
        {
            if (gBitDepth == 8)
            {
                Ipp8u *pSrc = inY->pData + inY->step * (startRow / gRatio);
                status = IPPResize(8)(pSrc, inY->step, pDst, step, {0, 0}, {(int)outY->width, segRows},
                                      ippBorderRepl, 0, gIppCtx.specY[threadIdx], gIppCtx.pbufferY[threadIdx]);
            }
            else
            {
                Ipp16u *pSrc = (Ipp16u *)(inY->pData + inY->step * (startRow / gRatio));
                status = IPPResize(16)((Ipp16u *)pSrc, inY->step, (Ipp16u *)pDst, step, {0, 0}, {(int)outY->width, segRows},
                                       ippBorderRepl, 0, gIppCtx.specY[threadIdx], gIppCtx.pbufferY[threadIdx]);
            }
            if (ippStsNoErr != status)
            {
                std::cout << "[RAISR ERROR] resize Y segment failed! segment id: " << threadIdx << std::endl;
                return RNLErrorBadParameter;
            }
        }
        else // no upscaling in this pass
        {
            Ipp8u *pSrc8u = inY->pData + inY->step * startRow; // inY is already upscaled
            memcpy(pDst, pSrc8u, step * segRows);
        }

        if (gBitDepth == 8)
            ippiConvert_8u32f_C1R(pDst, cols,
                                  pSeg32f, cols * sizeof(float), {(int)cols, segRows});
        else
            ippiConvert_16u32f_C1R((Ipp16u *)pDst, cols,
                                  pSeg32f, cols * sizeof(float), {(int)cols, segRows});

        // 2. Run hashing
        // Update startRow, endRow for hashing algo
        startRow = gIppCtx.segZones[passIdx][threadIdx].raisrStartRow;
        endRow = gIppCtx.segZones[passIdx][threadIdx].raisrEndRow;

        // Handle top and bottom borders
        if (startRow == 0)
        {
            // it needs to do memcpy line by line when the line size of outY->pData is not equal to pDst's line size.
            if (step == outY->step) {
                memcpy(outY->pData, pDst, outY->step * gLoopMargin + gLoopMargin);
            } else {
                for (int i = 0; i < gLoopMargin; i++) {
                     memcpy(outY->pData + i * outY->step, pDst + i * step, step);
                }
                memcpy(outY->pData + gLoopMargin * outY->step, pDst + gLoopMargin * step, gLoopMargin);
            }
        }
        if (endRow == rows)
        {
            if (step == outY->step) {
                memcpy(outY->pData + (rows - gLoopMargin) * step - gLoopMargin,
                       pDst + (segRows - gLoopMargin) * step - gLoopMargin,
                       outY->step * gLoopMargin + gLoopMargin);
            } else {
                memcpy(outY->pData + (rows - gLoopMargin - 1) * outY->step +  outY->width - gLoopMargin,
                       pDst + (segRows - gLoopMargin) * step - gLoopMargin,
                       gLoopMargin);

                for (int i = gLoopMargin; i > 0; i--) {
                     memcpy(outY->pData + (rows - i) * outY->step,
                            pDst  + (segRows - i) * step,
                            step);
                }
            }
        }
        memcpy(pRaisr32f, pSeg32f, sizeof(float) * cols * segRows);

        startRow = startRow < gLoopMargin ? gLoopMargin : startRow;
        endRow = endRow > (rows - gLoopMargin) ? (rows - gLoopMargin) : endRow;

        float GTWG[unrollSizePatchBased][4];
        int pixelType[unrollSizePatchBased];
        int hashValue[unrollSizePatchBased];
        const float *fbase[unrollSizePatchBased];
        float pixbuf[unrollSizePatchBased][128] __attribute__((aligned(64)));
        int pix;
        int census = 0;

        memset(pixbuf, 0, sizeof(float) * unrollSizePatchBased * 128);
        // NOTE: (r, c) is coordinate in the full HR image, which represents area processed by RAISR
        for (int r = startRow; r < endRow; r++)
        {
            // update coordinate: convert r to coordinate in seg buffer pSeg32f
            // NOTE: use rOffset with pSeg32f
            int rOffset = r - gIppCtx.segZones[passIdx][threadIdx].scaleStartRow;
            for (int c = gLoopMargin; c <= cols - gLoopMargin; c += unrollSizePatchBased)
            {
#pragma unroll(unrollSizePatchBased)
                for (pix = 0; pix < unrollSizePatchBased; pix++)
                {
                    pixelType[pix] = ((r - gPatchMargin) % gRatio) * gRatio + ((c + pix - gPatchMargin) % gRatio);
                }

#pragma unroll(unrollSizePatchBased / 2)
                for (pix = 0; pix < unrollSizePatchBased / 2; pix++)
                {
                    if (gAsmType == AVX2)
                        computeGTWG_Segment_AVX256_32f(pSeg32f, rows, cols, rOffset, c + 2 * pix, &GTWG[2 * pix], &pixbuf[2 * pix][0], &pixbuf[2 * pix + 1][0]);
                    else if (gAsmType == AVX512)
                        computeGTWG_Segment_AVX512_32f(pSeg32f, rows, cols, rOffset, c + 2 * pix, &GTWG[2 * pix], &pixbuf[2 * pix][0], &pixbuf[2 * pix + 1][0]);
                    else
                    {
                        std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                        return RNLErrorBadParameter;
                    }
                }

                GetHashValue_AVX256_32f(GTWG, passIdx, hashValue);

                for (pix = 0; pix < unrollSizePatchBased; pix++)
                {
                    if (passIdx == 0)
                        fbase[pix] = gFilterBuckets[hashValue[pix]][pixelType[pix]];
                    else
                        fbase[pix] = gFilterBuckets2[hashValue[pix]][pixelType[pix]];
                }

#pragma unroll(unrollSizePatchBased)
                for (pix = 0; pix < unrollSizePatchBased; pix++)
                {
                    if (likely(c + pix < cols - gLoopMargin))
                    {
                        float curPix;
                        if (gAsmType == AVX2)
                            curPix  = DotProdPatch_AVX256_32f(pixbuf[pix], fbase[pix]);
                        else if (gAsmType == AVX512)
                            curPix  = DotProdPatch_AVX512_32f(pixbuf[pix], fbase[pix]);
                        else 
                        {
                            std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                            return RNLErrorBadParameter;
                        }
                        if ((gBitDepth == 8 && curPix > gMin8bit && curPix < gMax8bit) ||
                            (gBitDepth != 8 && curPix > gMin16bit && curPix < gMax16bit))
                            pRaisr32f[rOffset * cols + c + pix] = curPix;
                        else
                            curPix = pSeg32f[rOffset * cols + c + pix];

                        // CT-Blending, CTRandomness
                        if (blendingMode == Randomness)
                        {
                            if (gAsmType == AVX2)
                                census = CTRandomness_AVX256_32f(pSeg32f, cols, rOffset, c, pix);
                            else if (gAsmType == AVX512)
                                census = CTRandomness_AVX512_32f(pSeg32f, cols, rOffset, c, pix);
                            else
                            {
                                std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                                return RNLErrorBadParameter;
                            }

                            float weight = (float)census / (float)CTnumberofPixel;
                            // position in the whole image: r * cols + c + pix
                            float val = weight * curPix + (1 - weight) * pSeg32f[rOffset * cols + c + pix];

                            val += 0.5; // to round the value
                            if (gBitDepth == 8)
                            {
                                outY->pData[r * outY->step + c + pix] = (unsigned char)(val < gMin8bit ? gMin8bit : (val > gMax8bit ? gMax8bit : val));
                            }
                            else
                            {
                                unsigned short *out = (unsigned short *)outY->pData;
                                out[r * outY->step + c + pix] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
                            }
                        }
                    }
                }
            }
            // Copy right border pixels for this row and left border pixels for next row
            if (step == outY->step) {
                memcpy(outY->pData + r * step - gLoopMargin, pDst + rOffset * step - gLoopMargin, 2 * gLoopMargin);
            } else {
                memcpy(outY->pData + (r -1 ) * outY->step + outY->width - gLoopMargin,
                       pDst + rOffset * step - gLoopMargin,
                       gLoopMargin);
                memcpy(outY->pData + r * outY->step,
                       pDst + rOffset * step,
                       gLoopMargin);
            }
        }
        // 3. Run CT-Blending
        if (blendingMode == CountOfBitsChanged)
        {
            int segStart = gIppCtx.segZones[passIdx][threadIdx].scaleStartRow;
            CTCountOfBitsChangedSegment_AVX256_32f(pSeg32f, pRaisr32f, segRows, segStart, {gIppCtx.segZones[passIdx][threadIdx].blendingStartRow, gIppCtx.segZones[passIdx][threadIdx].blendingEndRow}, outY->pData, cols, outY->step);
        }

        threadStatus[threadIdx] = 1;
    }

#ifdef MEASURE_TIME
    auto end = std::chrono::system_clock::now();
    std::chrono::duration<double> elapsed_seconds = end - start;
    std::time_t end_time = std::chrono::system_clock::to_time_t(end);
    std::cout << passIdx << ": " << elapsed_seconds.count() << "s\n";
#endif
    return RNLErrorNone;
}

/************************************************************
 *   API functions
 ************************************************************/
RNLERRORTYPE RNLProcess(VideoDataType *inY, VideoDataType *inCr, VideoDataType *inCb,
                        VideoDataType *outY, VideoDataType *outCr, VideoDataType *outCb, BlendingMode blendingMode)
{
    if (!inCr || !inCr->pData || !outCr || !outCr->pData ||
        !inCb || !inCb->pData || !outCb || !outCb->pData ||
        !inY || !inY->pData || !outY || !outY->pData)
        return RNLErrorBadParameter;

#ifndef DISABLE_AVX512

    memset((void *)threadStatus, 0, 120 * sizeof(threadStatus[0]));

    // multi-threaded patch-based approach
    std::vector<std::future<RNLERRORTYPE>> results;

    if (gBitDepth == 8)
    {
        // process Y channel by RAISR algorithm
        for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
            results.emplace_back(gPool->enqueue(processSegment, inY, outY, blendingMode, threadIdx));

        // meanwhile, run UV cheap upscale on current thread
        IPPResize(8)(inCr->pData, inCr->step, outCr->pData, outCr->step, {0, 0}, {(int)outCr->width, (int)outCr->height},
                     ippBorderRepl, 0, gIppCtx.specUV, gIppCtx.pbufferUV);

        IPPResize(8)(inCb->pData, inCb->step, outCb->pData, outCb->step, {0, 0}, {(int)outCb->width, (int)outCb->height},
                     ippBorderRepl, 0, gIppCtx.specUV, gIppCtx.pbufferUV);
    }
    else
    {
        for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
            results.emplace_back(gPool->enqueue(processSegment, inY, outY, blendingMode, threadIdx));

        IPPResize(16)((Ipp16u *)inCr->pData, inCr->step, (Ipp16u *)outCr->pData, outCr->step, {0, 0}, {(int)outCr->width, (int)outCr->height},
                      ippBorderRepl, 0, gIppCtx.specUV, gIppCtx.pbufferUV);

        IPPResize(16)((Ipp16u *)inCb->pData, inCb->step, (Ipp16u *)outCb->pData, outCb->step, {0, 0}, {(int)outCb->width, (int)outCb->height},
                      ippBorderRepl, 0, gIppCtx.specUV, gIppCtx.pbufferUV);
    }

    for (auto &&result : results)
    {
        result.get();
    }

#endif

    return RNLErrorNone;
}

RNLERRORTYPE RNLInit(std::string &modelPath,
                     unsigned int ratio,
                     unsigned int bitDepth,
                     RangeType    rangeType,
                     unsigned int threadCount,
                     ASMType      asmType,
                     unsigned int passes,
                     unsigned int twoPassMode)
{
    std::cout << "RAISR [version]:\tRAISR Native Lib v" << RAISR_VERSION_MAJOR << "." << RAISR_VERSION_MINOR << std::endl;
    std::cout << "LIB Build date: " << __DATE__ << ", " << __TIME__ << std::endl;
    std::cout << "-------------------------------------------\n";

    if (!is_machine_intel())
    {
        std::cout << "[RAISR ERROR] Only supported on Intel platforms. " << std::endl;
        return RNLErrorUndefined;
    }

    if (passes == 2)
    {
        gPasses = passes;
        gTwoPassMode = twoPassMode;
        std::cout << "--------------- running 2 pass ---------------\n";
    } else if(passes == 1 && twoPassMode == 2)
        std::cout << "[RAISR WARNING] 1 pass with upscale in 2d pass, mode = 2 ignored !" << std::endl;
    else if (passes != 1) {
        std::cout << "[RAISR ERROR] Only support passes 1 or 2. " << std::endl;
        return RNLErrorUndefined;
    }

    std::string hashtablePath = modelPath + "/" + "/filterbin_2";
    std::string QStrPath      = modelPath + "/" + "/Qfactor_strbin_2";
    std::string QCohPath      = modelPath + "/" + "/Qfactor_cohbin_2";
    std::string configPath    = modelPath + "/" + "/config";

    if (bitDepth == 8)
    {
        hashtablePath += "_8";
        QStrPath += "_8";
        QCohPath += "_8";
        gMin8bit = rangeType == VideoRange ? MIN8BIT_VIDEO : MIN_FULL;
        gMax8bit = rangeType == VideoRange ? MAX8BIT_VIDEO : MAX8BIT_FULL;
    }
    else if (bitDepth == 10)
    {
        hashtablePath += "_10";
        QStrPath += "_10";
        QCohPath += "_10";
        gMin16bit = rangeType == VideoRange ? MIN10BIT_VIDEO : MIN_FULL;
        gMax16bit = rangeType == VideoRange ? MAX10BIT_VIDEO : MAX10BIT_FULL;
    }
    else if (bitDepth == 16)
    {
        hashtablePath += "_16";
        QStrPath += "_16";
        QCohPath += "_16";
        gMin16bit = MIN_FULL;
        gMax16bit = MAX16BIT_FULL;
    }
    else
    {
        std::cout << "[RAISR ERROR] bit depth: " << bitDepth << "bits is NOT supported." << std::endl;
        return RNLErrorBadParameter;
    }
    gRatio = ratio;
    gAsmType = asmType;
    gBitDepth = bitDepth;

    // Read config file
    std::string line;
    std::ifstream configFile(configPath);
    if (!configFile.is_open())
    {
        std::cout << "[RAISR ERROR] Unable to open config file: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }

    std::getline(configFile, line);
    std::istringstream configiss(line);
    std::vector<std::string> configTokens{std::istream_iterator<std::string>{configiss},
                                          std::istream_iterator<std::string>{}};
    if (configTokens.size() != 4)
    {
        std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }

    gQuantizationAngle = std::stoi(configTokens[0].c_str());
    gQAngle = gQuantizationAngle / PI;
    gQuantizationStrength = std::stoi(configTokens[1].c_str());
    gQuantizationCoherence = std::stoi(configTokens[2].c_str());
    // Varify hashtable file format
    gPatchSize = std::stoi(configTokens[3].c_str());
    ;
    gPatchMargin = gPatchSize >> 1;
    gLoopMargin = (gPatchSize >> 1) + 1;
    gResizeExpand = (gLoopMargin + 2);
    unsigned int patchAreaSize = gPatchSize * gPatchSize;
    g64AlinedgPatchAreaSize = ((patchAreaSize + 64 - 1) / 64) * 64;
    configFile.close();

    if (RNLErrorNone != ReadTrainedData(hashtablePath, QStrPath, QCohPath, 1 /*first pass*/))
        return RNLErrorBadParameter;

    if (gPasses == 2 && RNLErrorNone != ReadTrainedData(hashtablePath, QStrPath, QCohPath, 2 /*second pass*/))
        return RNLErrorBadParameter;

    // create guassian kernel if patchSize is not default
    if (gPatchSize != defaultPatchSize)
    {
        float *kernel = new float[gPatchSize];
        createGaussianKernel<float>(gPatchSize, sigma, kernel);
        gPGaussian = new float[patchAreaSize * 2]; // 2 x n^2 array
        // compute kernel * kernel.t() ==> n x n
        for (int rowkernel = 0; rowkernel < gPatchSize; rowkernel++)
        {
            for (int colkernel = 0; colkernel < gPatchSize; colkernel++)
            {
                gPGaussian[rowkernel * gPatchSize + colkernel] = kernel[rowkernel] * kernel[colkernel];
            }
        }
        // append
        memcpy(gPGaussian + patchAreaSize, gPGaussian, patchAreaSize * sizeof(float));
        delete[] kernel;
    }

    threadCount = (threadCount == 0) ? 1 : threadCount;
    // multi-threaded patch-based approach
    gThreadCount = threadCount;
    gPool = new ThreadPool(gThreadCount);

    return RNLErrorNone;
}

RNLERRORTYPE RNLSetRes(VideoDataType *inY, VideoDataType *inCr, VideoDataType *inCb,
                       VideoDataType *outY, VideoDataType *outCr, VideoDataType *outCb)
{
    int rows, cols;
    IppStatus status = ippStsNoErr;

    if (gPasses == 2 && gTwoPassMode == 2)
    {
        rows = inY->height;
        cols = inY->width;
    }
    else
    {
        rows = outY->height;
        cols = outY->width;
    }

    if (gPasses == 2)
    {
        gIntermediateY = new VideoDataType;
        if (gBitDepth == 8)
        {
            gIntermediateY->pData = new unsigned char[cols * rows];
            gIntermediateY->step = cols;
        }
        else
        {
            gIntermediateY->pData = new unsigned char[cols * rows * BYTES_16BITS];
            gIntermediateY->step = cols * BYTES_16BITS;
        }
        gIntermediateY->width = cols;
        gIntermediateY->height = rows;
    }

    // multi-threaded patch-based approach
    // 1. split image into segments and store info in the buffer context
    // 2. init IPP for each segment

    gIppCtx.pbufferY = new Ipp8u *[gThreadCount];
    gIppCtx.specY = new IppiResizeSpec_32f *[gThreadCount];

    for (int i = 0; i < gPasses; i++)
    {
        gIppCtx.segZones[i] = new segZone[gThreadCount];
        rows = (i == 1 ? outY->height : rows);
        cols = (i == 1 ? outY->width : cols);
        int startRow = 0, endRow = 0, segHeight = 0;
        int rowsPerThread = ceil(rows / gThreadCount);
        int rowsOfRemainder = rows - rowsPerThread * gThreadCount;

        for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
        {
            threadStatus[threadIdx] = 0;
            if (threadIdx != gThreadCount - 1) // some middle threads cover 1 more row, having loopsOfRemainder divided evenly.
            {
                endRow = (threadIdx < rowsOfRemainder) ? (startRow + rowsPerThread + 1) : (startRow + rowsPerThread);
            }
            else
            {
                endRow = rows;
            }

            endRow = (endRow % 2 == 0) ? endRow : endRow + 1; // Resize works on even number
            endRow = endRow > rows ? rows : endRow;

            // set segment zone for blending
            gIppCtx.segZones[i][threadIdx].blendingStartRow = startRow <= 0 ? CTmargin : startRow;
            gIppCtx.segZones[i][threadIdx].blendingEndRow = endRow >= rows ? rows - CTmargin : endRow;
            // set segment zone for raisr hashing
            int expand = gHashingExpand;
            gIppCtx.segZones[i][threadIdx].raisrStartRow = (startRow - expand) < 0 ? 0 : (startRow - expand);
            gIppCtx.segZones[i][threadIdx].raisrEndRow = (endRow + expand) > rows ? rows : (endRow + expand);
            // set segment zone for cheap resize
            expand += gResizeExpand;
            gIppCtx.segZones[i][threadIdx].scaleStartRow = (startRow - expand) < 0 ? 0 : (startRow - expand);
            gIppCtx.segZones[i][threadIdx].scaleEndRow = (endRow + expand) > rows ? rows : (endRow + expand);
            // std::cout << threadIdx << ", blendingStartRow: " << gIppCtx.segZones[i][threadIdx].blendingStartRow << ", blendingEndRow: " << gIppCtx.segZones[i][threadIdx].blendingEndRow
            //<< ", raisrStartRow: " << gIppCtx.segZones[i][threadIdx].raisrStartRow << ", raisrEndRow: " << gIppCtx.segZones[i][threadIdx].raisrEndRow
            //<< ", scaleStartRow: " << gIppCtx.segZones[i][threadIdx].scaleStartRow << ", scaleEndRow: " << gIppCtx.segZones[i][threadIdx].scaleEndRow << std::endl;

            segHeight = gIppCtx.segZones[i][threadIdx].scaleEndRow - gIppCtx.segZones[i][threadIdx].scaleStartRow;

            // NOTE: the intermediate buffer is to hold resize tile
            if (gBitDepth == 8)
                gIppCtx.segZones[i][threadIdx].inYUpscaled = new Ipp8u[segHeight * cols];
            else
                gIppCtx.segZones[i][threadIdx].inYUpscaled = new Ipp8u[segHeight * cols * BYTES_16BITS];
            gIppCtx.segZones[i][threadIdx].inYUpscaled32f = new float[segHeight * cols];
            gIppCtx.segZones[i][threadIdx].raisr32f = new float[segHeight * cols];

            // Filter initialization for Y channel segment

            IppiSize srcSize = {(int)inY->width, segHeight / gRatio};
            IppiSize dstSize = {(int)outY->width, segHeight};
            status = ippInit(srcSize, inY->step, dstSize, outY->step, &gIppCtx.specY[threadIdx], &gIppCtx.pbufferY[threadIdx]);
            if (status != ippStsNoErr)
            {
                std::cout << "[RAISR ERROR] ippInit fialed for Y! status=" << status << std::endl;
                return RNLErrorBadParameter;
            }

            startRow = endRow;
            if (startRow >= rows)
            {
                if (i == (gPasses - 1))
                	gThreadCount = threadIdx + 1;
                break;
            }
        }
    }

    // Init for UV channel
    status = ippInit({(int)inCr->width, (int)inCr->height}, inCr->step, {(int)outCr->width, (int)outCr->height}, outCr->step, &gIppCtx.specUV, &gIppCtx.pbufferUV);
    if (status != ippStsNoErr)
    {
        std::cout << "[RAISR ERROR] ippInit fialed for UV! status=" << status << std::endl;
        return RNLErrorBadParameter;
    }

    return RNLErrorNone;
}

RNLERRORTYPE RNLDeinit()
{

    for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
    {
        ippsFree(gIppCtx.specY[threadIdx]);
        ippsFree(gIppCtx.pbufferY[threadIdx]);

        for (int i = 0; i < gPasses; i++)
        {
            delete[] gIppCtx.segZones[i][threadIdx].inYUpscaled;
            delete[] gIppCtx.segZones[i][threadIdx].inYUpscaled32f;
            delete[] gIppCtx.segZones[i][threadIdx].raisr32f;
        }
    }
    delete gPool;
    delete[] gIppCtx.segZones[0];

    delete[] gIppCtx.specY;
    delete[] gIppCtx.pbufferY;
    ippsFree(gIppCtx.specUV);
    ippsFree(gIppCtx.pbufferUV);

    if (gPasses == 2)
    {
        delete[] gIppCtx.segZones[1];
        delete[] gFilterBuffer2;
        delete[] gIntermediateY->pData;
        delete gIntermediateY;
    }
    delete[] gFilterBuffer;
    delete[] gPGaussian;
    return RNLErrorNone;
}
