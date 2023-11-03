/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2022 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include "Raisr.h"
#include "Raisr_globals.h"
#include "Raisr_AVX256.h"
#include "Raisr_AVX256.cpp"
#include <fstream>
#include <iterator>
#include <iostream>
#include <sstream>
#include <ipp.h>
#include <cstring>
#include <cmath>
#include <vector>
#include <immintrin.h>
#include "ThreadPool.h"
#include "cpuid.h"
#include <chrono>

#ifdef __AVX512F__
#include "Raisr_AVX512.h"
#include "Raisr_AVX512.cpp"
#endif
#ifdef __AVX512FP16__
#include "Raisr_AVX512FP16.h"
#include "Raisr_AVX512FP16.cpp"
#endif

#ifdef ENABLE_RAISR_OPENCL
#include "Raisr_OpenCL.h"
#endif

#ifndef WIN32
#include <unistd.h>
#endif

//#define MEASURE_TIME

/************************************************************
 *   helper functions
 ************************************************************/
#define ALIGNED_SIZE(size, align) (((size) + (align)-1) & ~((align)-1))

static MachineVendorType get_machine_vendor()
{

    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    __get_cpuid(0, &eax, &ebx, &ecx, &edx);

    char vendor_string[13];
    memcpy((vendor_string + 0), &ebx, 4);
    memcpy((vendor_string + 4), &edx, 4);
    memcpy((vendor_string + 8), &ecx, 4);
    vendor_string[12] = 0;

    if (!strcmp(vendor_string, "GenuineIntel"))
        gMachineVendorType = INTEL;
    else if (!strcmp(vendor_string, "AuthenticAMD"))
        gMachineVendorType = AMD;
    else
        gMachineVendorType = VENDOR_UNSUPPORTED;
    return gMachineVendorType;
}

static bool machine_supports_feature(MachineVendorType vendor, ASMType type)
{
    bool ret = false;
    unsigned int eax = 0, ebx = 0, ecx = 0, edx = 0;

    if (vendor == INTEL ) {
        __get_cpuid_count(0x7, 0x0, &eax, &ebx, &ecx, &edx);

        if (type == AVX512_FP16) {
            // check for avx512fp16 and avx512vl flags
            if ( ((edx >> 23) & 0x1)
                && ((ebx >> 31) & 0x1) )
            {
                ret = true;
            }
        } else if (type == AVX512) {
            // check for avx512f and avx512vl flags
            if ( ((ebx >> 16) & 0x1)
                && ((ebx >> 31) & 0x1) )
            {
                ret = true;
            }
        } else if (type == AVX2) {
            // check for avx2 flag
            if ( (ebx >> 5) & 0x1)
            {
                ret = true;
            }
        }
    }
    else if (vendor == AMD)
    {
        __get_cpuid_count(0x7, 0x0, &eax, &ebx, &ecx, &edx);
        if (type == AVX512_FP16) {
            ret = false;
        } else if (type == AVX512) {
            ret = false;
        } else if (type == AVX2) {
            if ( (ebx >> 5) & 0x1)
            {
                ret = true;
            }
        }
    }
    return ret;
}

#ifdef __AVX512FP16__
void inline Convert_8u16f_8bit(Ipp8u *input, _Float16 *output, int cols, int rows) {
    for (int j = 0; j < rows; j++) {
        int c_limit_avx = cols - (cols%32);
        _Float16* outBuff = &((output)[j*cols]);
        Ipp8u* inBuff = &((input)[j*cols]);
        for (int i = 0; i < c_limit_avx; i +=32) {
            __m256i in_epu8 = _mm256_loadu_si256((__m256i *) inBuff);
            __m512h out_ph = _mm512_cvt_roundepu16_ph( _mm512_cvtepu8_epi16(in_epu8), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ph(outBuff, out_ph);
            inBuff += 32; outBuff += 32;
        }
        for (int i = c_limit_avx; i < cols; i++) { // remainders
            (output)[j*cols+i] = (_Float16)((input)[j*cols+i]);
        }
    }
}

void inline Convert_8u16f_10bit(Ipp16u *input, _Float16 *output, int cols, int rows) {
    for (int j = 0; j < rows; j++) {
        int c_limit_avx = cols - (cols%32);
        _Float16* outBuff = &((output)[j*cols]);
        Ipp16u* inBuff = &((input)[j*cols]);
        for (int i = 0; i < c_limit_avx; i+=32) {
            __m512i in_epu16 = _mm512_loadu_si512(inBuff);
            __m512h out_ph = _mm512_cvt_roundepu16_ph(in_epu16, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
            _mm512_storeu_ph(outBuff, out_ph);

            inBuff += 32; outBuff += 32;
        }
        for (int i = c_limit_avx; i < cols; i++) { // remainders
            (output)[j*cols+i] = (_Float16)((input)[j*cols+i]);
        }
    }
}
#endif

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
        DT t = std::exp(float(x * x) * float(scale2X));
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

RNLERRORTYPE RNLStoi(unsigned int *pValue, const char *configContent, std::string configPath)
{
    try
    {
        int tmp;
        tmp = std::stoi(configContent);
        if (tmp < 0)
        {
            std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
            return RNLErrorBadParameter;
        }
        *pValue = tmp;
        return RNLErrorNone;
    }
    catch (const std::invalid_argument &ia)
    {
        std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }

    catch (const std::out_of_range &oor)
    {
        std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }

    catch (const std::exception &e)
    {
        std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }
}

template <typename DT>
static RNLERRORTYPE ReadTrainedData(std::string hashtablePath, std::string QStrPath, std::string QCohPath, int pass, std::vector<std::vector<DT*>>& filterBuckets_in, DT* *filterBuffer_in, std::vector<DT>& QStr_in, std::vector<DT>& QCoh_in)
{
    if (pass == 2)
    {
        hashtablePath += "_2";
        QStrPath += "_2";
        QCohPath += "_2";
    }
    auto &filterBuckets = filterBuckets_in;
    auto &filterBuffer = *filterBuffer_in;
    auto &QStr = QStr_in;
    auto &QCoh = QCoh_in;

    std::string line;
    // Read filter file
    std::ifstream filterFile;
    filterFile.open(hashtablePath, std::ifstream::binary);
    if (!filterFile.is_open())
    {
        std::cout << "[RAISR ERROR] Unable to load model: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }

    unsigned int hashkeySize = 0, pixelTypes = 0, rows = 0;
    std::string data_type = "fp32";
    int type_size = data_type.size();
    filterFile.seekg(0, filterFile.end);
    long file_size = filterFile.tellg();
    filterFile.seekg(0, filterFile.beg);
    unsigned int weight_type_size = 4;

    filterFile.read(&data_type[0], type_size);
    if (data_type != "fp32" && data_type != "fp16") {
        std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }

    if (data_type == "fp16") {
        weight_type_size = 2;
    }

    filterFile.read(reinterpret_cast<char*>(&hashkeySize), sizeof(unsigned int));
    filterFile.read(reinterpret_cast<char*>(&pixelTypes), sizeof(unsigned int));
    filterFile.read(reinterpret_cast<char*>(&rows), sizeof(unsigned int));

    unsigned head_size = data_type.size() + 3 * sizeof(unsigned int);

    if ((file_size - head_size) != (hashkeySize * pixelTypes * rows * weight_type_size)) {
        std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
        return RNLErrorBadParameter;
    }

    int aligned_rows = 16 * (int)((rows + 15) / 16);

    if (hashkeySize != gQuantizationAngle * gQuantizationStrength * gQuantizationCoherence)
    {
        std::cout << "[RAISR ERROR] HashTable format is not compatible in number of hash keys!\n";
        std::cout << hashkeySize << std::endl;
        return RNLErrorBadParameter;
    }

    if (pixelTypes != (int)gRatio * (int)gRatio)
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
    filterBuffer = new (std::align_val_t(64)) DT[aligned_rows * hashkeySize * pixelTypes];
    DT *AFilters = &filterBuffer[0];
    memset(AFilters, 0, sizeof(DT) * aligned_rows * hashkeySize * pixelTypes);

    // DT ==2
    int num = 0;
    if (sizeof(DT) == weight_type_size) {
        for (int i = 0 ; i < hashkeySize * pixelTypes; ++i) {
            DT *currentfilter = &AFilters[i * aligned_rows];
            filterFile.read(reinterpret_cast<char*>(currentfilter), sizeof(DT) * rows);
            filterBuckets[i / pixelTypes][i % pixelTypes] = currentfilter;
        }
    } else {
        if (weight_type_size == 4) {
            float weight = 0.0;
            for (int i = 0 ; i < hashkeySize * pixelTypes; ++i) {
                DT *currentfilter = &AFilters[i * aligned_rows];
                for (int j = 0; j < rows; ++j) {
                    filterFile.read(reinterpret_cast<char*>(&weight), weight_type_size);
                    currentfilter[j] = (DT)weight;
                }
                filterBuckets[i / pixelTypes][i % pixelTypes] = currentfilter;
            }
	 } else {
                std::cout << "[RAISR ERROR] hashtable corrupted: " << hashtablePath << std::endl;
                return RNLErrorBadParameter;
        }
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
            QStr.push_back((DT) std::stod(line));
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
            QCoh.push_back((DT) std::stod(line));
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
                out[(startRow + r) * outImageCols / sizeof(unsigned short) + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
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
                out[(startRow + r) * outImageCols / sizeof(unsigned short) + c] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
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

__m256i inline modulo_imm( __m256i a, int b) {

    __m256i div_epi32 = _mm256_cvtps_epi32(_mm256_floor_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(a), _mm256_set1_ps(1.0f / b))));
    return _mm256_sub_epi32(a, _mm256_mullo_epi32(div_epi32, _mm256_set1_epi32(b)));
}

void inline write_pixeltype(int c, __m256i gPatchMargin_epi32, __m256i partone, int* out) {
                        __m256i b = _mm256_sub_epi32( _mm256_add_epi32(_mm256_set1_epi32(c), _mm256_setr_epi32(0,1,2,3,4,5,6,7)), gPatchMargin_epi32);
                        __m256i pixelType_epi32 = _mm256_add_epi32( partone, modulo_imm(b, gRatio) );
                        _mm256_storeu_epi32(out, pixelType_epi32);
}

RNLERRORTYPE processSegment(VideoDataType *srcY, VideoDataType *final_outY, BlendingMode blendingMode, int threadIdx)
{
    VideoDataType *inY;
    VideoDataType *outY;
    int pix_bytes = int((gBitDepth + 7) / 8);

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
        const int step = outY->width * pix_bytes;

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
                Ipp8u *pSrc = inY->pData + inY->step * (int)(startRow / gRatio);
                status = IPPResize(8)(pSrc, inY->step, pDst, step, {0, 0}, {(int)outY->width, segRows},
                                      ippBorderRepl, 0, gIppCtx.specY[threadIdx], gIppCtx.pbufferY[threadIdx]);
            }
            else
            {
                Ipp16u *pSrc = (Ipp16u *)(inY->pData + inY->step * (int)(startRow / gRatio));
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

            if (step == inY->step) {
                memcpy(pDst, pSrc8u, inY->step * segRows);
            } else {
                for (int i = 0; i < segRows; ++i)
                    memcpy(pDst + step * i, pSrc8u + inY->step * i, step);
            }
        }
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            if (gBitDepth == 8)
                Convert_8u16f_8bit(pDst, (_Float16*)pSeg32f, cols, segRows);
            else
                Convert_8u16f_10bit((Ipp16u *)pDst, (_Float16*)pSeg32f, cols, segRows);
        } else
#endif
        {
        if (gBitDepth == 8)
            ippiConvert_8u32f_C1R(pDst, cols,
                                  pSeg32f, cols * sizeof(float), {(int)cols, segRows});
        else
            ippiConvert_16u32f_C1R((Ipp16u *)pDst, step,
                                  pSeg32f, cols * sizeof(float), {(int)cols, segRows});
        }

        // 2. Run hashing
        // Update startRow, endRow for hashing algo
        startRow = gIppCtx.segZones[passIdx][threadIdx].raisrStartRow;
        endRow = gIppCtx.segZones[passIdx][threadIdx].raisrEndRow;

        // Handle top and bottom borders
        if (startRow == 0)
        {
            // it needs to do memcpy line by line when the line size of outY->pData is not equal to pDst's line size.
            if (step == outY->step) {
                memcpy(outY->pData, pDst, outY->step * gLoopMargin + gLoopMargin * pix_bytes);
            } else {
                for (int i = 0; i < gLoopMargin; i++) {
                     memcpy(outY->pData + i * outY->step, pDst + i * step, step);
                }
                memcpy(outY->pData + gLoopMargin * outY->step, pDst + gLoopMargin * step, gLoopMargin * pix_bytes);
            }
        }
        if (endRow == rows)
        {
            if (step == outY->step) {
                memcpy(outY->pData + (rows - gLoopMargin) * step - gLoopMargin * pix_bytes,
                       pDst + (segRows - gLoopMargin) * step - gLoopMargin * pix_bytes,
                       outY->step * gLoopMargin + gLoopMargin * pix_bytes);
            } else {
                memcpy(outY->pData + (rows - gLoopMargin - 1) * outY->step +  (outY->width - gLoopMargin) * pix_bytes,
                       pDst + (segRows - gLoopMargin) * step - gLoopMargin * pix_bytes,
                       gLoopMargin * pix_bytes);

                for (int i = gLoopMargin; i > 0; i--) {
                     memcpy(outY->pData + (rows - i) * outY->step,
                            pDst  + (segRows - i) * step,
                            step);
                }
            }
        }
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            memcpy(pRaisr32f, pSeg32f, sizeof(_Float16) * cols * segRows);
        } else
#endif
        {
            memcpy(pRaisr32f, pSeg32f, sizeof(float) * cols * segRows);
        }

        startRow = startRow < gLoopMargin ? gLoopMargin : startRow;
        endRow = endRow > (rows - gLoopMargin) ? (rows - gLoopMargin) : endRow;

        alignas(64) float GTWG[3][16];
        alignas(64) int pixelType[unrollSizePatchBased];
        alignas(64) int hashValue[unrollSizePatchBased];
        alignas(64) const float *fbase[unrollSizePatchBased];
        alignas(64) float pixbuf[unrollSizePatchBased][128] __attribute__((aligned(64)));
        int pix;
        int census = 0;
        memset(pixelType, 0, sizeof(int) * unrollSizePatchBased);

#ifdef __AVX512FP16__
        alignas(64) _Float16 GTWG_fp16[3][32];
        alignas(64) const _Float16 *fbase_fp16[unrollSizePatchBased];
        alignas(64) _Float16 pixbuf_fp16[unrollSizePatchBased][128] __attribute__((aligned(64)));
#endif

        memset(pixbuf, 0, sizeof(float) * unrollSizePatchBased * 128);
        // NOTE: (r, c) is coordinate in the full HR image, which represents area processed by RAISR
        for (int r = startRow; r < endRow; r++)
        {
            // update coordinate: convert r to coordinate in seg buffer pSeg32f
            // NOTE: use rOffset with pSeg32f
            int rOffset = r - gIppCtx.segZones[passIdx][threadIdx].scaleStartRow;
            int loopItr = unrollSizePatchBased;
            //for (int c = gLoopMargin; c + loopItr <= cols - gLoopMargin; c += loopItr)
            int c = gLoopMargin;
            while (c + loopItr <= cols - gLoopMargin)
            {
               if (gUsePixelType)
                {
#ifdef __AVX512F__
                    __m256i pixelType_epi32;
                    if (loopItr == 8 || loopItr == 16 || loopItr == 32) {
                        // partone = a % gRatio * gRatio
                        __m256i gRatio_epi32 = _mm256_set1_epi32(gRatio);
                        __m256i gPatchMargin_epi32 = _mm256_set1_epi32(gPatchMargin);
                        __m256i a = _mm256_sub_epi32( _mm256_set1_epi32(r), gPatchMargin_epi32);
                        __m256i partone = _mm256_mullo_epi32( modulo_imm( a, gRatio), gRatio_epi32);
                        // parttwo = (b % gRatio)
                        write_pixeltype(c, gPatchMargin_epi32, partone, pixelType);
                        if (loopItr >= 16) {
                            write_pixeltype(c+8, gPatchMargin_epi32, partone, pixelType+8);
                        }
                        if (loopItr == 32) {
                            write_pixeltype(c+16, gPatchMargin_epi32, partone, pixelType+16);
                            write_pixeltype(c+24, gPatchMargin_epi32, partone, pixelType+24);
                        } 
                    }
                    else 
#endif
                    {
                        for (pix = 0; pix < loopItr; pix++)
                        {
                            pixelType[pix] = ((r - gPatchMargin) % (int)gRatio) * (int)gRatio + ((c + pix - gPatchMargin) % (int)gRatio);
                        }
                    }
                }

                for (pix = 0; pix < loopItr / 2; pix++)
                {
                    if (gAsmType == AVX2)
                        computeGTWG_Segment_AVX256_32f(pSeg32f, rows, cols, rOffset, c + 2 * pix, GTWG, pix, &pixbuf[2 * pix][0], &pixbuf[2 * pix + 1][0]);
#ifdef __AVX512F__
                    else if (gAsmType == AVX512)
                        computeGTWG_Segment_AVX512_32f(pSeg32f, rows, cols, rOffset, c + 2 * pix, GTWG, pix, &pixbuf[2 * pix][0], &pixbuf[2 * pix + 1][0]);
#endif
#ifdef __AVX512FP16__
                    else if (gAsmType == AVX512_FP16)
                        if (pix < loopItr / 4) {
                            computeGTWG_Segment_AVX512FP16_16f((_Float16 *)pSeg32f, rows, cols, rOffset, c + 4 * pix, GTWG_fp16, pix, &pixbuf_fp16[4 * pix][0], &pixbuf_fp16[4 * pix + 1][0], &pixbuf_fp16[4 * pix + 2][0], &pixbuf_fp16[4 * pix + 3][0]);
                        } else { break; }
 #endif
                    else
                    {
                        std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                        return RNLErrorBadParameter;
                    }
                }

#ifdef __AVX512FP16__
                if (gAsmType == AVX512_FP16) {
                    if (loopItr == 8) {
                        GetHashValue_AVX512FP16_16h_8Elements(GTWG_fp16, passIdx, hashValue);
                    }  
                    else if (loopItr == 32) { 
                        GetHashValue_AVX512FP16_16h_32Elements(GTWG_fp16, passIdx, hashValue);
                    } else {
                        std::cout << "unsupported loopItr value: " << loopItr << std::endl;
                        return RNLErrorBadParameter;
                    }
                } else
#endif
                {
                    if (loopItr == 8) {
                        GetHashValue_AVX256_32f_8Elements(GTWG, passIdx, hashValue); // 8 elements
                    }
#ifdef __AVX512F__
                    else if (loopItr == 16) {
                        GetHashValue_AVX512_32f_16Elements(GTWG, passIdx, hashValue); // 16 elements 
                    } 
#endif
                    else {
                        std::cout << "unsupported loopItr value: " << loopItr << std::endl;
                        return RNLErrorBadParameter;
                    }
                }

                for (pix = 0; pix < loopItr; pix++)
                {
#ifdef __AVX512FP16__
                    if (passIdx == 0 && gAsmType == AVX512_FP16)
                        fbase_fp16[pix] = gFilterBuckets_fp16[hashValue[pix]][pixelType[pix]];
                    else if (passIdx == 1 && gAsmType == AVX512_FP16)
                        fbase_fp16[pix] = gFilterBuckets2_fp16[hashValue[pix]][pixelType[pix]];
                    else
#endif
                    if (passIdx == 0)
                        fbase[pix] = gFilterBuckets[hashValue[pix]][pixelType[pix]];
                    else
                        fbase[pix] = gFilterBuckets2[hashValue[pix]][pixelType[pix]];
                }

                for (pix = 0; pix < loopItr; pix++)
                {
                    if (likely(c + pix < cols - gLoopMargin))
                    {
                        float curPix;
#ifdef __AVX512FP16__
                        _Float16 curPix_fp16;
#endif

                        if (gAsmType == AVX2)
                            curPix  = DotProdPatch_AVX256_32f(pixbuf[pix], fbase[pix]);
#ifdef __AVX512F__
                        else if (gAsmType == AVX512)
                            curPix  = DotProdPatch_AVX512_32f(pixbuf[pix], fbase[pix]);
#endif
#ifdef __AVX512FP16__
                        else if (gAsmType == AVX512_FP16)
                            curPix_fp16 = (float) DotProdPatch_AVX512FP16_16f(pixbuf_fp16[pix], fbase_fp16[pix]);
 #endif
                        else 
                        {
                            std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                            return RNLErrorBadParameter;
                        }
#ifdef __AVX512FP16__
                        if (gAsmType == AVX512_FP16) {
                            if ((gBitDepth == 8 && curPix_fp16 > gMin8bit && curPix_fp16 < gMax8bit) ||
                                (gBitDepth != 8 && curPix_fp16 > gMin16bit && curPix_fp16 < gMax16bit))
                                ((_Float16*)pRaisr32f)[rOffset * cols + c + pix] = curPix_fp16;
                            else
                                curPix_fp16 = ((_Float16*)pSeg32f)[rOffset * cols + c + pix];
                        } else
#endif
                        {
                        if ((gBitDepth == 8 && curPix > gMin8bit && curPix < gMax8bit) ||
                            (gBitDepth != 8 && curPix > gMin16bit && curPix < gMax16bit))
                            pRaisr32f[rOffset * cols + c + pix] = curPix;
                        else
                            curPix = pSeg32f[rOffset * cols + c + pix];
                        }

                        // CT-Blending, CTRandomness
                        if (blendingMode == Randomness)
                        {
                            if (gAsmType == AVX2)
                                census = CTRandomness_AVX256_32f(pSeg32f, cols, rOffset, c, pix);
#ifdef __AVX512F__
                            else if (gAsmType == AVX512)
                                census = CTRandomness_AVX512_32f(pSeg32f, cols, rOffset, c, pix);
#endif
#ifdef __AVX512FP16__
                            else if (gAsmType == AVX512_FP16)
                                census = CTRandomness_AVX512FP16_16f((_Float16*)pSeg32f, cols, rOffset, c, pix);
#endif
                            else
                            {
                                std::cout << "expected avx512 or avx2, but got " << gAsmType << std::endl;
                                return RNLErrorBadParameter;
                            }

                            float weight = (float)census / (float)CTnumberofPixel;
                            // position in the whole image: r * cols + c + pix
                            float val;
#ifdef __AVX512FP16__
                            if (gAsmType == AVX512_FP16)
                                val = weight * curPix_fp16 + (1 - weight) * ((_Float16*)pSeg32f)[rOffset * cols + c + pix];
                            else
#endif
                            val = weight * curPix + (1 - weight) * pSeg32f[rOffset * cols + c + pix];

                            val += 0.5; // to round the value
                            if (gBitDepth == 8)
                            {
                                outY->pData[r * outY->step + c + pix] = (unsigned char)(val < gMin8bit ? gMin8bit : (val > gMax8bit ? gMax8bit : val));
                            }
                            else
                            {
                                unsigned short *out = (unsigned short *)outY->pData;
                                out[r * outY->step / sizeof(unsigned short) + c + pix] = (unsigned short)(val < gMin16bit ? gMin16bit : (val > gMax16bit ? gMax16bit : val));
                            }
                        }
                    }
                }

                // corner case: we reached near edge of the image
                if ((loopItr > 8) && (c + 2 * unrollSizePatchBased > cols - gLoopMargin)) {
                    loopItr = 8;
                }
                c += loopItr;
            }
	    int unprocessed = cols - c; // unprocessed pixels on the right
	    // Copy right border pixels for this row and left border pixels for next row
	    if (step == outY->step) {
		memcpy(outY->pData + r * step - unprocessed * pix_bytes, pDst + rOffset * step - unprocessed * pix_bytes, (unprocessed + gLoopMargin) * pix_bytes);
            } else {
		// copy right border pixels for this row
                memcpy(outY->pData + (r -1 ) * outY->step + (outY->width - unprocessed) * pix_bytes,
                       pDst + rOffset * step - unprocessed * pix_bytes,
                       unprocessed * pix_bytes);
		// copy left border pixels for nex row
                memcpy(outY->pData + r * outY->step,
                       pDst + rOffset * step,
                       gLoopMargin * pix_bytes);
            }
        }
        // 3. Run CT-Blending
        if (blendingMode == CountOfBitsChanged)
        {
            int segStart = gIppCtx.segZones[passIdx][threadIdx].scaleStartRow;
            if (gAsmType != AVX512_FP16)
                CTCountOfBitsChangedSegment_AVX256_32f(pSeg32f, pRaisr32f, segRows, segStart, {gIppCtx.segZones[passIdx][threadIdx].blendingStartRow, gIppCtx.segZones[passIdx][threadIdx].blendingEndRow}, outY->pData, cols, outY->step);
#ifdef __AVX512FP16__
            else
                CTCountOfBitsChangedSegment_AVX512FP16_16f((_Float16*)pSeg32f, (_Float16*)pRaisr32f, segRows, segStart, {gIppCtx.segZones[passIdx][threadIdx].blendingStartRow, gIppCtx.segZones[passIdx][threadIdx].blendingEndRow}, outY->pData, cols, outY->step);
#endif
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
        !inY || !inY->pData || !outY || !outY->pData)
        return RNLErrorBadParameter;

#ifdef ENABLE_RAISR_OPENCL
    RNLERRORTYPE ret = RNLErrorNone;
    int nbComponent;
    if ((!inCb || !inCb->pData) && (!outCb || !outCb->pData))
        nbComponent = 2;
    else if (inCb && inCb->pData && outCb && outCb->pData)
        nbComponent = 1;
    else
        return RNLErrorBadParameter;

    if (gAsmType == OpenCL) {
        ret = RaisrOpenCLProcessY(&gOpenCLContext, inY->pData, inY->width, inY->height, inY->step,
                                  outY->pData, outY->step, inY->bitShift, blendingMode);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Process src Y failed." << std::endl;
            return RNLErrorUndefined;
        }
        ret = RaisrOpenCLProcessUV(&gOpenCLContext, inCr->pData, inCr->width, inCr->height, inCr->step,
                                   outCr->pData, outCr->step, inCr->bitShift, nbComponent);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Process src Cr failed." << std::endl;
            return RNLErrorUndefined;
        }
        if (nbComponent == 1) {
            ret = RaisrOpenCLProcessUV(&gOpenCLContext, inCb->pData, inCb->width, inCb->height, inCb->step,
                                       outCb->pData, outCb->step, inCb->bitShift, nbComponent);
            if (ret != RNLErrorNone) {
                std::cout << "[RAISR OPENCL ERROR] Process src Cb failed." << std::endl;
                return RNLErrorUndefined;
            }
        }
        return ret;
    } else if (gAsmType == OpenCLExternal) {
        ret = RaisrOpenCLProcessImageY(&gOpenCLContext, (cl_mem)inY->pData, inY->width, inY->height,
                                       (cl_mem)outY->pData, inY->bitShift, blendingMode);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Process clImage Y failed." << std::endl;
            return RNLErrorUndefined;
        }
        ret = RaisrOpenCLProcessImageUV(&gOpenCLContext, (cl_mem)inCr->pData, inCr->width, inCr->height,
                                        (cl_mem)outCr->pData, inCr->bitShift, nbComponent);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Process clImage Cr failed." << std::endl;
            return RNLErrorUndefined;
        }
        if (nbComponent == 1) {
            ret = RaisrOpenCLProcessImageUV(&gOpenCLContext, (cl_mem)inCb->pData, inCb->width, inCb->height,
                                            (cl_mem)outCb->pData, inCb->bitShift, nbComponent);
            if (ret != RNLErrorNone) {
                std::cout << "[RAISR OPENCL ERROR] Process clImage Cb failed." << std::endl;
                return RNLErrorUndefined;
        }
        }
        return ret;
    }
#endif

    if (!inCb || !inCb->pData || !outCb ||!outCb->pData)
        return RNLErrorBadParameter;

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

    return RNLErrorNone;
}

RNLERRORTYPE RNLSetOpenCLContext(void *context, void *deviceID, int platformIndex, int deviceIndex) {
#ifdef ENABLE_RAISR_OPENCL
    gOpenCLContext.context = (cl_context)context;
    gOpenCLContext.deviceID = (cl_device_id)deviceID;
    gOpenCLContext.platformIndex = platformIndex;
    gOpenCLContext.deviceIndex = deviceIndex;
#endif
    return RNLErrorNone;
}

RNLERRORTYPE RNLInit(std::string &modelPath,
                     float ratio,
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

    gMachineVendorType = get_machine_vendor();
    if (gMachineVendorType == VENDOR_UNSUPPORTED)
    {
        std::cout << "[RAISR ERROR] Only supported on x86 (Intel, AMD) platforms. " << std::endl;
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
    if (gRatio != 2.0)
    {
        gUsePixelType = false;
    }
#ifdef __AVX512FP16__
    if ( gAsmType > AVX512_FP16 || gAsmType < AVX2 ) gAsmType = AVX512_FP16;
#else
#ifdef __AVX512F__
    if ( gAsmType != AVX512 && gAsmType != AVX2 &&
         gAsmType != OpenCL && gAsmType != OpenCLExternal) gAsmType = AVX512;
#else
    if ( gAsmType != AVX2 && gAsmType != OpenCL && gAsmType != OpenCLExternal) gAsmType = AVX2;
#endif
#endif
#ifdef __AVX512FP16__
    if ( gAsmType == AVX512_FP16) {
        if (machine_supports_feature(gMachineVendorType, AVX512_FP16)) {
            std::cout << "ASM Type: AVX512FP16\n";
            unrollSizePatchBased = 32; // for FP16, increase unrollSizePatchBased to 32 to use full 512bit registers
        } else {
            std::cout << "ASM Type: AVX512FP16 requested, but machine does not support it.  Changing to AVX512\n";
            gAsmType = AVX512;
        }
    }
#endif
#ifdef __AVX512F__
    if ( gAsmType == AVX512) {
        if (machine_supports_feature(gMachineVendorType, AVX512)) {
            std::cout << "ASM Type: AVX512\n";
            unrollSizePatchBased = 16; // for F32AVX512, increase unrollSizePatchBased to 16 to use full 512bit registers
        } else {
            std::cout << "ASM Type: AVX512 requested, but machine does not support it.  Changing to AVX2\n";
            gAsmType = AVX2;
        }
    }
#endif
    if (gAsmType == OpenCL || gAsmType == OpenCLExternal) {
#ifdef ENABLE_RAISR_OPENCL
        std::cout << "ASM Type: OpenCL\n";
#else
        std::cout << "ASM Type: OpenCL requested, but OpenCL is not enabled.\n";
        return RNLErrorBadParameter;
#endif
    }
    if (gAsmType == AVX2) {
        if (machine_supports_feature(gMachineVendorType, AVX2)) {
            std::cout << "ASM Type: AVX2\n";
        } else {
            std::cout << "ASM Type: AVX2 requested, but machine does not support it.\n";
            return RNLErrorBadParameter;
        }
    }
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
    if (RNLErrorNone != RNLStoi(&gQuantizationAngle, configTokens[0].c_str(), configPath))
    {
        return RNLErrorBadParameter;
    }
    gQAngle = gQuantizationAngle / PI;
    if (RNLErrorNone != RNLStoi(&gQuantizationStrength, configTokens[1].c_str(), configPath))
    {
        return RNLErrorBadParameter;
    }
    if (RNLErrorNone != RNLStoi(&gQuantizationCoherence, configTokens[2].c_str(), configPath))
    {
        return RNLErrorBadParameter;
    }

    // Varify hashtable file format
    if (RNLErrorNone != RNLStoi(&gPatchSize, configTokens[3].c_str(), configPath))
    {
        return RNLErrorBadParameter;
    }
    if (gPatchSize != 11)
    {
        std::cout << "[RAISR ERROR] configFile corrupted: " << configPath << std::endl;
        return RNLErrorBadParameter;
    }
    gPatchMargin = gPatchSize >> 1;
    gLoopMargin = (gPatchSize >> 1) + 1;
    gResizeExpand = (gLoopMargin);
    unsigned int patchAreaSize = gPatchSize * gPatchSize;
    g64AlinedgPatchAreaSize = ((patchAreaSize + 64 - 1) / 64) * 64;
    configFile.close();

#ifdef __AVX512FP16__
    if (gAsmType == AVX512_FP16) {
        if (RNLErrorNone != ReadTrainedData<_Float16>(hashtablePath, QStrPath, QCohPath, 1 /*first pass*/, gFilterBuckets_fp16, &gFilterBuffer_fp16, gQStr_fp16, gQCoh_fp16))
            return RNLErrorBadParameter;
        if (gPasses == 2 && RNLErrorNone != ReadTrainedData<_Float16>(std::move(hashtablePath), std::move(QStrPath), std::move(QCohPath), 2 /*second pass*/, gFilterBuckets2_fp16, &gFilterBuffer2_fp16, gQStr2_fp16, gQCoh2_fp16))
            return RNLErrorBadParameter;
    } else
#endif
    {
    if (RNLErrorNone != ReadTrainedData<float>(hashtablePath, QStrPath, QCohPath, 1 /*first pass*/, gFilterBuckets, &gFilterBuffer, gQStr, gQCoh))
        return RNLErrorBadParameter;

    if (gPasses == 2 && RNLErrorNone != ReadTrainedData<float>(std::move(hashtablePath), std::move(QStrPath), std::move(QCohPath), 2 /*second pass*/, gFilterBuckets2, &gFilterBuffer2, gQStr2, gQCoh2))
        return RNLErrorBadParameter;
    }

    // create guassian kernel if patchSize is not default
    if (gPatchSize != defaultPatchSize)
    {
        float *kernel;
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            kernel = new float[gPatchSize/2];
            createGaussianKernel<_Float16>(gPatchSize, sigma, (_Float16*)kernel);
            gPGaussian = new float[patchAreaSize]; // 2 x n^2 array
        } else
#endif
        {
            kernel = new float[gPatchSize];
            createGaussianKernel<float>(gPatchSize, sigma, kernel);
            gPGaussian = new float[patchAreaSize * 2]; // 2 x n^2 array
        }
        // compute kernel * kernel.t() ==> n x n
        for (int rowkernel = 0; rowkernel < gPatchSize; rowkernel++)
        {
            for (int colkernel = 0; colkernel < gPatchSize; colkernel++)
            {
#ifdef __AVX512FP16__
                if (gAsmType == AVX512_FP16) {
                    ((_Float16*)gPGaussian)[rowkernel * gPatchSize + colkernel] = ((_Float16*)kernel)[rowkernel] * ((_Float16*)kernel)[colkernel];
                } else
#endif
                {
                    gPGaussian[rowkernel * gPatchSize + colkernel] = kernel[rowkernel] * kernel[colkernel];
                }
            }
        }
        // append
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            memcpy((_Float16*)gPGaussian + patchAreaSize, (_Float16*)gPGaussian, patchAreaSize * sizeof(_Float16));
        } else
#endif
        {
            memcpy(gPGaussian + patchAreaSize, gPGaussian, patchAreaSize * sizeof(float));
        }
        delete[] kernel;
    }

    threadCount = (threadCount == 0) ? 1 : threadCount;
    // multi-threaded patch-based approach
    gThreadCount = threadCount;
    gPool = new ThreadPool(gThreadCount);

#ifdef ENABLE_RAISR_OPENCL
    RNLERRORTYPE err;
    if (gAsmType == OpenCLExternal && (!gOpenCLContext.context || !gOpenCLContext.deviceID)) {
        std::cout << "[RAISR OPENCL ERROR] When use OpenCLExternal mode,"
                     "RNLSetExternalOpenCLContext() should be called before init." << std::endl;
        return RNLErrorBadParameter;
    }
    gOpenCLContext.gRatio = gRatio;
    gOpenCLContext.gAsmType = gAsmType;
    gOpenCLContext.gBitDepth = gBitDepth;
    gOpenCLContext.gQuantizationAngle = gQuantizationAngle;
    gOpenCLContext.gQuantizationStrength = gQuantizationStrength;
    gOpenCLContext.gQuantizationCoherence = gQuantizationCoherence;
    gOpenCLContext.gPatchSize = gPatchSize;
    gOpenCLContext.gQStr = gQStr;
    gOpenCLContext.gQCoh = gQCoh;
    gOpenCLContext.gQStr2 = gQStr2;
    gOpenCLContext.gQCoh2 = gQCoh2;
    gOpenCLContext.gFilterBuffer = gFilterBuffer;
    gOpenCLContext.gFilterBuffer2 = gFilterBuffer2;
    gOpenCLContext.gPasses = gPasses;
    gOpenCLContext.gTwoPassMode = gTwoPassMode;
    gOpenCLContext.gMin8bit = gMin8bit;
    gOpenCLContext.gMax8bit = gMax8bit;
    gOpenCLContext.gMin16bit = gMin16bit;
    gOpenCLContext.gMax16bit = gMax16bit;
    gOpenCLContext.gUsePixelType = (int)gUsePixelType;
    if (gAsmType == OpenCL || gAsmType == OpenCLExternal)
        if ((err = RaisrOpenCLInit(&gOpenCLContext)) != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Init Raisr OpenCL error." << std::endl;
            return err;
        }
#endif

    return RNLErrorNone;
}

RNLERRORTYPE RNLSetRes(VideoDataType *inY, VideoDataType *inCr, VideoDataType *inCb,
                       VideoDataType *outY, VideoDataType *outCr, VideoDataType *outCb)
{
    int rows, cols, step;
    IppStatus status = ippStsNoErr;
#ifdef ENABLE_RAISR_OPENCL
    if (gAsmType == OpenCL || gAsmType == OpenCLExternal) {
        RNLERRORTYPE ret = RNLErrorNone;
        int nbComponent = 1;
        if (!inCb->pData)
            nbComponent = 2;

        ret = RaisrOpenCLSetRes(&gOpenCLContext, inY->width, inY->height,
                                inCr->width, inCr->height, nbComponent);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Set resolution error." << std::endl;
            return ret;
        } else
            return RNLErrorNone;
    }
#endif

    if (gPasses == 2 && gTwoPassMode == 2)
    {
        rows = inY->height;
        cols = inY->width;
        step = inY->step;
    }
    else
    {
        rows = outY->height;
        cols = outY->width;
        step = outY->step;
    }

    if (gPasses == 2)
    {
        gIntermediateY = new VideoDataType;
        gIntermediateY->pData = new unsigned char[step * rows];
        gIntermediateY->step = step;
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
        int rowsPerThread = ceil(float(rows) / gThreadCount);
        rowsPerThread -= rowsPerThread % 2;
        int rowsOfRemainder = rows - rowsPerThread * gThreadCount;

        for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
        {
            threadStatus[threadIdx] = 0;
            if (threadIdx != gThreadCount - 1) // some middle threads cover 1 more row, having loopsOfRemainder divided evenly.
            {
                endRow = (threadIdx < rowsOfRemainder / 2) ? (startRow + rowsPerThread + 2) : (startRow + rowsPerThread);
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
            int startR = (startRow - expand) < 0 ? 0 : (startRow - expand);
            int endR = (endRow + expand) > rows ? rows : (endRow + expand);

            if (startRow == 0)
                while((endR / gRatio) - (int)(endR / gRatio) && endR < rows) endR++;
            else if (endRow == rows)
                while((startR / gRatio) - (int)(startR / gRatio) && startR > 0) startR--;
            else 
            {
                while((startR / gRatio) - (int)(startR / gRatio) && startR > 0) startR--;
                while((endR / gRatio) - (int)(endR / gRatio) && endR < rows) endR++;
            }
            gIppCtx.segZones[i][threadIdx].scaleStartRow = startR;
            gIppCtx.segZones[i][threadIdx].scaleEndRow = endR;
            // std::cout << threadIdx << ", blendingStartRow: " << gIppCtx.segZones[i][threadIdx].blendingStartRow << ", blendingEndRow: " << gIppCtx.segZones[i][threadIdx].blendingEndRow
            //<< ", raisrStartRow: " << gIppCtx.segZones[i][threadIdx].raisrStartRow << ", raisrEndRow: " << gIppCtx.segZones[i][threadIdx].raisrEndRow
            //<< ", scaleStartRow: " << gIppCtx.segZones[i][threadIdx].scaleStartRow << ", scaleEndRow: " << gIppCtx.segZones[i][threadIdx].scaleEndRow << std::endl;

            segHeight = gIppCtx.segZones[i][threadIdx].scaleEndRow - gIppCtx.segZones[i][threadIdx].scaleStartRow;

            // NOTE: the intermediate buffer is to hold resize tile
            if (gBitDepth == 8)
                gIppCtx.segZones[i][threadIdx].inYUpscaled = new Ipp8u[segHeight * cols];
            else
                gIppCtx.segZones[i][threadIdx].inYUpscaled = new Ipp8u[segHeight * cols * BYTES_16BITS];
            if (gAsmType == AVX512_FP16) {
                gIppCtx.segZones[i][threadIdx].inYUpscaled32f = new (std::align_val_t(64)) float[segHeight * cols / 2];
                gIppCtx.segZones[i][threadIdx].raisr32f = new (std::align_val_t(64)) float[segHeight * cols / 2];
            } else {
                gIppCtx.segZones[i][threadIdx].inYUpscaled32f = new (std::align_val_t(64)) float[segHeight * cols];
                gIppCtx.segZones[i][threadIdx].raisr32f = new (std::align_val_t(64)) float[segHeight * cols];
            }

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

#define SAFE_DELETE(x) {  \
       if (x)            \
           delete x;     \
        x = NULL;         \
}
#define SAFE_ARR_DELETE(x) { \
       if (x != NULL)       \
           delete[] x;      \
        x = NULL;            \
}

RNLERRORTYPE RNLDeinit()
{
#ifdef ENABLE_RAISR_OPENCL
    if (gAsmType == OpenCL || gAsmType == OpenCLExternal) {
        RaisrOpenCLRelease(&gOpenCLContext);
	SAFE_DELETE(gPool);
	if (gPasses == 2)
	    SAFE_ARR_DELETE(gFilterBuffer2);

	SAFE_ARR_DELETE(gFilterBuffer);
	SAFE_ARR_DELETE(gPGaussian);
	return RNLErrorNone;
    }
#endif

    for (int threadIdx = 0; threadIdx < gThreadCount; threadIdx++)
    {
	if (gIppCtx.specY[threadIdx])
	    ippsFree(gIppCtx.specY[threadIdx]);
        if (gIppCtx.pbufferY[threadIdx])
	    ippsFree(gIppCtx.pbufferY[threadIdx]);
        for (int i = 0; i < gPasses; i++)
        {
            SAFE_ARR_DELETE(gIppCtx.segZones[i][threadIdx].inYUpscaled);
            SAFE_ARR_DELETE(gIppCtx.segZones[i][threadIdx].inYUpscaled32f);
            SAFE_ARR_DELETE(gIppCtx.segZones[i][threadIdx].raisr32f);
        }
    }
    SAFE_DELETE(gPool);
    SAFE_ARR_DELETE(gIppCtx.segZones[0]);

    SAFE_ARR_DELETE(gIppCtx.specY);
    SAFE_ARR_DELETE(gIppCtx.pbufferY);

    if (gIppCtx.specUV)
        ippsFree(gIppCtx.specUV);
    if (gIppCtx.pbufferUV)
        ippsFree(gIppCtx.pbufferUV);

    if (gPasses == 2)
    {
        SAFE_ARR_DELETE(gIppCtx.segZones[1]);
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            SAFE_ARR_DELETE(gFilterBuffer2_fp16);
        } else
#endif
        {
        SAFE_ARR_DELETE(gFilterBuffer2);
        }
        if (gIntermediateY)
	    SAFE_ARR_DELETE(gIntermediateY->pData);
        SAFE_DELETE(gIntermediateY);
    }
#ifdef __AVX512FP16__
        if (gAsmType == AVX512_FP16) {
            SAFE_ARR_DELETE(gFilterBuffer_fp16);
        } else
#endif
        {
            SAFE_ARR_DELETE(gFilterBuffer);
        }
    SAFE_ARR_DELETE(gPGaussian);

    return RNLErrorNone;
}
