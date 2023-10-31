/**
 * Intel Library for Video Super Resolution
 *
 * Copyright (c) 2023 Intel Corporation
 * SPDX-License-Identifier: BSD-3-Clause
 */
#include "Raisr_OpenCL.h"
#include "Raisr_OpenCL_kernel.h"
#include <stdio.h>
#include <string.h>
#include <algorithm>
#include <math.h>

#include <iostream>
#include <vector>
#include <algorithm>

typedef struct RaisrModel {
    cl_program program;
    cl_command_queue queue;
    cl_kernel filterKernel;
    cl_kernel gradientKernel;
    cl_kernel blendKernel;
    cl_kernel preprocessKernel;
    cl_kernel postprocessKernel;
    cl_kernel hashKernel;
    cl_mem gaussianW;
    cl_mem filterBuckets;
    std::vector<float> qStr;
    std::vector<float> qCoh;
} RaisrModel;

typedef struct RaisrContextOpenCLPriv {
    cl_mem outputFilter;
    cl_mem outputBlend;
    cl_mem imgGx;
    cl_mem imgGy;
    cl_mem outputMemY;
    cl_mem inputImageY, outputImageY;
    cl_mem inputImageUV, outputImageUV;
    cl_command_queue queue;
    RaisrModel raisrModels[5];
} RaisrContextOpenCLPriv;

static void initGaussianKernel(float *kernel, float sigma, int kernelSize)
{
    double scale2X = -0.5/(sigma*sigma);
    double sum = 0;

    int i;
    for( i = 0; i < kernelSize; i++ )
    {
        double x = i - (kernelSize-1)*0.5;
        double t = exp(scale2X*x*x);
        kernel[i] = (float)t;
        sum += kernel[i];
    }

    sum = 1./sum;
    for( i = 0; i < kernelSize; i++ )
        kernel[i] = (float)(kernel[i]*sum);

    return;
}

static float *readGaussianMatrix(int gradientLength, int bitDepth)
{
    float *gk = NULL;
    float colorRange = (1 << bitDepth) - 1;
    gk = (float *)malloc(gradientLength * sizeof(float));
    if (!gk)
        return NULL;
    float *gaussianMatrixRaw = (float *)malloc(gradientLength*gradientLength*sizeof(float));
    if (!gaussianMatrixRaw) {
        free(gk);
        return NULL;
    }
    initGaussianKernel(gk, 2.0f, gradientLength);
    for (int y = 0; y < gradientLength; y++)
        for (int x = 0; x < gradientLength; x++)
                gaussianMatrixRaw[y * gradientLength + x] = gk[y] * gk[x] / colorRange / colorRange / 2.0f / 2.0f;

    free(gk);
    return gaussianMatrixRaw;
}

static RNLERRORTYPE buildProgram(RaisrOpenCLContext *raisrOpenCLContext,
                                 RaisrModel *raisrModel, cl_context context,
                                 cl_device_id deviceID, const char *modelPath)
{
    FILE *programHandle = NULL;
    char *programBuffer = NULL, *programLog = NULL;
    size_t programSize, logSize;
    RNLERRORTYPE ret = RNLErrorNone;
    int err, i, colorRangeMin, colorRangeMax;
    std::string strengthList, coherenceList;
    char *filterShader, *mallocFilterShader = NULL;

    for (i = 0; i < raisrModel->qStr.size() - 1; i++)
        strengthList += std::to_string(raisrModel->qStr[i]) + ",";
    strengthList += std::to_string(raisrModel->qStr[i]);
    for (i = 0; i < raisrModel->qCoh.size() - 1; i++)
        coherenceList += std::to_string(raisrModel->qCoh[i]) + ",";
    coherenceList += std::to_string(raisrModel->qCoh[i]);
    colorRangeMin = raisrOpenCLContext->gBitDepth == 8 ? raisrOpenCLContext->gMin8bit : raisrOpenCLContext->gMin16bit;
    colorRangeMax = raisrOpenCLContext->gBitDepth == 8 ? raisrOpenCLContext->gMax8bit : raisrOpenCLContext->gMax16bit;

    /* Read program file and place content into buffer */
    if (modelPath != nullptr) {
        programHandle = fopen(modelPath, "r");
        if(programHandle == NULL) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't find the kernel file" << std::endl;
            return RNLErrorBadParameter;
        }
        fseek(programHandle, 0, SEEK_END);
        err = ftell(programHandle);
        if (err <= 0) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't get the kernel file size" << std::endl;
            ret = RNLErrorBadParameter;
            goto fail;
        }
        programSize = err;
        rewind(programHandle);
        mallocFilterShader = (char*)malloc(programSize + 1);
        if (!mallocFilterShader) {
            ret = RNLErrorInsufficientResources;
            goto fail;
        }
        filterShader = mallocFilterShader;
        filterShader[programSize] = '\0';
        err = fread(filterShader, sizeof(char), programSize, programHandle);
        if (err < 0) {
            ret = RNLErrorBadParameter;
            goto fail;
        }
        fclose(programHandle);
        programHandle = NULL;
    } else
        filterShader = (char *)gFilterShader;

    programSize = strlen(filterShader);
    programBuffer = (char*)malloc(programSize + 2048);
    if (!programBuffer) {
         err = RNLErrorInsufficientResources;
         goto fail;
    }
    sprintf(programBuffer, filterShader, (int)raisrOpenCLContext->gRatio, raisrOpenCLContext->gRatio,
            raisrOpenCLContext->gUsePixelType,
            raisrOpenCLContext->gPatchSize, raisrOpenCLContext->gPatchSize,
            raisrOpenCLContext->gQuantizationAngle, raisrOpenCLContext->gQuantizationStrength,
            raisrOpenCLContext->gQuantizationCoherence, colorRangeMin, colorRangeMax,
            strengthList.c_str(), coherenceList.c_str());
    programSize = strlen(programBuffer);
    raisrModel->program = clCreateProgramWithSource(context, 1,
        (const char**)&programBuffer, &programSize, &err);
    if(err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create the kernel. "
                     "OpenCL error code: " << err << std::endl;
        ret = RNLErrorUndefined;
        goto fail;
    }

    err = clBuildProgram(raisrModel->program, 0, NULL, NULL, NULL, NULL);
    if(err != CL_SUCCESS) {
        /* Find size of log and print to std output */
        clGetProgramBuildInfo(raisrModel->program, deviceID, CL_PROGRAM_BUILD_LOG,
            0, NULL, &logSize);
        programLog = (char*) malloc(logSize + 1);
        if (!programLog) {
            ret = RNLErrorInsufficientResources;
            goto fail;
        }
        programLog[logSize] = '\0';
        clGetProgramBuildInfo(raisrModel->program, deviceID, CL_PROGRAM_BUILD_LOG,
            logSize + 1, programLog, NULL);
        printf("%s\n", programLog);
        ret = RNLErrorUndefined;
        goto fail;
    }
    int err_tmp;
    raisrModel->filterKernel = clCreateKernel(raisrModel->program, "filter", &err_tmp);
    err |= err_tmp;
    raisrModel->gradientKernel = clCreateKernel(raisrModel->program, "gradient", &err_tmp);
    err |= err_tmp;
    raisrModel->blendKernel = clCreateKernel(raisrModel->program, "blend", &err_tmp);
    err |= err_tmp;
    raisrModel->preprocessKernel = clCreateKernel(raisrModel->program, "preprocess", &err_tmp);
    err |= err_tmp;
    raisrModel->postprocessKernel = clCreateKernel(raisrModel->program, "postprocess", &err_tmp);
    err |= err_tmp;
    raisrModel->hashKernel = clCreateKernel(raisrModel->program, "hash_mul", &err_tmp);
    err |= err_tmp;
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create kernel. "
                     "OpenCL error code: " << err << std::endl;
        ret = RNLErrorUndefined;
    }

fail:
    if (mallocFilterShader)
        free(mallocFilterShader);
    if (programHandle)
        fclose(programHandle);
    if (programLog)
        free(programLog);
    if (programBuffer)
        free(programBuffer);
    return ret;
}

RNLERRORTYPE RaisrOpenCLInit(RaisrOpenCLContext *raisrOpenCLContext)
{
    int err = 0, filterSetSize = 0;
    float *pGaussianW = NULL;
    RNLERRORTYPE ret = RNLErrorNone;
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = NULL;
    int platformIndex = raisrOpenCLContext->platformIndex;
    int deviceIndex = raisrOpenCLContext->deviceIndex;
    if (raisrOpenCLContext->gTwoPassMode != 1 && raisrOpenCLContext->gTwoPassMode != 0) {
            std::cout << "[RAISR OPENCL ERROR] Raisr OpenCL only support twoPassMode == 1" << std::endl;
            return RNLErrorBadParameter;
    }
    if (raisrOpenCLContext->gAsmType == OpenCL) {
        cl_platform_id *platform;
        cl_device_id *device;
        cl_uint nbPlatform, nbDevice;
        if((err = clGetPlatformIDs(0, NULL, &nbPlatform)) < 0) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't get number of OpenCL platform. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        platform = (cl_platform_id *)malloc(nbPlatform * sizeof(*platform));
        if (!platform) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't alloocate memory for OpenCL platform. " << std::endl;
            return RNLErrorInsufficientResources;
        }
        if((err = clGetPlatformIDs(nbPlatform, platform, NULL)) < 0) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't identify a platform. "
                         "OpenCL error code: " << err << std::endl;
            free(platform);
            return RNLErrorUndefined;
        }
        err = clGetDeviceIDs(platform[platformIndex], CL_DEVICE_TYPE_ALL, 0, NULL, &nbDevice);
        if(err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't get number of devices. "
                         "OpenCL error code: " << err << std::endl;
            free(platform);
            return RNLErrorUndefined;
        }
        device = (cl_device_id *)malloc(nbDevice * sizeof(*device));
        if (!device) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't alloocate memory for OpenCL device. " << std::endl;
            free(platform);
            return RNLErrorInsufficientResources;
        }
        err = clGetDeviceIDs(platform[platformIndex], CL_DEVICE_TYPE_ALL, nbDevice, device, NULL);
        if(err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't access any devices. "
                         "OpenCL error code: " << err << std::endl;
            free(platform);
            free(device);
            return RNLErrorUndefined;
        }
        raisrOpenCLContext->deviceID = device[deviceIndex];
        free(platform);
        free(device);
        raisrOpenCLContext->context = clCreateContext(NULL, 1, &raisrOpenCLContext->deviceID, NULL, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create OpenCL context. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
    } else if (raisrOpenCLContext->gAsmType == OpenCLExternal) {
        if ((clRetainContext(raisrOpenCLContext->context)) < 0) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't retain context. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        if ((clRetainDevice(raisrOpenCLContext->deviceID)) < 0) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't retain deviceID. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
    }
    raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)malloc(sizeof(RaisrContextOpenCLPriv));
    if (!raisrContextOpenCLPriv)
        return RNLErrorInsufficientResources;
    memset(raisrContextOpenCLPriv, 0, sizeof(RaisrContextOpenCLPriv));
    raisrOpenCLContext->priv = (void *)raisrContextOpenCLPriv;
    raisrContextOpenCLPriv->queue = clCreateCommandQueueWithProperties(raisrOpenCLContext->context,
                                                                       raisrOpenCLContext->deviceID, NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create OpenCL queue. "
                        "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    for (int i = 0; i < raisrOpenCLContext->gPasses; i++) {
        RaisrModel *raisrModel = &raisrContextOpenCLPriv->raisrModels[i];
        float *filterBuffer;
        if (i == 0) {
            filterBuffer = raisrOpenCLContext->gFilterBuffer;
            raisrModel->qStr = raisrOpenCLContext->gQStr;
            raisrModel->qCoh = raisrOpenCLContext->gQCoh;
        } else {
            filterBuffer = raisrOpenCLContext->gFilterBuffer2;
            raisrModel->qStr = raisrOpenCLContext->gQStr2;
            raisrModel->qCoh = raisrOpenCLContext->gQCoh2;
        }
        ret = buildProgram(raisrOpenCLContext, raisrModel, raisrOpenCLContext->context,
                           raisrOpenCLContext->deviceID, NULL);
        if (ret != RNLErrorNone) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't build kernel" << std::endl;
            return ret;
        }
        raisrModel->queue =
            clCreateCommandQueueWithProperties(raisrOpenCLContext->context,
                                               raisrOpenCLContext->deviceID,
                                               NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create queue. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        int pixelTypes = raisrOpenCLContext->gUsePixelType ?
                raisrOpenCLContext->gRatio * raisrOpenCLContext->gRatio : 1;
        filterSetSize = raisrOpenCLContext->gQuantizationAngle * raisrOpenCLContext->gQuantizationStrength * raisrOpenCLContext->gQuantizationCoherence *
                          pixelTypes * raisrOpenCLContext->gPatchSize * raisrOpenCLContext->gPatchSize;
        raisrModel->filterBuckets = clCreateBuffer(raisrOpenCLContext->context,
            CL_MEM_READ_ONLY, filterSetSize*sizeof(float), NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create filterBuckets buffer. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        raisrModel->gaussianW = clCreateBuffer(raisrOpenCLContext->context,
            CL_MEM_READ_ONLY, raisrOpenCLContext->gPatchSize * raisrOpenCLContext->gPatchSize * sizeof(float), NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create gaussian buffer. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }

        int rows = raisrOpenCLContext->gPatchSize * raisrOpenCLContext->gPatchSize;
        int alignedRows = 16 * (int)((rows + 15) / 16);
        float *AFilters = &filterBuffer[0];
        int hashkeySize = raisrOpenCLContext->gQuantizationAngle *
                          raisrOpenCLContext->gQuantizationStrength *
                          raisrOpenCLContext->gQuantizationCoherence;
        for (int i = 0; i < hashkeySize * pixelTypes; i++) {
            err = clEnqueueWriteBuffer(raisrModel->queue, raisrModel->filterBuckets, CL_TRUE, i * rows * sizeof(float),
                                       rows * sizeof(float), AFilters + i * alignedRows,
                                       0, NULL, NULL);
            if (err != CL_SUCCESS) {
                std::cout << "[RAISR OPENCL ERROR] Couldn't write filterBuckets. "
                            "OpenCL error code: " << err << std::endl;
                return RNLErrorUndefined;
            }
        }

        pGaussianW = readGaussianMatrix(raisrOpenCLContext->gPatchSize, raisrOpenCLContext->gBitDepth);
        if (!pGaussianW) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create gaussian matrix. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        err = clEnqueueWriteBuffer(raisrModel->queue, raisrModel->gaussianW, CL_TRUE, 0,
                                   raisrOpenCLContext->gPatchSize * raisrOpenCLContext->gPatchSize * sizeof(float),
                                   pGaussianW, 0, NULL, NULL);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't write gaussianW. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        free(pGaussianW);
    }
    return RNLErrorNone;
}

RNLERRORTYPE RaisrOpenCLSetRes(RaisrOpenCLContext *raisrOpenCLContext, int widthY, int heightY,
                               int widthUV, int heightUV, int nbComponent)
{
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    raisrOpenCLContext->widthMax = widthY;
    raisrOpenCLContext->heightMax = heightY;
    int LR_width = widthY;
    int LR_height = heightY;
    int HR_width = widthY * raisrOpenCLContext->gRatio;
    int HR_height = heightY * raisrOpenCLContext->gRatio;
    int err = 0;

    raisrContextOpenCLPriv->outputMemY = clCreateBuffer(raisrOpenCLContext->context, CL_MEM_READ_WRITE, HR_width*HR_height*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create outputMemY buffer. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    raisrContextOpenCLPriv->imgGx = clCreateBuffer(raisrOpenCLContext->context, CL_MEM_READ_WRITE, HR_width*HR_height*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create imgGx buffer. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    raisrContextOpenCLPriv->imgGy = clCreateBuffer(raisrOpenCLContext->context, CL_MEM_READ_WRITE, HR_width*HR_height*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create imgGy buffer. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    raisrContextOpenCLPriv->outputFilter = clCreateBuffer(raisrOpenCLContext->context, CL_MEM_READ_WRITE, HR_width*HR_height*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create outputFilter buffer. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    raisrContextOpenCLPriv->outputBlend = clCreateBuffer(raisrOpenCLContext->context, CL_MEM_READ_WRITE, HR_width*HR_height*sizeof(float), NULL, &err);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't create outputBlend buffer. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3];
    cl_image_format imageFormat = { 0 };
    cl_image_desc   imageDesc = { 0 };

    if (raisrOpenCLContext->gAsmType == OpenCL) {
        if (raisrOpenCLContext->gBitDepth > 8)
            imageFormat.image_channel_data_type = CL_UNORM_INT16;
        else
            imageFormat.image_channel_data_type = CL_UNORM_INT8;
        imageFormat.image_channel_order = CL_R;
        imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
        imageDesc.image_width = LR_width;
        imageDesc.image_height = LR_height;
        imageDesc.image_row_pitch = 0;
        raisrContextOpenCLPriv->inputImageY = clCreateImage(raisrOpenCLContext->context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create inputImageY. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        imageDesc.image_width = HR_width;
        imageDesc.image_height = HR_height;
        imageDesc.image_row_pitch = 0;
        raisrContextOpenCLPriv->outputImageY = clCreateImage(raisrOpenCLContext->context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create outputImageY. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }

        if (nbComponent == 1)
            imageFormat.image_channel_order = CL_R;
        else
            imageFormat.image_channel_order = CL_RG;
        imageDesc.image_type = CL_MEM_OBJECT_IMAGE2D;
        imageDesc.image_width = widthUV;
        imageDesc.image_height = heightUV;
        imageDesc.image_row_pitch = 0;
        raisrContextOpenCLPriv->inputImageUV = clCreateImage(raisrOpenCLContext->context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create inputImageUV. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
        imageDesc.image_width = widthUV * raisrOpenCLContext->gRatio;
        imageDesc.image_height = heightUV * raisrOpenCLContext->gRatio;
        imageDesc.image_row_pitch = 0;
        raisrContextOpenCLPriv->outputImageUV = clCreateImage(raisrOpenCLContext->context, CL_MEM_READ_WRITE, &imageFormat, &imageDesc, NULL, &err);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Couldn't create outputImageUV. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }
    }

    return RNLErrorNone;
}

static int preprocess(RaisrOpenCLContext *raisrOpenCLContext,
                      cl_mem *input, int widthI, int heightI, int linesize,
                      cl_mem *output, int widthO, int heightO,
                      int linesizeO, int bitShift, int nbComponet) {
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    RaisrModel *raisrModel = &raisrContextOpenCLPriv->raisrModels[0];
    int err;
    float widthFactor, heightFactor;
    size_t imageWidth, imageHeight;
    err = clGetImageInfo(*input, CL_IMAGE_WIDTH,  sizeof(size_t),
                         &imageWidth, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't query image width." << std::endl;
        return err;
    }
    err = clGetImageInfo(*input, CL_IMAGE_HEIGHT, sizeof(size_t),
                         &imageHeight, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't query image height." << std::endl;
        return err;
    }
    widthFactor = 1.0f / (1.0f * imageWidth * widthO / widthI);
    heightFactor = 1.0f / (1.0f * imageHeight * heightO / heightI);
    err = clSetKernelArg(raisrModel->preprocessKernel, 0, sizeof(cl_mem), input);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 1, sizeof(int), &widthO);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 2, sizeof(int), &heightO);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 3, sizeof(int), &linesize);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 4, sizeof(cl_mem), output);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 5, sizeof(int), &linesizeO);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 6, sizeof(float), &widthFactor);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 7, sizeof(float), &heightFactor);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 8, sizeof(int), &bitShift);
    err |= clSetKernelArg(raisrModel->preprocessKernel, 9, sizeof(int), &nbComponet);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for preprocess." << std::endl;
        return err;
    }

    size_t globalSize[2];
    globalSize[0] = widthO;
    globalSize[1] = heightO;
    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->preprocessKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] preprocess failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }
    clFinish(raisrModel->queue);

    return 0;
}

static int postprocess(RaisrOpenCLContext *raisrOpenCLContext,
                       cl_mem *input, int width, int height, int linesize,
                       cl_mem *output, int linesizeO, int bitShift, int nbComponet) {
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    RaisrModel *raisrModel = &raisrContextOpenCLPriv->raisrModels[0];
    int err;
    err = clSetKernelArg(raisrModel->postprocessKernel, 0, sizeof(cl_mem), input);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 1, sizeof(int), &width);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 2, sizeof(int), &height);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 3, sizeof(int), &linesize);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 4, sizeof(cl_mem), output);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 5, sizeof(int), &linesizeO);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 6, sizeof(int), &bitShift);
    err |= clSetKernelArg(raisrModel->postprocessKernel, 7, sizeof(int), &nbComponet);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for postprocess." << std::endl;
        return err;
    }

    size_t globalSize[2];
    globalSize[0] = width;
    globalSize[1] = height;
    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->postprocessKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] postprocess failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }

    clFinish(raisrModel->queue);

    return 0;
}

static int gradient(RaisrOpenCLContext *raisrOpenCLContext , RaisrModel *raisrModel,
                    int width, int height, int linesize, cl_mem *cl_mem1) {
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    int err;
    size_t localSize;
    err = clSetKernelArg(raisrModel->gradientKernel, 0, sizeof(cl_mem), cl_mem1);
    err |= clSetKernelArg(raisrModel->gradientKernel, 1, sizeof(cl_mem), &raisrContextOpenCLPriv->imgGx);
    err |= clSetKernelArg(raisrModel->gradientKernel, 2, sizeof(cl_mem), &raisrContextOpenCLPriv->imgGy);
    err |= clSetKernelArg(raisrModel->gradientKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(raisrModel->gradientKernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(raisrModel->gradientKernel, 5, sizeof(int), &linesize);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for gradient." << std::endl;
        return err;
    }

    size_t globalSize[2];
    globalSize[0] = width;
    globalSize[1] = height;
    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->gradientKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] gradient failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }
    clFinish(raisrModel->queue);

    return 0;
}

static int filter(RaisrOpenCLContext *raisrOpenCLContext , RaisrModel *raisrModel,
                  int width, int height, int linesize, cl_mem *cl_mem1, cl_mem *cl_mem2) {
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    int err;
    err = clSetKernelArg(raisrModel->filterKernel, 0, sizeof(cl_mem), cl_mem2); // <=====OUTPUT
    err |= clSetKernelArg(raisrModel->filterKernel, 1, sizeof(cl_mem), &raisrContextOpenCLPriv->imgGx);
    err |= clSetKernelArg(raisrModel->filterKernel, 2, sizeof(cl_mem), &raisrContextOpenCLPriv->imgGy);
    err |= clSetKernelArg(raisrModel->filterKernel, 3, sizeof(cl_mem), &raisrModel->gaussianW);
    err |= clSetKernelArg(raisrModel->filterKernel, 4, sizeof(int), &width);
    err |= clSetKernelArg(raisrModel->filterKernel, 5, sizeof(int), &height);
    err |= clSetKernelArg(raisrModel->filterKernel, 6, sizeof(int), &linesize);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for filter." << std::endl;
        return err;
    }
    size_t globalSize[2];
    globalSize[0] = width/2;
    globalSize[1] = height/2;
    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->filterKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] filter failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }

    clFinish(raisrModel->queue);

    err = clSetKernelArg(raisrModel->hashKernel, 0, sizeof(cl_mem), cl_mem1);
    err |= clSetKernelArg(raisrModel->hashKernel, 1, sizeof(cl_mem), cl_mem2);
    err |= clSetKernelArg(raisrModel->hashKernel, 2, sizeof(cl_mem), &raisrModel->filterBuckets);
    err |= clSetKernelArg(raisrModel->hashKernel, 3, sizeof(cl_mem), cl_mem2);
    err |= clSetKernelArg(raisrModel->hashKernel, 4, sizeof(int), &width);
    err |= clSetKernelArg(raisrModel->hashKernel, 5, sizeof(int), &height);
    err |= clSetKernelArg(raisrModel->hashKernel, 6, sizeof(int), &linesize);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for hash." << std::endl;
        return err;
    }

    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->hashKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] hash failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }

    clFinish(raisrModel->queue);

    return 0;
}

static int blend(RaisrModel *raisrModel, int width, int height, int linesize,
                 cl_mem *cl_mem1, cl_mem *cl_mem2, cl_mem *cl_mem3) {
    int blend_type = 2;
    int err;
    err = clSetKernelArg(raisrModel->blendKernel, 0, sizeof(cl_mem), cl_mem1);
    err |= clSetKernelArg(raisrModel->blendKernel, 1, sizeof(cl_mem), cl_mem2);
    err |= clSetKernelArg(raisrModel->blendKernel, 2, sizeof(cl_mem), cl_mem3);
    err |= clSetKernelArg(raisrModel->blendKernel, 3, sizeof(int), &width);
    err |= clSetKernelArg(raisrModel->blendKernel, 4, sizeof(int), &height);
    err |= clSetKernelArg(raisrModel->blendKernel, 5, sizeof(int), &linesize);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't set args for blend." << std::endl;
        return err;
    }

    size_t globalSize[2];
    globalSize[0] = width;
    globalSize[1] = height;
    err = clEnqueueNDRangeKernel(raisrModel->queue, raisrModel->blendKernel, 2, NULL, globalSize, NULL, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] blend failed on clEnqueueNDRangeKernel." << std::endl;
        return err;
    }

    clFinish(raisrModel->queue);

    return 0;
}

RNLERRORTYPE RaisrOpenCLProcessY(RaisrOpenCLContext *raisrOpenCLContext,
                                 uint8_t *inputY, int width, int height, int linesizeI,
                                 uint8_t *outputY, int linesizeO, int bitShift, BlendingMode blend)
{
    int HR_width = width * raisrOpenCLContext->gRatio;
    int HR_height = height * raisrOpenCLContext->gRatio;
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    RNLERRORTYPE ret;
    int err;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3];

    region[0] = width;
    region[1] = height;
    region[2] = 1;
    err = clEnqueueWriteImage(raisrContextOpenCLPriv->queue, raisrContextOpenCLPriv->inputImageY, CL_TRUE, origin, region,
                              linesizeI, 0, inputY, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't write inputImageY. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    ret = RaisrOpenCLProcessImageY(raisrOpenCLContext, raisrContextOpenCLPriv->inputImageY, width, height,
                                 raisrContextOpenCLPriv->outputImageY, bitShift, blend);
    if (ret != RNLErrorNone) {
        std::cout << "[RAISR OPENCL ERROR] Process clImage Y failed, in RaisrOpenCLProcessY." << std::endl;
        return ret;
    }
    region[0] = HR_width;
    region[1] = HR_height;
    region[2] = 1;
    err = clEnqueueReadImage(raisrContextOpenCLPriv->queue, raisrContextOpenCLPriv->outputImageY, CL_TRUE, origin, region,
                             linesizeO, 0, outputY, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't write outputImageY. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    return ret;
}

RNLERRORTYPE RaisrOpenCLProcessUV(RaisrOpenCLContext *raisrOpenCLContext,
                                  uint8_t *inputUV, int width, int height,
                                  int linesizeI, uint8_t *outputUV, int linesizeO,
                                  int bitShift, int nbComponent)
{
    int HR_width = width * raisrOpenCLContext->gRatio;
    int HR_height = height * raisrOpenCLContext->gRatio;
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    RNLERRORTYPE ret;
    int err;
    size_t origin[3] = { 0, 0, 0 };
    size_t region[3];

    region[0] = width;
    region[1] = height;
    region[2] = 1;
    err = clEnqueueWriteImage(raisrContextOpenCLPriv->queue, raisrContextOpenCLPriv->inputImageUV, CL_TRUE, origin, region,
                              linesizeI, 0, inputUV, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't write inputImageUV. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    ret = RaisrOpenCLProcessImageUV(raisrOpenCLContext, raisrContextOpenCLPriv->inputImageUV, width, height,
                            raisrContextOpenCLPriv->outputImageUV, bitShift, nbComponent);
    if (ret != RNLErrorNone) {
        std::cout << "[RAISR OPENCL ERROR] Process clImage UV failed, in RaisrOpenCLProcessUV." << std::endl;
        return ret;
    }
    region[0] = HR_width;
    region[1] = HR_height;
    region[2] = 1;
    err = clEnqueueReadImage(raisrContextOpenCLPriv->queue, raisrContextOpenCLPriv->outputImageUV, CL_TRUE, origin, region,
                                linesizeO, 0, outputUV, 0, NULL, NULL);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Couldn't write outputImageUV. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    return ret;
}

RNLERRORTYPE RaisrOpenCLProcessImageY(RaisrOpenCLContext* raisrOpenCLContext,
                                      cl_mem inputImageY, int LR_width, int LR_height,
                                      cl_mem outputImageY, int bitShift, BlendingMode blendMode)
{
    int HR_width = LR_width * raisrOpenCLContext->gRatio;
    int HR_height = LR_height * raisrOpenCLContext->gRatio;
    RNLERRORTYPE ret;
    int err, modelIn = 0;
    int passes = 0;
    cl_mem *finall_mem, *cl_mem1, *cl_mem2, *cl_mem3, *cl_tmp;
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    if (blendMode == Randomness) {
        std::cout << "[RAISR OPENCL ERROR] Raisr OpenCL only support CountOfBitsChanged blend." << std::endl;
        return RNLErrorBadParameter; 
    }
    err = preprocess(raisrOpenCLContext, &inputImageY, LR_width, LR_height, LR_width,
                     &raisrContextOpenCLPriv->outputMemY, HR_width, HR_height, HR_width,
                     bitShift, 1);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Run kernel func preprocess failed. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    cl_mem1 = &raisrContextOpenCLPriv->outputMemY;
    cl_mem2 = &raisrContextOpenCLPriv->outputFilter;
    cl_mem3 = &raisrContextOpenCLPriv->outputBlend;
    while (passes < raisrOpenCLContext->gPasses) {
        RaisrModel *raisrModel = &raisrContextOpenCLPriv->raisrModels[passes];
        err = gradient(raisrOpenCLContext, raisrModel, HR_width, HR_height, HR_width, cl_mem1);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Run kernel func gradient failed. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }

        err = filter(raisrOpenCLContext, raisrModel, HR_width, HR_height, HR_width, cl_mem1, cl_mem2);
        if (err != CL_SUCCESS) {
            std::cout << "[RAISR OPENCL ERROR] Run kernel func filter failed. "
                         "OpenCL error code: " << err << std::endl;
            return RNLErrorUndefined;
        }

        if (blendMode) {
            err = blend(raisrModel, HR_width, HR_height, HR_width, cl_mem1, cl_mem2, cl_mem3);
            if (err != CL_SUCCESS) {
                std::cout << "[RAISR OPENCL ERROR] Run kernel func blend failed. "
                            "OpenCL error code: " << err << std::endl;
                return RNLErrorUndefined;
            }
            finall_mem = cl_mem3;
            cl_tmp = cl_mem1;
            cl_mem1 = cl_mem3;
            cl_mem3 = cl_tmp;
        } else {
            finall_mem = cl_mem2;
            cl_tmp = cl_mem1;
            cl_mem1 = cl_mem2;
            cl_mem2 = cl_tmp;
        }
        passes += 1;
    }
    err = postprocess(raisrOpenCLContext, finall_mem, HR_width, HR_height, HR_width,
                      &outputImageY, HR_width, bitShift, 1);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Run kernel func postprocess failed. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    return RNLErrorNone;
}

RNLERRORTYPE RaisrOpenCLProcessImageUV(RaisrOpenCLContext* raisrOpenCLContext,
                                       cl_mem inputImageUV, int width, int height, cl_mem outputImageUV,
                                       int bitShift, int nbComponent)
{
    int LR_width = width;
    int LR_height = height;
    int HR_width = width * raisrOpenCLContext->gRatio;
    int HR_height = height * raisrOpenCLContext->gRatio;
    int err;
    RNLERRORTYPE ret;
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;

    err = preprocess(raisrOpenCLContext, &inputImageUV, LR_width, LR_height, LR_width,
                     &raisrContextOpenCLPriv->outputMemY, HR_width, HR_height, HR_width, bitShift, nbComponent);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Run kernel func preprocess failed. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    err = postprocess(raisrOpenCLContext, &raisrContextOpenCLPriv->outputMemY, HR_width, HR_height, HR_width,
                      &outputImageUV, HR_width, bitShift, nbComponent);
    if (err != CL_SUCCESS) {
        std::cout << "[RAISR OPENCL ERROR] Run kernel func postprocess failed. "
                     "OpenCL error code: " << err << std::endl;
        return RNLErrorUndefined;
    }
    return RNLErrorNone;
}

void RaisrOpenCLRelease(RaisrOpenCLContext *raisrOpenCLContext)
{
    RaisrContextOpenCLPriv *raisrContextOpenCLPriv = (RaisrContextOpenCLPriv *)raisrOpenCLContext->priv;
    if (!raisrContextOpenCLPriv)
        return;
    if (!raisrContextOpenCLPriv->outputMemY)
        clReleaseMemObject(raisrContextOpenCLPriv->outputMemY);
    if (!raisrContextOpenCLPriv->outputFilter)
        clReleaseMemObject(raisrContextOpenCLPriv->outputFilter);
    if (!raisrContextOpenCLPriv->outputBlend)
        clReleaseMemObject(raisrContextOpenCLPriv->outputBlend);
    if (!raisrContextOpenCLPriv->imgGx)
        clReleaseMemObject(raisrContextOpenCLPriv->imgGx);
    if (!raisrContextOpenCLPriv->imgGy)
        clReleaseMemObject(raisrContextOpenCLPriv->imgGy);
    if (!raisrContextOpenCLPriv->inputImageY)
        clReleaseMemObject(raisrContextOpenCLPriv->inputImageY);
    if (!raisrContextOpenCLPriv->inputImageUV)
        clReleaseMemObject(raisrContextOpenCLPriv->inputImageUV);
    if (!raisrContextOpenCLPriv->outputImageY)
        clReleaseMemObject(raisrContextOpenCLPriv->outputImageY);
    if (!raisrContextOpenCLPriv->outputImageUV)
        clReleaseMemObject(raisrContextOpenCLPriv->outputImageUV);
    for (int i = 0; i < raisrOpenCLContext->gPasses; i++) {
        RaisrModel *raisrModel = &raisrContextOpenCLPriv->raisrModels[i];
        if (!raisrModel->filterKernel)
            clReleaseKernel(raisrModel->filterKernel);
        if (!raisrModel->gradientKernel)
            clReleaseKernel(raisrModel->gradientKernel);
        if (!raisrModel->blendKernel)
            clReleaseKernel(raisrModel->blendKernel);
        if (!raisrModel->preprocessKernel)
            clReleaseKernel(raisrModel->preprocessKernel);
        if (!raisrModel->postprocessKernel)
            clReleaseKernel(raisrModel->postprocessKernel);
        if (!raisrModel->hashKernel)
            clReleaseKernel(raisrModel->hashKernel);
        if (!raisrModel->filterBuckets)
            clReleaseMemObject(raisrModel->filterBuckets);
        if (!raisrModel->gaussianW)
            clReleaseMemObject(raisrModel->gaussianW);
        if (!raisrModel->queue)
            clReleaseCommandQueue(raisrModel->queue);
        if (!raisrModel->program)
            clReleaseProgram(raisrModel->program);
    }
    if (!raisrOpenCLContext ->context)
        clReleaseContext(raisrOpenCLContext ->context);
    if (!raisrOpenCLContext ->deviceID)
        clReleaseDevice(raisrOpenCLContext ->deviceID);
    free(raisrOpenCLContext ->priv);
    raisrOpenCLContext ->priv = NULL;
}
