/*
 * Intel Library for Video Super Resolution ffmpeg plugin
 *
 * Copyright (c) 2021 Intel Corporation
 *
 * FFmpeg is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * FFmpeg is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with this program; if not, write to the Free Software
 * Foundation, Inc., 51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA
 */

/**
 * @file
 * Raisr filter
 *
 * @see https://arxiv.org/pdf/1606.01299.pdf
 */

#include "libavutil/avassert.h"
#include "libavutil/imgutils.h"
#include "libavutil/opt.h"
#include "libavutil/pixfmt.h"
#include "avfilter.h"
#include "formats.h"
#include "internal.h"
#include "video.h"
#include "raisr/RaisrHandler.h"
#include "raisr/RaisrDefaults.h"
#include <unistd.h>

#define MIN_RATIO 1
#define MAX_RATIO 4
#define DEFAULT_RATIO 2

#define MIN_THREADCOUNT 1
#define MAX_THREADCOUNT 120
#define DEFAULT_THREADCOUNT 20

#define BLENDING_RANDOMNESS 1
#define BLENDING_COUNT_OF_BITS_CHANGED 2

struct plane_info
{
    int width;
    int height;
    int linesize;
};

typedef struct RaisrContext
{
    const AVClass *class;
    float ratio;
    int bits;
    char *range;
    int threadcount;
    char *filterfolder;
    int blending;
    int passes;
    int mode;
    char *asmStr;
    int platform;
    int device;

    struct plane_info inplanes[3];
    int nb_planes;
    int framecount;
} RaisrContext;

#define OFFSET(x) offsetof(RaisrContext, x)
#define FLAGS AV_OPT_FLAG_FILTERING_PARAM | AV_OPT_FLAG_VIDEO_PARAM
static const AVOption raisr_options[] = {
    {"ratio", "ratio of the upscaling, between 1 and 4", OFFSET(ratio), AV_OPT_TYPE_FLOAT, {.dbl = DEFAULT_RATIO}, MIN_RATIO, MAX_RATIO, FLAGS},
    {"bits", "bit depth", OFFSET(bits), AV_OPT_TYPE_INT, {.i64 = 8}, 8, 10, FLAGS},
    {"range", "input color range", OFFSET(range), AV_OPT_TYPE_STRING, {.str = "video"}, 0, 0, FLAGS},
    {"threadcount", "thread count", OFFSET(threadcount), AV_OPT_TYPE_INT, {.i64 = DEFAULT_THREADCOUNT}, MIN_THREADCOUNT, MAX_THREADCOUNT, FLAGS},
    {"filterfolder", "absolute filter folder path", OFFSET(filterfolder), AV_OPT_TYPE_STRING, {.str = "filters1"}, 0, 0, FLAGS},
    {"blending", "CT blending mode (1: Randomness, 2: CountOfBitsChanged)", OFFSET(blending), AV_OPT_TYPE_INT, {.i64 = BLENDING_COUNT_OF_BITS_CHANGED}, BLENDING_RANDOMNESS, BLENDING_COUNT_OF_BITS_CHANGED, FLAGS},
    {"passes", "passes to run (1: one pass, 2: two pass)", OFFSET(passes), AV_OPT_TYPE_INT, {.i64 = 1}, 1, 2, FLAGS},
    {"mode", "mode for two pass (1: upscale in 1st pass, 2: upscale in 2nd pass)", OFFSET(mode), AV_OPT_TYPE_INT, {.i64 = 1}, 1, 2, FLAGS},
    {"asm", "x86 asm type: (avx512, avx2 or opencl)", OFFSET(asmStr), AV_OPT_TYPE_STRING, {.str = "avx512"}, 0, 0, FLAGS},
    {"platform", "select the platform", OFFSET(platform), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, FLAGS},
    {"device", "select the device", OFFSET(device), AV_OPT_TYPE_INT, {.i64 = 0}, 0, INT_MAX, FLAGS},
    {NULL}};

AVFILTER_DEFINE_CLASS(raisr);

static av_cold int init(AVFilterContext *ctx)
{
    RaisrContext *raisr = ctx->priv;

    char cwd[255];
    if (getcwd(cwd, 255) == NULL)
        return AVERROR(ENOENT);

    char basepath[255];
    strcpy(basepath, cwd);
    if (strcmp(raisr->filterfolder, "") == 0)
    {
        strcat(basepath, "/filters1");
    }
    else
    {
        strcpy(basepath, raisr->filterfolder);
    }

    RangeType rangeType = VideoRange;
    if (strcmp(raisr->range, "full") == 0)
        rangeType = FullRange;

    ASMType asm_t;
    if (strcmp(raisr->asmStr, "avx2") == 0)
        asm_t = AVX2;
    else if (strcmp(raisr->asmStr, "avx512") == 0)
        asm_t = AVX512;
    else if (strcmp(raisr->asmStr, "opencl") == 0)
        asm_t = OpenCL;
    else {
        av_log(ctx, AV_LOG_VERBOSE, "asm field expects avx2 or avx512 but got: %s\n", raisr->asmStr);
        return AVERROR(ENOENT);
    }

    if (asm_t == OpenCL)
    {
        RNLERRORTYPE ret = RNLHandler_SetOpenCLContext(NULL, NULL, raisr->platform, raisr->device);
        if (ret != RNLErrorNone)
        {
            av_log(ctx, AV_LOG_ERROR, "RNLHandler_SetOpenCLContext error\n");
            return AVERROR(ENOMEM);
        }
    }


    RNLERRORTYPE ret = RNLHandler_Init(basepath, raisr->ratio, raisr->bits, rangeType, raisr->threadcount, asm_t, raisr->passes, raisr->mode);

    if (ret != RNLErrorNone)
    {
        av_log(ctx, AV_LOG_VERBOSE, "RNLHandler_Init error\n");
        return AVERROR(ENOMEM);
    }
    raisr->framecount = 0;

    return 0;
}

static int query_formats(AVFilterContext *ctx)
{
    int raisr_fmts[] = {AV_PIX_FMT_YUV420P, AV_PIX_FMT_YUV420P10LE,
                        AV_PIX_FMT_YUV422P, AV_PIX_FMT_YUV422P10LE, AV_PIX_FMT_NONE};
    AVFilterFormats *fmts_list;

    fmts_list = ff_make_format_list(raisr_fmts);
    if (!fmts_list)
    {
        return AVERROR(ENOMEM);
    }
    return ff_set_common_formats(ctx, fmts_list);
}

static int config_props_input(AVFilterLink *inlink)
{
    AVFilterContext *ctx = inlink->dst;
    RaisrContext *raisr = ctx->priv;

    // Return n a pixel format descriptor for provided pixel format or NULL if this pixel format is unknown.
    const AVPixFmtDescriptor *desc = av_pix_fmt_desc_get(inlink->format);

    // Determine the number of planes  (will be 3 except for grayscale)
    raisr->nb_planes = inlink->format == AV_PIX_FMT_GRAY8 ? 1 : 3;

    // for each plane
    for (int p = 0; p < raisr->nb_planes; p++)
    {
        // Get a pointer to the plane info
        struct plane_info *plane = &raisr->inplanes[p];

        // Get horziontal and vertical power of 2 factors
        int vsub = p ? desc->log2_chroma_h : 0;
        int hsub = p ? desc->log2_chroma_w : 0;

        // Determine the width and height of this plane/channel
        plane->width = AV_CEIL_RSHIFT(inlink->w, hsub);
        plane->height = AV_CEIL_RSHIFT(inlink->h, vsub);
        plane->linesize = av_image_get_linesize(inlink->format, plane->width, p);
    }
    return 0;
}

static int config_props_output(AVFilterLink *outlink)
{
    AVFilterContext *ctx = outlink->src;
    RaisrContext *raisr = ctx->priv;
    AVFilterLink *inlink0 = outlink->src->inputs[0];

    outlink->w = inlink0->w * raisr->ratio;
    outlink->h = inlink0->h * raisr->ratio;

    return 0;
}

static int filter_frame(AVFilterLink *inlink, AVFrame *in)
{
    AVFilterContext *ctx = inlink->dst;
    RaisrContext *raisr = ctx->priv;
    AVFilterLink *outlink = ctx->outputs[0];
    AVFrame *out;
    RNLERRORTYPE ret;
    VideoDataType vdt_in[3] = { 0 };
    VideoDataType vdt_out[3] = { 0 };

    av_log(ctx, AV_LOG_VERBOSE, "Frame\n");

    // Request a picture buffer - must be released with. This must be unreferenced with
    // avfilter_unref_buffer when you are finished with it
    out = ff_get_video_buffer(outlink, outlink->w, outlink->h);
    if (!out)
    {
        // Unable to get a picture buffer.
        // Delete the input buffer and return
        av_frame_free(&in);
        return AVERROR(ENOMEM);
    }
    av_log(ctx, AV_LOG_VERBOSE, "Got Frame %dx%d\n", outlink->w, outlink->h);

    // Copy only "metadata" fields from src to dst.
    // Metadata for the purpose of this function are those fields that do not affect
    // the data layout in the buffers.
    av_frame_copy_props(out, in);
    av_log(ctx, AV_LOG_VERBOSE, "Copied props \n");

    // For each plane
    for (int p = 0; p < raisr->nb_planes; p++)
    {
        // get the plane data
        struct plane_info *plane = &raisr->inplanes[p];

        // make sure the input data is valid
        av_assert1(in->data[p]);

        // get a pointer to the out plane data
        av_assert1(out->data[p]);

        // fill in the input video data type structure
        vdt_in[p].pData = in->data[p];
        vdt_in[p].width = plane->width;
        vdt_in[p].height = plane->height;
        vdt_in[p].step = in->linesize[p];

        // fill in the output video data type structure
        vdt_out[p].pData = out->data[p];
        vdt_out[p].width = plane->width * raisr->ratio;
        vdt_out[p].height = plane->height * raisr->ratio;
        vdt_out[p].step = out->linesize[p];
    }
    if (raisr->framecount == 0)
    {
        // Process the planes
        ret = RNLHandler_SetRes(
            &vdt_in[0],
            &vdt_in[1],
            &vdt_in[2],
            &vdt_out[0],
            &vdt_out[1],
            &vdt_out[2]);

        if (ret != RNLErrorNone)
        {
            av_log(ctx, AV_LOG_INFO, "RNLHandler_SetRes error\n");
            return AVERROR(ENOMEM);
        }
    }

    // Process the planes
    ret = RNLHandler_Process(
        &vdt_in[0],
        &vdt_in[1],
        &vdt_in[2],
        &vdt_out[0],
        &vdt_out[1],
        &vdt_out[2],
        raisr->blending);

    if (ret != RNLErrorNone)
    {
        av_log(ctx, AV_LOG_INFO, "RNLHandler_Process error\n");
        return AVERROR(ENOMEM);
    }

    // increment framecount
    raisr->framecount++;

    // Free the input frame
    av_frame_free(&in);

    // ff_filter_frame sends a frame of data to the next filter
    // outlink is the output link over which the data is being sent
    // out is a reference to the buffer of data being sent.
    // The receiving filter will free this reference when it no longer
    // needs it or pass it on to the next filter.
    return ff_filter_frame(outlink, out);
}

static av_cold void uninit(AVFilterContext *ctx)
{
    RNLHandler_Deinit();
}

static const AVFilterPad raisr_inputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props_input,
        .filter_frame = filter_frame,
    },
    {NULL}};

static const AVFilterPad raisr_outputs[] = {
    {
        .name = "default",
        .type = AVMEDIA_TYPE_VIDEO,
        .config_props = config_props_output,
    },
    {NULL}};

AVFilter ff_vf_raisr = {
    .name = "raisr",
    .description = NULL_IF_CONFIG_SMALL("Perform Raisr super resolution."),
    .priv_size = sizeof(RaisrContext),
    .init = init,
    .uninit = uninit,
    .query_formats = query_formats,
    .inputs = raisr_inputs,
    .outputs = raisr_outputs,
    .priv_class = &raisr_class,
    .flags = AVFILTER_FLAG_SUPPORT_TIMELINE_GENERIC,
};
