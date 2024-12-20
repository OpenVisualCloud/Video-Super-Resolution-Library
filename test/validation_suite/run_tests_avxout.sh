#!/bin/bash

# SPDX-License-Identifier: BSD-3-Clause
# Copyright 2024-2025 Intel Corporation

set -x

SCRIPT_DIR="$(readlink -f "$(dirname -- "${BASH_SOURCE[0]}")")"
REPO_DIR="$(readlink -f "${SCRIPT_DIR}/../..")"
LD_LIBRARY_PATH="/usr/local/lib:${LD_LIBRARY_PATH}"

ffmpeg_path="${ffmpeg_path:-${REPO_DIR}/raisr/ffmpeg}"
test_video_path="${test_video_path:-${ffmpeg_path}/test_videos}"

avx_array=("avx2")
if lscpu | grep "avx512f " | grep "avx512vl ";
then avx_array+=("avx512");
else echo "No avx512 found";
fi
if lscpu | grep "avx512_fp16 ";
then avx_array+=("avx512fp16");
else echo "No avx512fp16 found";
fi

thread_array=(1 10 20 30 40 50 60 90 120 88)
pass_array=(1 2)
blending_array=(1 2)
mode_array=(1 2)
fixed_thread=120

#bad inputs
wrong_bit=9
wrong_blending=0
wrong_ratio=0
wrong_mode=-1
wrong_thread=121
wrong_pass=3
negtive_pass=-1
negtive_thread=-1
bad_configs=(12_3_3_11 24_3_3 24_3_3_6 24_3_3_9)
bad_nums=(bad_cohpath_nums bad_hashtable_nums bad_strpath_nums)
no_pathes=(noCohPath noConfig noHashTable noStrPath)
#log path
common_log_path="${ffmpeg_path}/test_logs"
wrong_input_log_path="${ffmpeg_path}/test_bad_input_logs"
opath="${ffmpeg_path}/outputs/"
mkdir -p "${common_log_path}" "${wrong_input_log_path}" "${opath}"

cd $ffmpeg_path
for filename in $test_video_path/*; do
    fname=$(basename $filename)
    ext="${fname##*.}"
    fname1="${fname%.*}"
    ofname="$fname1"_out."$ext"
    for avx in ${avx_array[@]}; do
    #test diff thread &diff avx
        for thread in ${thread_array[@]};do
	    printf -v digi '%02d' ${thread##+(0)}
            ofname="$fname1"_out_th"$digi"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$thread:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff pass &diff avx, thread is fixed to 120
        for pass in ${pass_array[@]};do
	    printf -v digip '%02d' ${pass##+(0)}
            ofname="$fname1"_out_pass"$digip"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:passes=$pass:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff blending &diff avx, thread is fixed to 120
        for blending in ${blending_array[@]};do
        printf -v digib '%02d' ${blending##+(0)}
            ofname="$fname1"_out_bld"$digib"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:blending=$blending:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff mode &diff avx, thread is fixed to 120
        for mode in ${mode_array[@]};do
        printf -v digim '%02d' ${mode##+(0)}
            ofname="$fname1"_out_mode"$digim"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:mode=$mode:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff filter_1 &diff avx, thread is fixed to 120
        for f in filters_1*/*;do
            digif="${f/"/"/"_"}"
            ofname="$fname1"_out_ft"$digif"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:filterfolder=$f:ratio=1.5:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff filter_2 &diff avx, thread is fixed to 120
        for f in filters_2*/*;do
            digif="${f/"/"/"_"}"
            ofname="$fname1"_out_ft"$digif"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:filterfolder=$f:ratio=2.0:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        done

    #test diff bit &diff avx, thread is fixed to 120
        result=$(echo $fname1 | grep "10bit")
        if [[ "$result" != "" ]]
        then
            ofname="$fname1"_out_bit"10"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:bits=10:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        else
            ofname="$fname1"_out_bit"08"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:bits=8:asm=$avx $opath$ofname >$common_log_path/test_log_$ofname.log 2>&1
        fi

    #test wrong inputs
        #bad bits:
        printf -v digi '%02d' ${wrong_bit##+(0)}
        ofname="$fname1"_out_wbit"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:bits=$wrong_bit:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad blending:
        printf -v digi '%02d' ${wrong_blending##+(0)}
        ofname="$fname1"_out_wbld"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:blending=$wrong_blending:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad raito
        printf -v digi '%02d' ${wrong_raito##+(0)}
        ofname="$fname1"_out_wrat"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:radio=$wrong_raito:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad mode
        printf -v digi '%02d' ${wrong_mode##+(0)}
        ofname="$fname1"_out_wmod"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:mode=$wrong_mode:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad thread count
        printf -v digi '%02d' ${wrong_thread##+(0)}
        ofname="$fname1"_out_wth"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$wrong_thread:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad pass
        printf -v digi '%02d' ${wrong_pass##+(0)}
        ofname="$fname1"_out_wps"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:passes=$wrong_pass:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #not matching pass&mode#[RAISR WARNING] 1 pass with upscale in 2d pass, mode = 2 ignored !
        ofname="$fname1"_no_match_pm_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:passes=1:mode=2:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #negtive pass
        printf -v digi '%02d' ${negtive_pass##+(0)}
        ofname="$fname1"_out_nps"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:passes=$negtive_pass:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #negtive thread
        printf -v digi '%02d' ${negtive_thread##+(0)}
        ofname="$fname1"_out_nth"$digi"_"$avx".mp4
        ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$negtive_thread:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #directory_input
        ofname="$fname1"_directory_input_"$avx".mp4
        ./ffmpeg -y -i $test_video_path -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #no_input #Did you mean file:-profile:v?
        ofname="$fname1"_no_input_"$avx".mp4
        ./ffmpeg -y -i -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:asm=$avx $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        #bad config 12 3 3 11|24 3 3|24 3 3 6|24 3 3 9
        for config in ${bad_configs[@]}; do
            ofname="$fname1"_config_"$config"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:asm=$avx:filterfolder=filters_wrongConfig/filters1_$config $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        done
        #bad nums
        for badpath in ${bad_nums[@]}; do
            ofname="$fname1"_"$badpath"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:asm=$avx:filterfolder=filters_badNums/filters1_$badpath $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        done
        #no pathes
        for confpath in ${no_pathes[@]}; do
            ofname="$fname1"_"$confpath"_"$avx".mp4
            ./ffmpeg -y -i $filename -profile:v high -b:v 6M -maxrate 12M -bufsize 24M -crf 28 -vf raisr=threadcount=$fixed_thread:asm=$avx:filterfolder=filters_noPathes/filters1_$confpath $opath$ofname >$wrong_input_log_path/test_log_$ofname.log 2>&1
        done
    done
done

echo "========== Start of summary"
echo "Finished tests:"
echo "Test logs: ${ffmpeg_path}/test_logs"
echo "Test bad input logs: ${ffmpeg_path}/test_bad_input_logs"
echo "Test outputs: ${ffmpeg_path}/outputs/"
echo "Results:"
echo "Should find 0 match per file:"
grep -c failed ${ffmpeg_path}/test_logs/*
echo "Should find 1 match per file:"
grep -Ec "failed|not found|RAISR WARNING|Is a directory" ${ffmpeg_path}/test_bad_input_logs/*
echo "========== End of summary"
