#!/bin/sh
export LD_LIBRARY_PATH=/data/local/tmp/snpebm/artifacts/aarch64-ubuntu-gcc7.5/lib:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH="/data/local/tmp/snpebm/artifacts/aarch64-ubuntu-gcc7.5/lib/../../dsp/lib;/system/lib/rfsa/adsp;/usr/lib/rfsa/adsp;/system/vendor/lib/rfsa/adsp;/dsp;/etc/images/dsp;"
cd /data/local/tmp/snpebm/YOLOV5S
rm -rf output
/data/local/tmp/snpebm/artifacts/aarch64-ubuntu-gcc7.5/bin/snpe-net-run --container yolov5s.dlc --input_list target_raw_list.txt --output_dir output --use_dsp --userbuffer_tfN 8 --perf_profile high_performance --profiling_level detailed
