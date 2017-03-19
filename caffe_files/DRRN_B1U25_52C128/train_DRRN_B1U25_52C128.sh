#!/usr/bin/env sh
LOG=./log/DRRN_B1U25_52C128_291_31.log
CAFFE=/data2/taiying/MSU_Code/my_caffe/build/tools/caffe # your caffe path

$CAFFE train --solver=./DRRN_B1U25_52C128_solver.prototxt -gpu 0 2>&1 | tee $LOG

## resume training
#$CAFFE train --solver=./DRRN_B1U25_52C128_solver.prototxt \
#--snapshot=../../model/DRRN_B1U25_52C128_iter_6536.solverstate -gpu 0 2>&1 | tee $LOG

