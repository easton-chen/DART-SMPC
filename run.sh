#!/bin/bash 

EXP_TYPE="test"     # 
ENV_TYPE="random"   # random, fix, random-long
PRED_TYPE="fuse"    # fuse, latest
PLAN_TYPE="smpc"    # 

LOG_FILE="./Results/DART"

if [ "$1" != "" ]; then
    EXP_TYPE=$1
    LOG_FILE=${LOG_FILE}-${EXP_TYPE}
fi
if [ "$2" != "" ]; then
    ENV_TYPE=$2
    LOG_FILE=${LOG_FILE}-${ENV_TYPE}
fi
if [ "$3" != "" ]; then
    PRED_TYPE=$3
    LOG_FILE=${LOG_FILE}-${PRED_TYPE}
fi
if [ "$4" != "" ]; then
    PLAN_TYPE=$4
    LOG_FILE=${LOG_FILE}-${PLAN_TYPE}
fi


if [ $EXP_TYPE == "test" ]; then
    echo "main.py $EXP_TYPE > MPC.log"
    python3 main.py $EXP_TYPE > MPC.log
elif [ $EXP_TYPE == "timing" ]; then
    for ENV_CASE in {0..19}
    do
        echo "python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log"
        python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log
    done  
elif [ $EXP_TYPE == "eff" ]; then
    for ENV_CASE in {0..49}
    do
        echo "python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log"
        python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log
    done  
elif [ $EXP_TYPE == "pred" ]; then
    for ENV_CASE in {0..49}
    do
        echo "python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log"
        python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log
    done  
elif [ $EXP_TYPE == "setting-u" ]; then
    for ENV_CASE in {20..49}
    do
        echo "python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log"
        python3 main.py $EXP_TYPE $ENV_TYPE $ENV_CASE $PRED_TYPE $PLAN_TYPE >MPC.log
    done  
fi