#!/bin/bash

PYTHON_SCRIPT="../src/object_classification.py"

#DATA_DIR="../data/"
DATA_DIR="D:\DATASETS\EEG-ImageNet"
G_OPTION="all"
M_OPTION="eegnet"
B_OPTION=40
S_OPTION=0
P_OPTION="eegnet_s${S_OPTION}_1x_0.pth"
O_OPTION="../output/"

#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION -s $S_OPTION -o $O_OPTION
#python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $S_OPTION -o $O_OPTION

for i in {0..15}
do
    P_OPTION1="eegnet_s${i}_1x_1.pth"
#    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -s $i -o $O_OPTION
    python $PYTHON_SCRIPT -d $DATA_DIR -g $G_OPTION -m $M_OPTION -b $B_OPTION -p $P_OPTION1 -s $i -o $O_OPTION
done