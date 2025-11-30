#!/usr/bin/env bash
set -e

# Always run relative to this script's folder
cd "$(dirname "$0")"

PYTHON="../.EEG/Scripts/python.exe"
PYTHON_SCRIPT="../src/my_object_classification.py"
DATA_DIR="D:\DATASETS\EEG-ImageNet"
G_OPTION="fine3"
#M_OPTION="eegnet"
B_OPTION=80
O_OPTION="../output"


MODELS=("svm" "mlp")

# Start fresh CSV
rm -f "$OUT_DIR/classification_results.csv"

for m in "${MODELS[@]}"; do
    for i in {0..8}; do
        "$PYTHON" "$PYTHON_SCRIPT" \
            -d "$DATA_DIR" -g "$G_OPTION" -m "$m" \
            -b "$B_OPTION" -p "$P_OPTION1" -s "$i" -o "$O_OPTION"
    done
done