#!/usr/bin/env bash
# End-to-end lumbar vertebrae body center extraction (Docker wrapper around inference.py)
#
# Usage: ./run_inference.sh <input_dir> <output_dir>
#   <input_dir>  : folder with CT images named <case_id>_0000.nii.gz (searched recursively)
#   <output_dir> : per case: <case_id>.json (centers), <case_id>.nii.gz (segmentation), <case_id>.png (QA)
#
# Image name can be overridden with IMAGE=<name> ./run_inference.sh ...
set -euo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <input_dir> <output_dir>" >&2
    exit 1
fi

IMAGE="${IMAGE:-lumbar-centers}"

if [ ! -d "$1" ]; then
    echo "Input folder '$1' does not exist" >&2
    exit 1
fi
mkdir -p "$2"

INPUT_DIR="$(cd "$1" && pwd)"
OUTPUT_DIR="$(cd "$2" && pwd)"

docker run --rm \
    --gpus all \
    --shm-size=8g \
    --user "$(id -u):$(id -g)" \
    -v "${INPUT_DIR}":/data/input:ro \
    -v "${OUTPUT_DIR}":/data/output \
    "${IMAGE}"
