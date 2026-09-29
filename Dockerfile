# Lumbar vertebrae body center extraction - end-to-end inference image
#
# Build (the trained model lives outside this repo, so it is passed as a named build context):
#   docker build -t lumbar-centers \
#     --build-context model=../results/Dataset000_lumbar/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres .
#
# Run: ./run_inference.sh <input_dir> <output_dir>

FROM pytorch/pytorch:2.8.0-cuda12.6-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # nnU-Net only needs results for inference; raw/preprocessed are set to silence warnings
    nnUNet_raw=/tmp/nnUNet_raw \
    nnUNet_preprocessed=/tmp/nnUNet_preprocessed \
    nnUNet_results=/opt/nnunet_results \
    # Container may run as a non-root host user
    HOME=/tmp \
    MPLCONFIGDIR=/tmp/matplotlib \
    MPLBACKEND=Agg

WORKDIR /app

# nnU-Net (vendored copy, same commit used for training) + extra requirements
COPY nnUNet /opt/nnUNet
COPY requirements.txt /app/requirements.txt
RUN pip install /opt/nnUNet && pip install -r /app/requirements.txt

# Trained model: only the files needed by nnUNetPredictor
ARG MODEL_DIR=/opt/nnunet_results/Dataset000_lumbar/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres
COPY --from=model dataset.json plans.json ${MODEL_DIR}/
COPY --from=model fold_0/checkpoint_final.pth ${MODEL_DIR}/fold_0/

# Code
COPY inference.py /app/
COPY data_io /app/data_io
COPY utils /app/utils
RUN echo "{\"model_dir\": \"${MODEL_DIR}\"}" > /app/cfg.json

ENTRYPOINT ["python", "inference.py", "--i", "/data/input", "--o", "/data/output"]
