# CT lumbar vertebrae body center extraction

Repo for lumbar vertebrae body center estimation, trained and internally tested on TotalSegmentator (https://zenodo.org/records/22688904) and externally tested on Verse (https://github.com/anjany/verse#data)

# Installation instructions

```
conda create -n nnunet_assign python=3.11
conda activate nnunet_assign
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/MIC-DKFZ/nnUNet.git 
pip install -e nnUNet
pip install -r requirements.txt
```

# Path setup
```
export nnUNet_raw=../converted_data
export nnUNet_preprocessed=../preprocessed_data
export nnUNet_results=../results
export nnUNet_n_proc_DA=4
```


# Datasets:

TotalSegmentator (v300): https://zenodo.org/records/22688904

Verse (2019+2020): https://github.com/anjany/verse#data

Download instructions:

Download TotalSegmentator in: ../raw_data/totalsegmentator

Download Verse in: ../raw_data/verse

Unzip all related .zip files 


# Docker (end-to-end inference)

Requires Docker with the NVIDIA container runtime (GPU needed). The trained model is copied into the image, so the image is self-contained.

1. Get the image (either option A or option B, not both)

A. Build it from this repo (the model folder is passed as a named build context)
```
docker build -t lumbar-centers --build-context model=../path/to/model/nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres .
```

B. Load a previously exported image (no repo or model files needed). The export is created on the build machine with:
```
docker save lumbar-centers | gzip > /path/to/lumbar-centers.tar.gz
```
and loaded on the target machine with:
```
docker load < lumbar-centers.tar.gz
```

2. Run inference from the host (run_inference.sh starts the container, mounts the folders and runs inference.py inside it). Input images must be named <case_id>.nii.gz; per case, <case_id>.json with the centers, <case_id>.nii.gz with the segmentation and <case_id>.png with the QA plot are written to the output folder
```
./run_inference.sh <input_dir> <output_dir>
```
If you run the Docker container on a mounted drive and the option above does not work, try otherwise with:
```
bash run_inference.sh <input_dir> <output_dir>
```

Example for TotalSegmentator and Verse
```
./run_inference.sh ../raw_data/TotalSegmentator_dataset_v300 ../locsTs_totalsegmentator_docker

./run_inference.sh ../raw_data/verse ../locsTs_docker
```

Check that the container sees the GPU (should print True)
```
docker run --rm --gpus all --entrypoint python lumbar-centers -c "import torch; print(torch.cuda.is_available())"
```





# Runs (step by step)

Curate Verse as imagesTs, labelsTs
```
python curate_verse.py --i ../raw_data/verse --orient RAS --o ../converted_data/Dataset000_lumbar
```

Curate TotalSegmentator as imagesTr, labelsTr, imagesTs\_totalsegmentator, labelsTs\_segmentator (here done with version 3.0.0, but can be done with other versions, too)

```
python curate_totalsegmentator.py --i ../raw_data/Totalsegmentator_dataset_v300 --orient RAS --o ../converted_data/Dataset000_lumbar
```

nnU-Net-based preprocessing (parallel workers can be adjusted)

```
nnUNetv2_extract_fingerprint -d 000 -np WORKERS

nnUNetv2_plan_experiment -d 000 -pl nnUNetPlannerResEncM

nnUNetv2_preprocess -d 000 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres -np WORKERS
```

nnUNet-based training

```
nnUNetv2_train 000 3d_fullres 0 -p nnUNetResEncUNetMPlans -tr nnUNetTrainer
```

nnUNet-based prediction of internal TotalSegmentator test set and external Verse test set (WORKERS here set to 1, can be modified)

```
nnUNetv2_predict -o ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator -i ../converted_data/Dataset000_lumbar/imagesTs_totalsegmentator -d 000 -tr nnUNetTrainer -c 3d_fullres -f 0 -npp 1 -nps 1 -p nnUNetResEncUNetMPlans

nnUNetv2_predict -o /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/converted_data/Dataset000_lumbar/predsTs -i /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/converted_data/Dataset000_lumbar/imagesTs -d 000 -tr nnUNetTrainer -c 3d_fullres -f 0 -npp 1 -nps 1 -p nnUNetResEncUNetMPlans
```

nnU-Net-based evaluation of internal TotalSegmentator test set and external Verse test set. It generates a file called summary.json with metrics for the different lumbar vertebrae

```
nnUNetv2_evaluate_folder ../converted_data/Dataset000_lumbar/labelsTs_totalsegmentator ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator -djfile ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator/dataset.json -pfile ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator/plans.json -o ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator/summary.json -np WORKERS 

nnUNetv2_evaluate_folder ../converted_data/Dataset000_lumbar/labelsTs ../converted_data/Dataset000_lumbar/predsTs -djfile ../converted_data/Dataset000_lumbar/predsTs/dataset.json -pfile ../converted_data/Dataset000_lumbar/predsTs/plans.json -o ../converted_data/Dataset000_lumbar/predsTs/summary.json -np WORKERS 
```

Segmentation metrics bootstrapping for TotalSegmentator test set and Verse test set (after running nnU-Net based evaluation). It generates a file called summary_bootstrapping.json with confidence intervals for Dice and IoU metrics for the different lumbar vertebrae
```
python utils/bootstrap_metrics.py --i ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator/summary.json --n 1000 --np WORKERS

python utils/bootstrap_metrics.py --i ../converted_data/Dataset000_lumbar/predsTs/summary.json --n 1000 --np WORKERS
```

Postprocessing for TotalSegmentator and Verse

```
python postprocess.py --i ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator --o ../converted_data/Dataset000_lumbar/locsTs_totalsegmentator

python postprocess.py --i ../converted_data/Dataset000_lumbar/predsTs --o ../converted_data/Dataset000_lumbar/locsTs
```

Postprocessing QA for TotalSegmentator and Verse
```
python postprocess_qa.py --p ../converted_data/Dataset000_lumbar/predsTs_totalsegmentator --l ../converted_data/Dataset000_lumbar/locsTs_totalsegmentator --o ../converted_data/Dataset000_lumbar/qaTs_locs_totalsegmentator

python postprocess_qa.py --p ../converted_data/Dataset000_lumbar/predsTs --l ../converted_data/Dataset000_lumbar/locsTs --o ../converted_data/Dataset000_lumbar/qaTs_locs
```

End-to end inference for TotalSegmentator and Verse
```
python inference.py --i ../converted_data/Dataset000_lumbar/imagesTs_totalsegmentator --o ../converted_data/Dataset000_lumbar/locsTs_totalsegmentator

python inference.py --i ../converted_data/Dataset000_lumbar/imagesTs --o ../converted_data/Dataset000_lumbar/locsTs
```
