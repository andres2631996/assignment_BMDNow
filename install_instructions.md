Installation instructions

Conda environment

```
conda create -n nnunet_assign python=3.11
conda activate nnunet_assign
pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
git clone https://github.com/MIC-DKFZ/nnUNet.git 
pip install -e nnUNet
pip install -r requirements.txt
```

Path setup
```
export nnUNet_raw=/media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/converted_data
export nnUNet_preprocessed=/home/a870a/preprocessed_assignment
export nnUNet_results=/media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/results
export nnUNet_n_proc_DA=4
```


Datasets:

TotalSegmentator (v300): https://zenodo.org/records/22688904
Verse (2019+2020): https://github.com/anjany/verse#data

Download TotalSegmentator in: assignment/raw_data/totalsegmentator
Download Verse in: assignment/raw_data/verse

Unzip all related .zip files 




Issues:
Set all volumes to Int16, otherwise for some volumes RAM exploded
sub-verse650, 651, 641, slight mismatches (head info only)
Changing FOVs, while Totalsegmentator is mostly whole-body always
Duplicated IDs 400-417 for Verse
Meta CSV from TotalSegmentator: weird
No orthonormal direction from TotalSegm: weird, also in MITK reader
No official TotalSegmentator validation set 
Keeping only cases with spinal cord: 1830 --> 1688 / Test set: 109 --> 108


Runs
```
python code/curate_verse.py --i /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/raw_data/verse --orient RAS --o /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/converted_data/Dataset000_lumbar
python code/curate_totalsegmentator.py --i /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/raw_data/Totalsegmentator_dataset_v300 --orient RAS --o /media/E132-Projekte/Projects/2025_MartinezMora_SSLBrain/assignment/converted_data/Dataset000_lumbar
nnUNetv2_extract_fingerprint -d 000 -np 4
nnUNetv2_plan_experiment -d 000 -pl nnUNetPlannerResEncM
nnUNetv2_preprocess -d 000 -plans_name nnUNetResEncUNetMPlans -c 3d_fullres -np 2

```
