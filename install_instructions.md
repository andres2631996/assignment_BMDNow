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
