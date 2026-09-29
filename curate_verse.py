import argparse
import os, sys
import numpy as np
import SimpleITK as sitk
import time
from joblib import Parallel, delayed
import glob
import matplotlib.pyplot as plt
from utils.curation import (
    output_folder_creation,
    logger_creation,
    process_image_verse,
)


def main(args):
    infolder = args.i
    outfolder = args.o
    orient = args.orient
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"
    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative workers"

    # Create output nnU-Net structure
    output_folder_creation(outfolder, "Ts")
    # Create logger
    logger_creation(outfolder, "verse_curation")

    # Extract all _ct.nii.gz files (raw image files)
    img_files = glob.glob(f"{infolder}/**/*_ct.nii.gz", recursive=True)

    # Main run
    Parallel(n_jobs=workers)(
        delayed(process_image_verse)(file, outfolder, orient.upper().strip())
        for file in img_files
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Verse datasets into an only clean folder in nnU-Net folder"
    )
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument(
        "--orient", help="Target orientation", required=False, type=str, default="RAS"
    )
    parser.add_argument(
        "--o", help="Output folder", required=False, default=None, type=str
    )
    parser.add_argument(
        "--np", help="Parallel workers", required=False, default=4, type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    # Example run:
    # python curate_verse.py --i INPUT_FOLDER --orient RAS --o /CONVERTED/DATA/FOLDER/Dataset000_lumbar --np 1
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
