import argparse
import os, sys
import pandas as pd
import time
from joblib import Parallel, delayed
import glob
from utils.curation import (
    output_folder_creation,
    logger_creation,
    process_image_totalsegmentator,
)
from data_io.loaders import load_totalsegmentator_split


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
    output_folder_creation(outfolder, "Tr")
    output_folder_creation(outfolder, "Ts_totalsegmentator")

    # Create logger
    logger_creation(outfolder, "totalsegmentator_curation")

    # Extract all _ct.nii.gz files (raw image files)
    img_files = glob.glob(f"{infolder}/**/ct.nii.gz", recursive=True)

    # Load split information
    split_file = os.path.join(infolder, "meta.csv")
    assert os.path.exists(split_file) and split_file.endswith(
        ".csv"
    ), f"Split file '{split_file}' does not exist or is not .csv"

    split_info = load_totalsegmentator_split(split_file)

    # Main run
    Parallel(n_jobs=workers)(
        delayed(process_image_totalsegmentator)(
            file, outfolder, orient.upper().strip(), split_info
        )
        for file in img_files
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Convert Totalsegmentator datasets into an only clean folder in nnU-Net folder"
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
    # python curate_totalsegmentator.py --i INPUT_FOLDER --orient RAS --o /CONVERTED/DATA/FOLDER/Dataset000_lumbar --np 1
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
