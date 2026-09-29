import os, sys
import argparse
import time
from joblib import Parallel, delayed
import glob
from utils.postprocess_utils import process_file
from utils.curation import logger_creation


def main(args):
    infolder = args.i
    outfolder = args.o
    workers = args.np

    assert os.path.exists(infolder), f"Input folder '{infolder}' does not exist"

    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative workers"

    # Extract all .nii.gz files (predicted files)
    img_files = glob.glob(f"{infolder}/**/*.nii.gz", recursive=True)

    # Create output folder if it does not exist
    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    # Set up logger
    logger_creation(outfolder, "postprocess")

    Parallel(n_jobs=workers)(
        delayed(process_file)(img_file, outfolder) for img_file in img_files
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Obtain centroid information from a set of predictions as .nii.gz files"
    )
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument(
        "--np", help="Parallel workers", required=False, default=4, type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    # Example run: python postprocess.py --i /folder/with/predictions --o /folder/with/output/locations
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
