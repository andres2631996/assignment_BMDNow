import os, sys
import argparse
import time
from joblib import Parallel, delayed
import glob
from utils.qa import process_file_postprocess


def main(args):
    pred_folder = args.p
    loc_folder = args.l
    outfolder = args.o
    workers = args.np

    assert os.path.exists(
        pred_folder
    ), f"Prediction folder '{pred_folder}' does not exist"
    assert os.path.exists(loc_folder), f"Location folder '{loc_folder}' does not exist"

    assert os.path.exists(
        os.path.dirname(outfolder)
    ), f"Parent output folder '{os.path.dirname(outfolder)}' does not exist"
    assert workers > 0, "Zero or negative workers"

    # Extract all .nii.gz files (predicted files)
    pred_files = glob.glob(f"{pred_folder}/**/*.nii.gz", recursive=True)

    # Create output folder if it does not exist
    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    Parallel(n_jobs=workers)(
        delayed(process_file_postprocess)(pred_file, loc_folder, outfolder)
        for pred_file in pred_files
    )


def get_args():
    parser = argparse.ArgumentParser(
        description="Run QA for a set of predictions and locations"
    )
    parser.add_argument("--p", help="Prediction folder", required=True, type=str)
    parser.add_argument("--l", help="Location folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument(
        "--np", help="Parallel workers", required=False, default=4, type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    # Example run: python postprocess_qa.py --p /folder/with/predictions --l /folder/with/output/locations --o /folder/with/QA/plots
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
