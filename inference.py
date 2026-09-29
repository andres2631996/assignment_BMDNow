import os, sys
import argparse
import time
import glob
from batchgenerators.utilities.file_and_folder_operations import load_json
from utils.infer_utils import process_file
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
    img_files = glob.glob(f"{infolder}/**/*_0000.nii.gz", recursive=True)

    # Create output folder if it does not exist
    if not (os.path.exists(outfolder)):
        os.makedirs(outfolder)

    # Set up logger
    logger_creation(outfolder, "inference")

    # Load config with model information
    cfg_file = "cfg.json"
    cfg = load_json(cfg_file)

    # Process files in series
    for file in img_files:
        process_file(file, outfolder, cfg)


def get_args():
    parser = argparse.ArgumentParser(description="Run end-to-end inference")
    parser.add_argument("--i", help="Input folder", required=True, type=str)
    parser.add_argument("--o", help="Output folder", required=True, type=str)
    parser.add_argument(
        "--np", help="Parallel workers", required=False, default=4, type=int
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    # Example run: python inference.py --i /folder/with/images --o /folder/with/output/vertebral_center_locations
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
