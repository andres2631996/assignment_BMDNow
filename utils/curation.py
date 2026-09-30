import os, sys
import numpy as np
import SimpleITK as sitk
from loguru import logger
from data_io.loaders import (
    load_image,
    save_image,
    load_label_files_totalsegmentator,
    load_spinal_cord_file_totalsegmentator,
)
from utils.qa import check_same_geometry, qa_plot
import time

# Log file configured in the current process (each joblib worker is a separate process)
_configured_log_file = None


def output_folder_creation(out: str, tag: str = "Tr"):
    """
    Create output folder

    Params
    ------
    out : output folder
    str : folder tag ("Tr" for train, "Ts" for test)

    Returns
    -------
    Created output image and label folders

    """
    # Create output nnU-Net structure
    out_img_folder = os.path.join(out, f"images{tag}")
    out_label_folder = os.path.join(out, f"labels{tag}")
    out_qa_folder = os.path.join(out, f"qa{tag}")

    if not (os.path.exists(out_img_folder)):
        os.makedirs(out_img_folder)

    if not (os.path.exists(out_label_folder)):
        os.makedirs(out_label_folder)

    if not (os.path.exists(out_qa_folder)):
        os.makedirs(out_qa_folder)


def logger_creation(out: str, tag: str):
    """
    Create file logger (once per process)

    Params
    ------
    out : output folder
    tag : tag for log file, saved as '<tag>.log'

    """
    global _configured_log_file
    log_file = os.path.join(out, f"{tag}.log")
    if _configured_log_file == log_file:
        # Already configured in this process
        return
    assert os.path.exists(out), f"Output folder '{out}' does not exist"
    logger.remove()
    logger.add(log_file)
    _configured_log_file = log_file


def relabel_verse(label_arr: np.ndarray) -> np.ndarray:
    """
    Relabel Verse dataset


    Params
    ------
    label_arr : array to relabel

    Returns
    -------
    out : relabeled array

    """
    # L1-L5 labels for Verse: 20-24 --> set to L1-L5: 1-5
    out = label_arr - 19
    out[out <= 0] = 0
    out[out > 5] = 0

    return out


def image_label_process_verse(
    image_file: str, label_file: str, cid: str, orienter: sitk.DICOMOrientImageFilter
):
    """
    Process corresponding image and label files for Verse dataset

    Params
    ------
    image_file : input image file
    label_file : input label file


    Returns
    -------
    out_image : output image sitk object
    out_label : output label sitk object

    """
    assert os.path.exists(image_file) and image_file.endswith(
        ".nii.gz"
    ), f"Image file '{image_file}' does not exist or is not .nii.gz"
    assert os.path.exists(label_file) and label_file.endswith(
        ".nii.gz"
    ), f"Image file '{label_file}' does not exist or is not .nii.gz"

    image = load_image(image_file)
    if image.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64):
        # Set to Int16 to avoid RAM booming
        image = sitk.Round(image)
    image = sitk.Cast(image, sitk.sitkInt16)
    label = sitk.Cast(load_image(label_file), sitk.sitkUInt8)

    image_orient = orienter.Execute(image)
    label_orient = orienter.Execute(label)

    # Check image and label geometries
    check_same_geometry(image_orient, label_orient, cid)

    # Relabel
    label_arr = sitk.GetArrayFromImage(label_orient)
    relabel = relabel_verse(label_arr)
    relabel_image = sitk.GetImageFromArray(relabel.astype(np.uint8))
    relabel_image.CopyInformation(label_orient)

    return image_orient, relabel_image


def derive_label_verse(cid_folder: str, cid: str) -> str:
    """
    Derive label file from raw Verse dataset

    Params
    ------
    file : raw file
    cid_folder : raw image file CID folder
    cid : case ID (CID)

    Returns
    -------
    label_file : file with label information

    """
    cid_split = cid.split("_")[0]  # Splitted case ID
    # Derive label file from image file
    label_cid_folder = os.path.join(
        os.path.dirname(os.path.dirname(cid_folder)), "derivatives", cid_split
    )
    label_file = os.path.join(label_cid_folder, f"{cid}_seg-vert_msk.nii.gz")

    return label_file


def derive_label_totalsegmentator(cid_folder: str) -> str:
    """
    Derive label file from raw Totalsegmentator dataset

    Params
    ------
    file : raw file
    cid_folder : raw image file CID folder
    cid : case ID (CID)

    Returns
    -------
    target_files : label files

    """

    # Derive label file from image file
    label_cid_folder = os.path.join(cid_folder, "segmentations")
    target_files = [
        os.path.join(label_cid_folder, f"vertebrae_L{i}.nii.gz")
        for i in range(1, 6)
        if os.path.exists(os.path.join(label_cid_folder, f"vertebrae_L{i}.nii.gz"))
    ]

    return target_files


def process_image_verse(file: str, out: str, orient: str):
    """
    Process raw image from Verse

    Params:
    ------
    file : raw file
    out : output folder
    orienter : sitk orienter object to force a desired orientation

    Returns
    -------
    Saved image in nnUNet format

    """
    # Set up logger in this worker process
    logger_creation(out, "verse_curation")

    # Set up outfile, if outfile exists, skip
    cid_folder = os.path.dirname(file)  # Raw image CID folder
    cid = os.path.basename(file).replace("_ct.nii.gz", "")
    outfile_img = os.path.join(out, "imagesTs", f"{cid}_0000.nii.gz")
    outfile_label = os.path.join(out, "labelsTs", f"{cid}.nii.gz")
    outfile_qa = os.path.join(out, "qaTs", f"{cid}.png")

    if not (os.path.exists(outfile_img)) and not (os.path.exists(outfile_label)):
        logger.info(f"Processing {cid}")
        t1 = time.time()

        # Derive label image
        label_file = derive_label_verse(cid_folder=cid_folder, cid=cid)

        # Set orienter:
        orienter = sitk.DICOMOrientImageFilter()
        orienter.SetDesiredCoordinateOrientation(orient)

        # Process image and label sitk objects (fix orientation and label information)
        image, label = image_label_process_verse(file, label_file, cid, orienter)

        # Save image, label, and run QA
        image_saving(image, label, outfile_img, outfile_label, outfile_qa, cid)

        logger.info(f"Time ellapsed: {round(time.time()-t1,2)} sec")


def image_label_process_totalsegmentator(
    image_file: str, label_files: list, cid: str, orienter: sitk.DICOMOrientImageFilter
):
    """
    Process corresponding image and label files for Verse dataset

    Params
    ------
    image_file : input image file
    label_file : input label file


    Returns
    -------
    out_image : output image sitk object
    out_label : output label sitk object

    """
    assert os.path.exists(image_file) and image_file.endswith(
        ".nii.gz"
    ), f"Image file '{image_file}' does not exist or is not .nii.gz"

    image = load_image(image_file)
    if image.GetPixelID() in (sitk.sitkFloat32, sitk.sitkFloat64):
        # Set to Int16 to avoid RAM booming
        image = sitk.Round(image)
    image = sitk.Cast(image, sitk.sitkInt16)

    # Load label files
    label = load_label_files_totalsegmentator(image, label_files)

    image_orient = orienter.Execute(image)
    label_orient = orienter.Execute(label)

    # Check image and label geometries
    check_same_geometry(image_orient, label_orient, cid)

    return image_orient, label_orient


def image_saving(
    image: sitk.Image,
    label: sitk.Image,
    outfile_img: str,
    outfile_label: str,
    outfile_qa: str,
    cid: str,
):
    """
    Save images and conduct QA

    Params
    ------
    image : image to process
    label : lable to process
    outfile_img : image output file
    outfile_label : label output file
    outfile_qa : QA output file
    cid : case ID

    """

    # Save image and label
    save_image(image, outfile_img)
    save_image(label, outfile_label)

    # Provide QA
    qa_plot(image, label, outfile_qa, cid)


def derive_spinal_info_totalsegmentator(folder: str):
    """
    From data folder of case ID, derive whether file contains
    spinal cord or not for TotalSegmentator

    Params
    ------
    folder : TotalSegmentator case ID folder

    Returns
    -------
    contains_spinal_cord : image contains or not spinal cord

    """
    spinal_file = os.path.join(folder, "segmentations", "spinal_cord.nii.gz")
    contains_spinal_cord = load_spinal_cord_file_totalsegmentator(spinal_file)
    return contains_spinal_cord


def process_image_totalsegmentator(file: str, out: str, orient: str, split: dict):
    """
    Process raw image from TotalSegmentator

    Params:
    ------
    file : raw file
    out : output folder
    orienter : sitk orienter object to force a desired orientation
    split : information on IDs and split

    Returns
    -------
    Saved image in nnUNet format

    """
    # Set up logger in this worker process
    logger_creation(out, "totalsegmentator_curation")

    # Set up outfile, if outfile exists, skip
    cid_folder = os.path.dirname(file)  # Raw image CID folder
    cid = os.path.basename(cid_folder)
    keys = list(split.keys())
    assert cid in keys, f"Case ID '{cid}' not in split information"

    # Decide where to store case (training or test folder for TotalSegmentator)
    tag = "Tr"  # Default tag for training case
    if split[cid].lower().strip() == "test":
        tag = "Ts_totalsegmentator"

    outfile_img = os.path.join(out, f"images{tag}", f"{cid}_0000.nii.gz")
    outfile_label = os.path.join(out, f"labels{tag}", f"{cid}.nii.gz")
    outfile_qa = os.path.join(out, f"qa{tag}", f"{cid}.png")

    if not (os.path.exists(outfile_img)) and not (os.path.exists(outfile_label)):
        logger.info(f"Processing {cid}")
        t1 = time.time()

        # Derive label files to process and spinal cord file
        label_files = derive_label_totalsegmentator(cid_folder=cid_folder)
        contains_spinal_cord = derive_spinal_info_totalsegmentator(folder=cid_folder)
        if contains_spinal_cord:
            # Set orienter:
            orienter = sitk.DICOMOrientImageFilter()
            orienter.SetDesiredCoordinateOrientation(orient)

            # Process image and label sitk objects (fix orientation and label information)
            image, label = image_label_process_totalsegmentator(
                file, label_files, cid, orienter
            )

            # Save image, label, and run QA
            image_saving(image, label, outfile_img, outfile_label, outfile_qa, cid)

            logger.info(f"Time ellapsed: {round(time.time()-t1,2)} sec")
        else:
            logger.info(f"No spinal cord information for case ID '{cid}', skipping...")
