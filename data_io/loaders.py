import os, sys, re
import SimpleITK as sitk
import numpy as np
import pandas as pd
import nibabel as nib
from loguru import logger


def load_image(file: str, arr: bool = False) -> sitk.Image:
    """
    Load image with SimpleITK
    Optionally load array, too

    Params
    ------
    file : image file
    arr : array

    """
    try:
        image = sitk.ReadImage(file)
    except:
        # Load with nibabel
        logger.error(f"Failed loading '{file}', trying with Nibabel...")
        image = _load_with_nibabel(file)
    if arr:
        img = sitk.GetArrayFromImage(image)
        return image, img
    return image


def save_image(image: sitk.Image, file: str):
    """
    Save input image

    Params
    ------
    image : input image
    file : output image file

    """
    sitk.WriteImage(image, file)


def load_label_files_totalsegmentator(image: sitk.Image, files: list) -> sitk.Image:
    """
    Combine TotalSegmentator lumbar vertebra masks into one label image.

    Params
    ------
    image : reference SimpleITK image (defines shape and geometry)
    files : TotalSegmentator vertebrae_L*.nii.gz files

    Returns
    -------
    label_image : label image with L1=1, L2=2, ...; overlapping voxels set to 0
    """
    ref = sitk.GetArrayFromImage(image)
    label = np.zeros(ref.shape, dtype=np.uint8)
    count = np.zeros(ref.shape, dtype=np.uint8)

    for file in files:
        if not (os.path.exists(file) and file.endswith(".nii.gz")):
            raise FileNotFoundError(
                f"Label file '{file}' does not exist or is not .nii.gz"
            )

        match = re.match(r"vertebrae_L(\d+)\.nii\.gz$", os.path.basename(file))
        if match is None:
            raise ValueError(f"Unexpected file name: '{file}'")
        vertebra = int(match.group(1))

        _, arr = load_image(file, arr=True)
        mask = arr.astype(bool)
        if mask.shape != label.shape:
            raise ValueError(
                f"Shape mismatch for '{file}': {mask.shape} vs {label.shape}"
            )

        label[mask] = vertebra
        count += (mask > 0).astype(np.uint8)

    # Set clashing labels to background
    label[count > 1] = 0

    label_image = sitk.GetImageFromArray(label)
    label_image.CopyInformation(image)
    return label_image


def load_totalsegmentator_split(split_file: str) -> dict:
    """
    Load split information from TotalSegmentator

    Params
    ------
    split_file : split file

    Returns
    -------
    split_info : {"id" : "train/val/test"}

    """
    df = pd.read_csv(split_file, sep=";")
    split_info = dict(zip(df["image_id"], df["split"]))
    return split_info


def _load_with_nibabel(file: str) -> sitk.Image:
    """
    Load data in nibabel and convert to sitk

    Params
    ------
    file : input file

    Outputs
    -------
    image : loaded image

    """
    nii = nib.load(str(file))
    data = np.asanyarray(nii.dataobj)

    # nibabel uses RAS, ITK uses LPS -> flip x and y
    affine = np.diag([-1, -1, 1, 1]) @ nii.affine
    rot_zoom = affine[:3, :3]
    spacing = np.linalg.norm(rot_zoom, axis=0)
    direction = rot_zoom / spacing

    # project onto the closest orthonormal matrix
    u, _, vt = np.linalg.svd(direction)
    direction = u @ vt

    # nibabel array is (x, y, z); SimpleITK expects (z, y, x)
    image = sitk.GetImageFromArray(np.transpose(data, (2, 1, 0)))
    image.SetSpacing(spacing.tolist())
    image.SetOrigin(affine[:3, 3].tolist())
    image.SetDirection(direction.flatten().tolist())
    return image
