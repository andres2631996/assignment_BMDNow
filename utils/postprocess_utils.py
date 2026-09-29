import os, sys
import numpy as np
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter1d, label, distance_transform_edt
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import save_json
from data_io.loaders import load_orient_image, extract_spacing_np
from loguru import logger


def u_minimum(y, sigma=2):
    try:
        y = np.asarray(y, dtype=float)
        ys = gaussian_filter1d(y, sigma=sigma)
        valleys, props = find_peaks(-ys, prominence=0)
        keep = ys[valleys] > 0.05 * ys.max()  # skip the zero baseline
        v = valleys[keep][np.argmax(props["prominences"][keep])]
        lo, hi = max(v - 5, 0), v + 6  # refine on raw data
        return lo + np.argmin(y[lo:hi])
    except:
        # No specific local minimum in projection,
        # return whole vertebral mask
        return None


def extract_largest_cc(arr: np.ndarray) -> np.ndarray:
    """
    Extract largest connected component
    If array is empty, a copy of it is returned

    Params
    ------
    arr : segmentation array

    Returns
    -------
    largest : largest CC array

    """
    cc, n = label(arr)
    largest = arr.copy()
    if n > 0:
        sizes = np.bincount(cc.ravel())
        sizes[0] = 0  # ignore background
        largest = cc == sizes.argmax()
    return largest


def process_vertebra(image: sitk.Image, arr: np.ndarray, i: int, spacing: list):
    """
    Extract vertebral body center of vertebra `i`

    Returns
    -------
    image : reference image
    body : boolean vertebral body mask
    centre : (z, y, x) voxel index of the body centre, or None if the body is empty
    """
    # Set vertebra mask to binary mask
    vertebra = (arr == i).astype(np.uint8)

    # Get largest connected component, avoid noise
    largest = extract_largest_cc(arr=vertebra).astype(bool)

    # Obtain projections to get posterior coordinates of vertebra
    # where there is sinking due to vertebral shape
    mip = np.sum(largest, 0)
    line = np.sum(mip, 1)
    # Posterior part of vertebral body: minimum between projected peaks of vertebral body and remaining part voxels
    idx = u_minimum(line)

    # Vertebral body, by default copy information from largest CC
    body = largest.copy()
    if idx is not None and idx > 0:
        # Build "vertebral body" mask: remove posterior vertebral mask voxels
        body[:, :idx, :] = False
        if not body.any():
            body = largest.copy()

    if not body.any():
        return body, None
    edt = distance_transform_edt(body, sampling=spacing)
    centre = np.unravel_index(np.argmax(edt), edt.shape)

    # Obtain world coordinate in mm
    loc = image.TransformContinuousIndexToPhysicalPoint(np.flip(centre))

    # Convert all locations to float, for json file saves
    out_loc = [float(l) for l in loc]
    return out_loc


def extract_locations(arr: np.ndarray, image: sitk.Image, cid: str, outfile: str):
    """
    Extract center locations of vertebral bodies

    Params
    ------
    arr : prediction array
    image : reference image (in RAS orientation)
    cid : case ID
    outfile : output file

    """
    # Derive spacing information
    spacing = extract_spacing_np(image)

    # Set up output dict. Set centroids empty by default
    output_dict = {
        "case_id": cid,
        "coordinate_system": "RAS",
        "unit": "mm",
        "centroids": {},
    }
    centroids = {}
    if arr.sum() > 0:
        # Some vertebra have been detected. Get centroid locations
        idxes = np.unique(arr)
        assert idxes.shape[0] > 1, f"No vertebra detected in '{cid}'"
        locs = [process_vertebra(image, arr, i, spacing) for i in idxes]
        centroids = {f"L{i}": loc for i, loc in zip(idxes, locs)}
        output_dict["centroids"] = centroids

    logger.info(f"{cid} : {output_dict}")

    save_json(output_dict, outfile)


def process_file(file: str, out: str):
    """
    Postprocess file in parallel

    Params
    ------
    file : file to postprocess
    out : output folder

    """
    cid = os.path.basename(file).replace(".nii.gz", "")
    outfile = os.path.join(out, f"{cid}.json")

    if not (os.path.exists(outfile)):
        # Skip already processed files
        assert os.path.exists(file) and file.endswith(
            ".nii.gz"
        ), f"File '{file}' does not exist or is not nii.gz"

        # Load image and enforce RAS orientation
        image, arr = load_orient_image(file, True, "RAS")

        # Extract locations
        extract_locations(arr=arr, image=image, cid=cid, outfile=outfile)
