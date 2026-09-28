import os, sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from loguru import logger


def check_same_geometry(
    image: sitk.Image, label: sitk.Image, cid: str, atol: float = 1e-5
):
    """
    Check that image and label objects have the same geometry

    Params
    ------
    image : image object
    label : label object
    cid : case ID
    atol : tolerance (default: 1e-5)

    """
    checks = {
        "size": image.GetSize() == label.GetSize(),
        "spacing": np.allclose(image.GetSpacing(), label.GetSpacing(), atol=atol),
        "origin": np.allclose(image.GetOrigin(), label.GetOrigin(), atol=atol),
        "direction": np.allclose(image.GetDirection(), label.GetDirection(), atol=atol),
    }

    if not all(checks.values()):
        print("Geometry mismatch:")
        for name, result in checks.items():
            if not result:
                logger.info(f"{cid} : {name} : MISMATCH")


def qa_plot(image: sitk.Image, label: sitk.Image, qa_file: str, cid: str):
    """
    Complete QA plot with image and corresponding label

    Params
    ------
    image : input image
    label : input label
    qa_file : file with output QA information

    """
    img = sitk.GetArrayFromImage(image)
    mask_arr = sitk.GetArrayFromImage(label)

    if mask_arr.sum() > 0:
        ind = np.where(mask_arr > 0)
        ind = np.median(ind, 0)
    else:
        ind = np.array(img.shape) // 2

    plt.figure()
    plt.subplot(321)
    plt.imshow(img[img.shape[0] // 2], cmap="gray")
    plt.colorbar()
    plt.subplot(322)
    plt.imshow(mask_arr[img.shape[0] // 2], cmap="gray")
    plt.colorbar()
    plt.subplot(323)
    plt.imshow(img[:, img.shape[1] // 2], cmap="gray")
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(mask_arr[:, img.shape[1] // 2], cmap="gray")
    plt.colorbar()
    plt.subplot(325)
    plt.imshow(img[:, :, img.shape[2] // 2], cmap="gray")
    plt.colorbar()
    plt.subplot(326)
    plt.imshow(mask_arr[:, :, img.shape[2] // 2], cmap="gray")
    plt.colorbar()
    plt.suptitle(cid)
    plt.savefig(qa_file)
    plt.close()
