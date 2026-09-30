import os, sys
import numpy as np
import matplotlib.pyplot as plt
import SimpleITK as sitk
from loguru import logger
from batchgenerators.utilities.file_and_folder_operations import load_json
from data_io.loaders import load_image


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

    ind = np.array(img.shape) // 2

    plt.figure()
    plt.subplot(321)
    plt.imshow(img[ind[0]], cmap="gray")
    plt.colorbar()
    plt.subplot(322)
    plt.imshow(mask_arr[ind[0]], cmap="gray")
    plt.colorbar()
    plt.subplot(323)
    plt.imshow(img[:, ind[1]], cmap="gray")
    plt.colorbar()
    plt.subplot(324)
    plt.imshow(mask_arr[:, ind[1]], cmap="gray")
    plt.colorbar()
    plt.subplot(325)
    plt.imshow(img[:, :, ind[2]], cmap="gray")
    plt.colorbar()
    plt.subplot(326)
    plt.imshow(mask_arr[:, :, ind[2]], cmap="gray")
    plt.colorbar()
    plt.suptitle(cid)
    plt.savefig(qa_file)
    plt.close()


def qa_plot_postprocess(image: sitk.Image, centroids: dict, qa_file: str, cid: str):
    """
    Complete QA plot with image and corresponding centroids for postprocessing

    Params
    ------
    image : input image
    label : input label
    qa_file : file with output QA information

    """
    img = sitk.GetArrayFromImage(image)

    if len(centroids.keys()) == 0:  # No locations
        plt.figure()
        # Plot only three views

        ind = np.array(img.shape) // 2

        plt.subplot(131)
        plt.imshow(img[ind[0]], cmap="gray")
        plt.subplot(132)
        plt.imshow(img[:, ind[1]], cmap="gray")
        plt.subplot(133)
        plt.imshow(img[:, :, ind[2]], cmap="gray")
        plt.suptitle(cid)
        plt.tight_layout()
        plt.savefig(qa_file)
        plt.close()
    else:
        keys = list(centroids.keys())
        n = len(keys)
        sz, sy, sx = image.GetSpacing()[::-1]
        fig, axes = plt.subplots(n, 3, figsize=(12, 4 * n), squeeze=False)

        for row, key in enumerate(keys):
            # Convert world index centroid to image index
            cz, cy, cx = image.TransformPhysicalPointToContinuousIndex(centroids[key])[
                ::-1
            ]
            z, y, x = (int(round(v)) for v in (cz, cy, cx))

            # Check if centroid lies outside volume
            if not (
                0 <= z < img.shape[0]
                and 0 <= y < img.shape[1]
                and 0 <= x < img.shape[2]
            ):
                continue  # centroid outside the volume

            # Main plotting for the three axes
            ax = axes[row]
            ax[0].imshow(img[z], cmap="gray", aspect=sy / sx)  # axial
            ax[0].plot(cx, cy, "r+", ms=15, mew=2)

            ax[1].imshow(img[:, y, :], cmap="gray", aspect=sz / sx)  # coronal
            ax[1].plot(cx, cz, "r+", ms=15, mew=2)
            ax[1].set_title(key)

            ax[2].imshow(img[:, :, x], cmap="gray", aspect=sz / sy)  # sagittal
            ax[2].plot(cy, cz, "r+", ms=15, mew=2)

            for a in ax:
                a.axis("off")

        fig.suptitle(cid)
        plt.tight_layout()
        fig.savefig(qa_file)
        plt.close(fig)


def process_file_postprocess(file: str, loc_folder: str, out: str):
    """
    Process image file for QA postprocessing

    Params
    ------
    file : file to postprocess
    loc_folder : folder with location files
    out : output folder

    """
    # Search for location file and output file
    cid = os.path.basename(file).replace(".nii.gz", "")
    outfile = os.path.join(out, f"{cid}.png")
    if not (os.path.exists(outfile)):
        loc_file = os.path.join(loc_folder, f"{cid}.json")
        assert os.path.exists(loc_file), f"Location file '{loc_file}' does not exist"
        centroids = load_json(loc_file)["centroids"]
        image = load_image(file, False)
        qa_plot_postprocess(image, centroids, outfile, cid)
