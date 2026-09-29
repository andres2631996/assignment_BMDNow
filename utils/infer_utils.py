import os, sys
import torch
import numpy as np
import time
from loguru import logger
import SimpleITK as sitk

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


from data_io.loaders import load_orient_image, save_image
from utils.postprocess_utils import extract_locations
from utils.qa import qa_plot_postprocess


class MySegmentation:
    def __init__(
        self,
        task="Dataset000_lumbar",
        nnunet_model_dir="../results",
        model_name="nnUNetTrainer__nnUNetResEncUNetMPlans__3d_fullres",
        folds=(0,),
    ):
        # network parameters
        self.predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            perform_everything_on_device=True,
            device=torch.device("cuda", 0),
            verbose=True,
            verbose_preprocessing=False,
            allow_tqdm=True,
        )
        self.predictor.initialize_from_trained_model_folder(
            os.path.join(nnunet_model_dir, f"{task}/{model_name}"),
            use_folds=folds,
            checkpoint_name="checkpoint_final.pth",
        )

    def process_image(self, image_np, properties):
        ret = self.predictor.predict_single_npy_array(
            image_np, properties, None, None, False
        )
        return ret


def cfg2model(cfg: dict):
    """
    Convert config file with model information to actual model object

    Params
    ------
    cfg : model configuration file

    Returns
    -------
    model : final model object

    """

    # Set up model
    assert "model_dir" in list(cfg.keys()), "Configuration lacks 'model_dir' field"
    # model_dir = <nnunet_model_dir>/<task>/<model_name>
    model_dir = os.path.normpath(cfg["model_dir"])
    model_name = os.path.basename(model_dir)
    task = os.path.basename(os.path.dirname(model_dir))
    nnunet_model_dir = os.path.dirname(os.path.dirname(model_dir))
    model = MySegmentation(task, nnunet_model_dir, model_name)

    return model


def predict_file(file: str, model):
    """
    Predict file with model object

    Params
    ------
    file : file to be predicted
    model : model object

    Returns
    -------
    image : reference image
    pred : lumbar vertebra segmentation

    """
    # Load image
    image_np, properties = NibabelIOWithReorient().read_images([file])

    # Heuristic to discard label files
    unique = np.unique(image_np)
    is_label = np.issubdtype(image_np.dtype, np.integer) and len(unique) < 100

    if not (is_label):
        # Load image and enforce RAS orientation
        image = load_orient_image(file, False, "RAS")
        pred = model.process_image(image_np, properties)

        return pred, image

    return None, None


def process_file(file: str, out: str, model):
    """
    Process file for inference

    Params
    ------
    file : input image file
    out : output folder
    model : model information

    """
    # Extract case ID
    cid = os.path.basename(file).replace(".nii.gz", "")
    outfile = os.path.join(out, f"{cid}.json")

    if not (os.path.exists(outfile)):
        # Skip already processed files
        assert os.path.exists(file) and file.endswith(
            ".nii.gz"
        ), f"File '{file}' does not exist or is not nii.gz"

        logger.info(f"Processing {cid}")

        # Predict file with model information
        t1 = time.time()
        pred, image = predict_file(file, model)

        if pred is not None and image is not None:

            # Extract locations
            t1 = time.time()
            centroid_info = extract_locations(pred, image, cid, outfile)

            # Save predicted image
            pred_image = sitk.GetImageFromArray(pred)
            pred_image.CopyInformation(image)
            outfile_img = os.path.join(out, f"{cid}.nii.gz")
            save_image(pred_image, outfile_img)

            # Save QA too
            # Run postprocess QA
            qa_plot_postprocess(
                image, centroid_info["centroids"], outfile.replace(".json", ".png"), cid
            )
            logger.info(
                f"Elapsed location time for '{cid}' : {round(time.time()-t1,2)}"
            )
        else:
            logger.info(f"File '{file}' seems to be a label file, skipping...")
