import os, sys
import torch
import time
from loguru import logger

from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


from data_io.loaders import load_orient_image
from utils.postprocess_utils import extract_locations


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


def predict_file(file: str, cfg: dict):
    """
    Predict file with model provided in configuration path

    Params
    ------
    file : file to be predicted
    cfg : configuration with model path

    Returns
    -------
    image : reference image
    pred : lumbar vertebra segmentation

    """
    # Load image
    image_np, properties = NibabelIOWithReorient().read_images([file])

    # Load image and enforce RAS orientation
    image = load_orient_image(file, False, "RAS")

    # Set up model
    assert "model_dir" in list(cfg.keys()), "Configuration lacks 'model_dir' field"
    model_dir = cfg["model_dir"]
    split_info = model_dir.split("/")
    task = split_info[2]
    nnunet_model_dir = "/".join(split_info[:1])
    model_name = split_info[-1]
    model = MySegmentation(task, nnunet_model_dir, model_name)
    pred = model.process_image(image_np, properties)

    return pred, image


def process_file(file: str, out: str, cfg: dict):
    """
    Process file for inference

    Params
    ------
    file : input image file
    out : output folder
    cfg : model configuration information

    """
    # Extract case ID
    cid = os.path.basename(file).replace("_0000.nii.gz", "")
    outfile = os.path.join(out, f"{cid}.json")

    if not (os.path.exists(outfile)):
        # Skip already processed files
        assert os.path.exists(file) and file.endswith(
            ".nii.gz"
        ), f"File '{file}' does not exist or is not nii.gz"

        logger.info(f"Processing {cid}")

        # Predict file with model information
        t1 = time.time()
        pred, image = predict_file(file, cfg)
        logger.info(f"Elapsed prediction time for '{cid}' : {round(time.time()-t1,2)}")

        # Extract locations
        t1 = time.time()
        extract_locations(pred, image, cid, outfile)
        logger.info(f"Elapsed location time for '{cid}' : {round(time.time()-t1,2)}")
