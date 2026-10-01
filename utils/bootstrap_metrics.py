import os, sys
from batchgenerators.utilities.file_and_folder_operations import load_json, save_json
import argparse
import time
import numpy as np
import random
from joblib import Parallel, delayed


def extract_ids(df: dict) -> dict:
    """
    Extract IDs from dictionary and build a dict to apply bootstrapping

    Params
    ------
    df : input dict with metrics

    Returns
    -------
    out : cleaned dict with Dice and IoU metrics for case ID

    """
    out = {}
    for d in df:
        # Iterate through list
        # Extract ID
        assert "prediction_file" in list(d.keys()), "'prediction_file' not in key"
        basefile = os.path.basename(d["prediction_file"])
        cid = basefile.split("_")[0].replace(".nii.gz", "")

        if "_" in basefile:
            # This CID has several cases
            # Set all possible cases in a list of dicts, for later sampling during bootstrapping
            if cid in list(out.keys()):
                out[cid] += [d["metrics"]]
            else:
                out[cid] = [d["metrics"]]
        else:
            out[cid] = [d["metrics"]]
    return out


def bootstrap_iter(d: dict) -> dict:
    """
    Collect metric information for bootstrapping iteration

    Params
    -------
    d : cleaned dictionary in "extract_ids"

    Returns
    -------
    metrics : dict with aggregated metrics for sampled cases, for every vertebra

    """
    # Sampling indexes
    rng = np.random.default_rng(42)

    keys = list(d.keys())

    bootstrap_keys = np.random.choice(keys, size=len(keys), replace=True)

    # Aggregate and obtain results
    dices, ious = {"1": [], "2": [], "3": [], "4": [], "5": []}, {
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
    }
    for i in bootstrap_keys:
        case = d[i]
        idx = 0
        if len(case) > 1:  # Multi instance case, pick one sample at random
            idx = int(random.uniform(0, len(case)))
        case = case[idx]
        for i in range(1, 6):
            if case[str(i)]["n_pred"] == 0 and case[str(i)]["n_ref"] == 0:
                # True negative, skip it
                continue
            elif case[str(i)]["n_pred"] > 0 and case[str(i)]["n_ref"] == 0:
                # True positive, set to zero
                dices[str(i)].append(0.0)
                ious[str(i)].append(0.0)
            else:  # Normal append
                dices[str(i)].append(case[str(i)]["Dice"])
                ious[str(i)].append(case[str(i)]["IoU"])

    # Aggregation
    agg_dice, agg_iou = {"1": [], "2": [], "3": [], "4": [], "5": []}, {
        "1": [],
        "2": [],
        "3": [],
        "4": [],
        "5": [],
    }
    for i in range(1, 6):
        if len(dices[str(i)]) > 0:  # Skip if all sampled IDs come from true negatives
            agg_dice[str(i)] = np.nanmean(np.array(dices[str(i)]))
        if len(ious[str(i)]) > 0:
            agg_iou[str(i)] = np.nanmean(np.array(ious[str(i)]))

    return agg_dice, agg_iou


def extract_stats(agg: list) -> dict:
    """
    Extract statistics from aggregated metrics after bootstrapping

    Params
    ------
    agg : list of dict with all metrics for all bootstrap iters for all labels

    Returns
    -------
    stats : dict with stats for every label

    """
    stats = {}
    for i in range(1, 6):
        data = np.array([a[str(i)] for a in agg])
        means = float(np.nanmean(data))
        stds = float(np.nanstd(data))
        p2_5 = float(np.nanpercentile(data, 2.5))
        p97_5 = float(np.nanpercentile(data, 97.5))
        p25 = float(np.nanpercentile(data, 25))
        p75 = float(np.nanpercentile(data, 75))
        median = float(np.nanmedian(data))
        min_ = float(np.nanmin(data))
        max_ = float(np.nanmax(data))

        stats[str(i)] = {
            "mean": means,
            "std": stds,
            "p2_5": p2_5,
            "p97_5": p97_5,
            "p25": p25,
            "p75": p75,
            "median": median,
            "min": min_,
            "max": max_,
        }
    return stats


def main(args):
    infile = args.i
    n = args.n
    id_file = args.ids
    workers = args.np

    assert os.path.exists(infile) and infile.endswith(
        ".json"
    ), f"Input folder '{infile}' does not exist or is not .json"

    outfile = infile.replace(".json", "_bootstrap.json")

    df = load_json(infile)
    assert "metric_per_case" in list(df.keys())
    df = df["metric_per_case"]

    # Clean dictionary to handle multi instance cases
    clean = extract_ids(df)

    # If ids_file exists, bootstrap only for selected IDs
    if os.path.exists(id_file) and id_file.endswith(".txt"):
        ids = np.loadtxt(id_file, dtype=str).tolist()
        clean = {i: val for i, val in clean.items() if i in ids}
        outfile = outfile.replace(
            ".json", f"_{os.path.basename(id_file).replace('.txt', '')}.json"
        )

    # Bootstrapping
    results = Parallel(workers)(delayed(bootstrap_iter)(clean) for _ in range(n))
    results = np.array(results, dtype=object)
    # Dice
    stats_dice = extract_stats(agg=results[:, 0].tolist())
    stats_iou = extract_stats(agg=results[:, -1].tolist())

    # Save results
    out = {"dice": stats_dice, "iou": stats_iou}
    save_json(out, outfile)


def get_args():
    parser = argparse.ArgumentParser(
        description="Run 1000-bootstrap iteration on Dice and IoU results"
    )
    parser.add_argument("--i", help="Input file", required=True, type=str)
    parser.add_argument(
        "--n", help="# bootstrap iterations", required=False, type=int, default=1000
    )
    parser.add_argument(
        "--ids",
        help="Input file with ID information",
        required=False,
        default="",
        type=str,
    )
    parser.add_argument("--np", help="# workers", required=False, type=int, default=4)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    t1 = time.time()
    # Example run: python bootstrap_metrics.py --i /summary/file/from/nnUNet --n 1000 --np 4
    main(get_args())
    print(f"Time ellapsed: {time.time()-t1}sec")
