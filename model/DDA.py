import argparse
from multiprocessing import Pool, cpu_count
import numpy as np
from pathlib import Path
from PIL import Image


def _img_mean_std(img_path):
    with Image.open(img_path) as im:
        arr = np.asarray(im, dtype=np.float32) / 255.0
    mean = arr.mean(axis=(0, 1))
    std = arr.std(axis=(0, 1))
    return mean, std

def DDA_average(folder_path):
    jpgs = list(Path(folder_path).rglob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No image found in {folder_path}.")

    with Pool(cpu_count()) as pool:
        stats = pool.map(_img_mean_std, jpgs)

    means = np.stack([m for m, _ in stats])
    stds = np.stack([s for _, s in stats])

    pixel_means = means.mean(axis=0)
    pixel_stds = stds.mean(axis=0)

    print(f"PIXEL_MEAN: {[round(x, 4) for x in pixel_means]}")
    print(f"PIXEL_STD: {[round(x, 4) for x in pixel_stds]}")


def _img_sums(img_path):
    with Image.open(img_path) as im:
        arr = np.asarray(im, dtype=np.float64) / 255.0
    flat = arr.reshape(-1, 3)
    return flat.sum(0), (flat ** 2).sum(0), flat.shape[0]

def DDA_global(folder_path):
    jpgs = list(Path(folder_path).rglob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No image found in {folder_path}.")

    with Pool(cpu_count()) as pool:
        results = pool.map(_img_sums, jpgs)

    sum_c = sum(r[0] for r in results)
    sumsq_c = sum(r[1] for r in results)
    n_pix = sum(r[2] for r in results)

    mean = sum_c / n_pix
    std = np.sqrt(sumsq_c / n_pix - mean ** 2)

    print(f"PIXEL_MEAN: {[round(x, 4) for x in mean]}")
    print(f"PIXEL_STD: {[round(x, 4) for x in std]}")


def _img_sums_resized(img_path, w=128, h=256):
    with Image.open(img_path) as im:
        arr = np.asarray(im.resize((w, h), Image.Resampling.LANCZOS), dtype=np.float64) / 255.0
    flat = arr.reshape(-1, 3)
    return flat.sum(0), (flat ** 2).sum(0), flat.shape[0]

def DDA_global_resized(folder_path):
    jpgs = list(Path(folder_path).rglob("*.jpg"))
    if not jpgs:
        raise FileNotFoundError(f"No image found in {folder_path}.")

    with Pool(cpu_count()) as pool:
        results = pool.map(_img_sums_resized, jpgs)

    sum_c = sum(r[0] for r in results)
    sumsq_c = sum(r[1] for r in results)
    n_pix = sum(r[2] for r in results)

    mean = sum_c / n_pix
    std = np.sqrt(sumsq_c / n_pix - mean ** 2)

    print(f"PIXEL_MEAN: {[round(x, 4) for x in mean]}")
    print(f"PIXEL_STD: {[round(x, 4) for x in std]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser( description="Calculate the statistics of the dataset to perform DDA.")
    parser.add_argument("folder_path", type=str, help="The path of the training set.")
    args = parser.parse_args()

    # Implementation of DDA module in the paper
    DDA_average(args.folder_path)

    # Some optional calculation methods
    # DDA_global(args.folder_path)
    # DDA_global_resized(args.folder_path)
