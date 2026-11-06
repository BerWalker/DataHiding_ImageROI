"""Steganalysis: compares a cover image with a watermarked one using LBP.

The watermark changes the local texture of the image, so the statistics and
the histogram of the LBP (and of the pixels) change after embedding.
"""
import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np

from main import crop, to_gray

# The 8 neighbors in a 3x3 window as (row offset, column offset),
# clockwise starting from the top-left pixel:
#   1 2 3
#   8 p 4
#   7 6 5
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


def compute_lbp(image):
    """LBP of a 2D image (borders are skipped: result is 2 pixels smaller)."""
    image = image.astype(np.int32)
    h, w = image.shape
    center = image[1:h - 1, 1:w - 1]
    lbp = np.zeros_like(center)

    for i, (dy, dx) in enumerate(NEIGHBORS, start=1):
        neighbor = image[1 + dy:h - 1 + dy, 1 + dx:w - 1 + dx]
        lbp += (neighbor >= center).astype(np.int32) * (2 ** i)

    return lbp


def histogram(values, bins):
    """Normalized histogram of integer values."""
    counts = np.bincount(values.ravel(), minlength=bins)
    return counts / counts.sum()


def print_comparison(title, cover, stego, bins):
    print(f"\n{title}")
    print(f"{'metric':<10}{'cover':>14}{'stego':>14}{'difference':>14}")
    for name, func in (("mean", np.mean), ("std", np.std), ("variance", np.var)):
        c, s = func(cover), func(stego)
        print(f"{name:<10}{c:>14.4f}{s:>14.4f}{s - c:>14.4f}")

    # 0 = identical histograms, 2 = completely different histograms
    distance = np.abs(histogram(cover, bins) - histogram(stego, bins)).sum()
    print(f"Histogram distance (0 to 2): {distance:.4f}")


def plot_histograms(cover_pix, stego_pix, cover_lbp, stego_lbp, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Pixel histogram
    ax1.plot(histogram(cover_pix, 256), label="before")
    ax1.plot(histogram(stego_pix, 256), label="after", alpha=0.7)
    ax1.set_title("Pixel histogram")
    ax1.set_xlabel("Pixel value")

    # LBP histogram (values are always even, so plot only even positions)
    x = np.arange(0, 511, 2)
    ax2.plot(x, histogram(cover_lbp, 511)[::2], label="before")
    ax2.plot(x, histogram(stego_lbp, 511)[::2], label="after", alpha=0.7)
    ax2.set_title("LBP histogram")
    ax2.set_xlabel("LBP value")

    for ax in (ax1, ax2):
        ax.set_ylabel("Frequency")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    print(f"\nHistograms saved to '{output_path}'")


def analyze(cover_path, stego_path, output_path):
    cover = cv2.imread(cover_path, cv2.IMREAD_COLOR)
    stego = cv2.imread(stego_path, cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_path}")
    if stego is None:
        raise FileNotFoundError(f"Could not read stego image: {stego_path}")
    if cover.shape != stego.shape:
        raise ValueError("Cover and stego images must have the same size.")

    # Same gray image used by the watermark (cropped to multiples of 4)
    cover_gray = np.rint(to_gray(crop(cover))).astype(np.uint8)
    stego_gray = np.rint(to_gray(crop(stego))).astype(np.uint8)

    changed = int(np.count_nonzero(cover_gray != stego_gray))
    print(f"Pixels changed: {changed} of {cover_gray.size}")

    cover_lbp = compute_lbp(cover_gray)
    stego_lbp = compute_lbp(stego_gray)

    print_comparison("Pixels", cover_gray, stego_gray, 256)
    print_comparison("LBP", cover_lbp, stego_lbp, 511)

    plot_histograms(cover_gray, stego_gray, cover_lbp, stego_lbp, output_path)


def main():
    parser = argparse.ArgumentParser(
        description="LBP steganalysis of the DWT watermark."
    )
    parser.add_argument("--cover", required=True, help="Image WITHOUT watermark.")
    parser.add_argument("--stego", required=True, help="Image WITH watermark.")
    parser.add_argument("--output", default="histograms.png", help="Plot file to save.")
    args = parser.parse_args()
    analyze(args.cover, args.stego, args.output)


if __name__ == "__main__":
    main()
