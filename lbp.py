"""Steganalysis module: uses Local Binary Pattern (LBP) to detect LSB steganography.

Idea: LSB embedding adds pseudo-random noise, which destroys the local smoothness
of the image. LBP measures that local texture, so its statistics (mean, standard
deviation, variance) and histogram change when a message is hidden.
"""
import argparse

import cv2
import numpy as np

# Reuse the same ROI and channel used to hide the secret image.
from main import CHANNEL, CHANNELS, ROI_HEIGHT, ROI_WIDTH, ROI_X, ROI_Y

# The 8 neighbors in a 3x3 window as (row offset, column offset),
# clockwise starting from the top-left pixel:
#   1 2 3
#   8 p 4
#   7 6 5
NEIGHBORS = [(-1, -1), (-1, 0), (-1, 1), (0, 1), (1, 1), (1, 0), (1, -1), (0, -1)]


# ---------------------------------------------------------------
# LBP (follows Algorithm 1)
# ---------------------------------------------------------------
def compute_lbp(image):
    """Return the LBP image of a 2D (single channel) image.

    Border pixels have no full 3x3 neighborhood, so they are skipped:
    the result is 2 pixels smaller in each dimension. The pixel (y, x) of
    the input corresponds to the pixel (y - 1, x - 1) of the result.
    """
    image = image.astype(np.int32)  # avoids uint8 overflow in the math
    height, width = image.shape

    # p: the center pixel of every 3x3 window (all pixels except the borders)
    center = image[1:height - 1, 1:width - 1]

    # p' <- 0 (for every pixel at once)
    lbp = np.zeros_like(center)

    # for i = 1 to 8
    for i, (dy, dx) in enumerate(NEIGHBORS, start=1):
        # x_i: the i-th neighbor of every pixel (the image shifted by dy, dx)
        neighbor = image[1 + dy:height - 1 + dy, 1 + dx:width - 1 + dx]

        # x'_i = 0 if x_i < p, otherwise 1
        bit = (neighbor >= center).astype(np.int32)

        # p' <- sum of x'_i * 2^i
        lbp += bit * (2 ** i)

    return lbp


# ---------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------
def get_statistics(lbp):
    return {
        "mean": float(np.mean(lbp)),
        "std": float(np.std(lbp)),
        "variance": float(np.var(lbp)),
    }


def histogram(lbp):
    """Normalized histogram of LBP values (values go from 0 to 510)."""
    counts = np.bincount(lbp.ravel(), minlength=511)
    return counts / counts.sum()


def print_comparison(title, cover_lbp, stego_lbp):
    cover_stats = get_statistics(cover_lbp)
    stego_stats = get_statistics(stego_lbp)

    print(f"\n{title}")
    print(f"{'metric':<10}{'cover':>14}{'stego':>14}{'difference':>14}")
    for name in ("mean", "std", "variance"):
        c, s = cover_stats[name], stego_stats[name]
        print(f"{name:<10}{c:>14.4f}{s:>14.4f}{s - c:>14.4f}")

    # 0 = identical histograms, 2 = completely different histograms
    distance = np.abs(histogram(cover_lbp) - histogram(stego_lbp)).sum()
    print(f"LBP histogram distance (0 to 2): {distance:.4f}")


# ---------------------------------------------------------------
# Test: cover (no message) vs stego (with message)
# ---------------------------------------------------------------
def analyze(cover_path, stego_path):
    cover = cv2.imread(str(cover_path), cv2.IMREAD_COLOR)
    stego = cv2.imread(str(stego_path), cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_path}")
    if stego is None:
        raise FileNotFoundError(f"Could not read stego image: {stego_path}")
    if cover.shape != stego.shape:
        raise ValueError("Cover and stego images must have the same size.")

    # Analyze only the channel where the bits were hidden.
    channel_index = CHANNELS[CHANNEL]
    cover_channel = cover[:, :, channel_index]
    stego_channel = stego[:, :, channel_index]

    changed = int(np.count_nonzero(cover_channel != stego_channel))
    print(f"Pixels changed in channel {CHANNEL}: {changed}")

    cover_lbp = compute_lbp(cover_channel)
    stego_lbp = compute_lbp(stego_channel)

    # Scenario (a) vs (b), whole image.
    print_comparison("Whole image", cover_lbp, stego_lbp)

    # Same comparison, but only inside the ROI (where the message is).
    # LBP index = image index - 1, because the border was skipped.
    rows = slice(ROI_Y - 1, ROI_Y - 1 + ROI_HEIGHT)
    cols = slice(ROI_X - 1, ROI_X - 1 + ROI_WIDTH)
    print_comparison("ROI only", cover_lbp[rows, cols], stego_lbp[rows, cols])


def main():
    parser = argparse.ArgumentParser(description="LBP steganalysis of LSB images.")
    parser.add_argument("--cover", required=True, help="Image WITHOUT hidden message.")
    parser.add_argument("--stego", required=True, help="Image WITH hidden message.")
    args = parser.parse_args()
    analyze(args.cover, args.stego)


if __name__ == "__main__":
    main()