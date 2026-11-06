"""Embed and extract a secret image using the Haar discrete wavelet transform.

Each pixel of the 50x50 secret image is encoded as a +1 or -1 signal. Each bit
is spread across multiple DWT coefficients using a key-generated pattern. The
secret can be extracted by correlating the watermarked image with that pattern;
the original cover image is not required.
"""

import argparse
from pathlib import Path

import cv2
import numpy as np
import pywt

WM_SIZE = 50  # Watermark width and height in pixels.
BLOCK = 4  # A two-level DWT coefficient represents a 4x4 pixel block.
KEY = 1234  # The same key is used for embedding and extraction.
STRENGTH = 46.0  # Higher strength improves robustness but increases visibility.
CLIP = 40.0  # Limit extreme values (for example, scribbles) during extraction.


def to_gray(img):
    """Convert a BGR image to grayscale as floating-point values."""
    return 0.114 * img[..., 0] + 0.587 * img[..., 1] + 0.299 * img[..., 2]


def crop(img):
    """Crop an image to dimensions divisible by four, as required by the DWT."""
    h, w = img.shape[:2]
    return img[: h - h % BLOCK, : w - w % BLOCK]


def get_pattern(coef_shape):
    """Generate the +/-1 pattern and assign coefficients to each watermark bit."""
    total = coef_shape[0] * coef_shape[1]
    nbits = WM_SIZE * WM_SIZE
    per_bit = total // nbits
    if per_bit < 1:
        raise ValueError("Image is too small for this watermark.")

    rng = np.random.default_rng(KEY)
    pattern = rng.choice([-1.0, 1.0], size=total)
    groups = rng.permutation(total)[: nbits * per_bit].reshape(nbits, per_bit)
    return pattern, groups, per_bit


def measure(gray):
    """Measure how strongly the image matches the pattern for each bit."""
    approx = pywt.wavedec2(gray, "haar", level=2)[0]

    # Remove the smooth image component, which would interfere with the measurement.
    detail = approx - cv2.blur(approx, (3, 3), borderType=cv2.BORDER_REFLECT)
    detail = np.clip(detail, -CLIP, CLIP).ravel()

    pattern, groups, _ = get_pattern(approx.shape)
    return (detail[groups] * pattern[groups]).mean(axis=1)


def expected_secret(secret_path):
    """Return the expected watermark as a 50x50 boolean array."""
    secret = cv2.imread(str(secret_path), cv2.IMREAD_GRAYSCALE)
    if secret is None:
        raise FileNotFoundError(f"Could not read secret image: {secret_path}")
    secret = cv2.resize(secret, (WM_SIZE, WM_SIZE))
    return secret > 127


def hide_secret(cover_path, secret_path, output_path):
    cover = cv2.imread(str(cover_path), cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_path}")

    # White maps to +1 and black maps to -1.
    signs = np.where(expected_secret(secret_path), 1.0, -1.0).ravel()

    base = crop(cover).astype(np.float64)
    coef_shape = (base.shape[0] // BLOCK, base.shape[1] // BLOCK)
    pattern, groups, per_bit = get_pattern(coef_shape)

    # Required match strength for each bit.
    target = max(6.0, STRENGTH / np.sqrt(per_bit))

    added = np.zeros(coef_shape[0] * coef_shape[1])
    stego = base
    for _ in range(8):
        # Reinforce only the bits that have not yet reached the target.
        missing = np.maximum(target - signs * measure(to_gray(stego)), 0.0)
        if missing.max() < 0.05 * target:
            break
        added[groups] += (signs * missing)[:, None] * pattern[groups]

        # Map coefficient adjustments back to pixels in 4x4 blocks.
        delta = np.kron(added.reshape(coef_shape), np.ones((BLOCK, BLOCK))) / BLOCK
        stego = np.clip(np.rint(base + delta[..., None]), 0, 255)

    result = cover.copy()
    result[: stego.shape[0], : stego.shape[1]] = stego.astype(np.uint8)

    cv2.imwrite(str(output_path), result)  # PNG preserves the embedded watermark.
    print(f"Secret image hidden in '{output_path}'")


def reveal_secret(stego_path, output_path):
    stego = cv2.imread(str(stego_path), cv2.IMREAD_COLOR)
    if stego is None:
        raise FileNotFoundError(f"Could not read stego image: {stego_path}")

    score = measure(to_gray(crop(stego)))
    bits = (score > 0).reshape(WM_SIZE, WM_SIZE)

    cv2.imwrite(str(output_path), (bits * 255).astype(np.uint8))
    print(f"Secret image extracted to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Embed or extract an image watermark using the DWT."
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Embed the watermark. Without this flag, extract the watermark.",
    )
    parser.add_argument(
        "-i", "--input-image", "--input-img", dest="input_image", required=True,
        help="Cover image when embedding or watermarked image when extracting.",
    )
    parser.add_argument(
        "-s", "--secret-image", "--secret-img", dest="secret_image",
        help="Secret image to embed (required with --hide).",
    )
    parser.add_argument(
        "-o", "--output-image", "--output-img", dest="output_image", required=True,
        help=(
            "Output watermarked image when embedding or extracted image "
            "when extracting."
        ),
    )
    args = parser.parse_args()

    if args.hide:
        if not args.secret_image:
            parser.error("--hide requires --secret-image")
        hide_secret(
            Path(args.input_image),
            Path(args.secret_image),
            Path(args.output_image),
        )
    else:
        reveal_secret(Path(args.input_image), Path(args.output_image))


if __name__ == "__main__":
    main()
