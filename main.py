import argparse
from pathlib import Path

import cv2

# Region of interest settings.
ROI_X = 30
ROI_Y = 30
ROI_WIDTH = 50
ROI_HEIGHT = 50
CHANNEL = "G"

# OpenCV stores color channels in BGR order.
CHANNELS = {"B": 0, "G": 1, "R": 2}


def validate_roi(image):
    """Ensure the region of interest fits inside the image."""
    height, width = image.shape[:2]
    if ROI_X + ROI_WIDTH > width or ROI_Y + ROI_HEIGHT > height:
        raise ValueError(
            f"Region of interest is outside the image. "
            f"Image size: {width}x{height} (width x height)."
        )


def hide_secret(cover_path, secret_path, output_path):
    cover = cv2.imread(str(cover_path), cv2.IMREAD_COLOR)
    if cover is None:
        raise FileNotFoundError(f"Could not read cover image: {cover_path}")

    secret = cv2.imread(str(secret_path), cv2.IMREAD_GRAYSCALE)
    if secret is None:
        raise FileNotFoundError(f"Could not read secret image: {secret_path}")

    validate_roi(cover)

    secret = cv2.resize(secret, (ROI_WIDTH, ROI_HEIGHT))
    bits = (secret > 127).astype("uint8")

    channel_index = CHANNELS[CHANNEL]
    roi = cover[
        ROI_Y : ROI_Y + ROI_HEIGHT,
        ROI_X : ROI_X + ROI_WIDTH,
        channel_index,
    ]
    cover[
        ROI_Y : ROI_Y + ROI_HEIGHT,
        ROI_X : ROI_X + ROI_WIDTH,
        channel_index,
    ] = (roi & 0xFE) | bits

    # PNG is lossless, so it preserves the embedded least significant bits.
    cv2.imwrite(str(output_path), cover)
    print(f"Secret image hidden in '{output_path}'")


def reveal_secret(stego_path, output_path):
    stego = cv2.imread(str(stego_path), cv2.IMREAD_COLOR)
    if stego is None:
        raise FileNotFoundError(f"Could not read stego image: {stego_path}")

    validate_roi(stego)

    channel_index = CHANNELS[CHANNEL]
    roi = stego[
        ROI_Y : ROI_Y + ROI_HEIGHT,
        ROI_X : ROI_X + ROI_WIDTH,
        channel_index,
    ]
    bits = roi & 1
    secret = (bits * 255).astype("uint8")

    cv2.imwrite(str(output_path), secret)
    print(f"Secret image extracted to '{output_path}'")


def main():
    parser = argparse.ArgumentParser(
        description="Hide and extract images using LSB steganography."
    )
    parser.add_argument(
        "--hide",
        action="store_true",
        help="Hide the secret image. Extract it when this flag is omitted.",
    )
    parser.add_argument(
        "-i",
        "--input-image",
        "--input-img",
        dest="input_image",
        required=True,
        help="Cover image when hiding; stego image when extracting.",
    )
    parser.add_argument(
        "-s",
        "--secret-image",
        "--secret-img",
        dest="secret_image",
        help="Secret image to hide (required with --hide).",
    )
    parser.add_argument(
        "-o",
        "--output-image",
        "--output-img",
        dest="output_image",
        required=True,
        help="Output stego image when hiding; extracted image otherwise.",
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
