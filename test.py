"""Test watermark robustness against noise, scribbles, and resizing.

For each attack, the script saves the attacked image and its extracted
watermark in the output directory. The bit error rate (BER) is the fraction
of extracted watermark pixels that differ from the expected watermark:
0 is a perfect match and about 0.5 is equivalent to guessing.
"""

import argparse
import importlib.util
from pathlib import Path

import cv2
import numpy as np

THRESHOLD = 0.20  # Maximum BER for a successful extraction.
SEED = 42


# Each attack accepts (image, rng) so results are reproducible.
def gaussian_noise(img, rng):
    sigma = 0.10 * 255  # 10% of the intensity range.
    noisy = img.astype(np.float64) + rng.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def _random_lines(out, rng, n, thickness):
    h, w = out.shape[:2]
    for _ in range(n):
        p1 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        p2 = (int(rng.integers(0, w)), int(rng.integers(0, h)))
        color = tuple(int(c) for c in rng.integers(0, 256, 3))
        cv2.line(out, p1, p2, color, thickness)


def scribbles(img, rng):
    out = img.copy()
    _random_lines(out, rng, 40, 3)
    return out


def scribbles_heavy(img, rng):
    """Apply many thick strokes across the image."""
    out = img.copy()
    _random_lines(out, rng, 80, 4)
    return out


def resize_half(img, rng):
    h, w = img.shape[:2]
    small = cv2.resize(img, (w // 2, h // 2))  # Downscale to 50%.
    return cv2.resize(small, (w, h))  # Restore the original dimensions.


def resize_double(img, rng):
    h, w = img.shape[:2]
    big = cv2.resize(img, (w * 2, h * 2))  # Upscale to 200%.
    return cv2.resize(big, (w, h))  # Restore the original dimensions.


ATTACKS = {
    "no_attack": lambda img, rng: img.copy(),  # Control case.
    "gauss_10pct": gaussian_noise,
    "scribbles": scribbles,
    "heavy_scribbles": scribbles_heavy,
    "resize_0.5": resize_half,
    "resize_2x": resize_double,
}


def load_algorithm(path):
    spec = importlib.util.spec_from_file_location("algoritmo", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    parser = argparse.ArgumentParser(
        description="Test the robustness of an image watermark."
    )
    parser.add_argument("-c", "--cover", required=True, help="Path to the cover image.")
    parser.add_argument(
        "-s", "--secret", required=True, help="Path to the secret image."
    )
    parser.add_argument(
        "--algo", default="main.py", help="Path to the watermark algorithm."
    )
    parser.add_argument(
        "--out-dir",
        help="Directory for generated images (defaults to the cover directory).",
    )
    args = parser.parse_args()

    algo = load_algorithm(args.algo)
    cover = Path(args.cover)
    out_dir = Path(args.out_dir) if args.out_dir else cover.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    expected = algo.expected_secret(args.secret)

    # Embed the watermark.
    stego_path = out_dir / f"{cover.stem}_stego.png"
    algo.hide_secret(cover, Path(args.secret), stego_path)
    stego = cv2.imread(str(stego_path))

    # Apply each attack, extract the watermark, and compare the results.
    print(f"\n{'Attack':<16}{'BER':>8}  Result")
    all_ok = True
    for name, attack in ATTACKS.items():
        rng = np.random.default_rng(SEED)
        attacked_path = out_dir / f"{cover.stem}_atk_{name}.png"
        extracted_path = out_dir / f"{cover.stem}_atk_{name}_extracted.png"
        cv2.imwrite(str(attacked_path), attack(stego, rng))
        algo.reveal_secret(attacked_path, extracted_path)

        extracted = cv2.imread(str(extracted_path), cv2.IMREAD_GRAYSCALE) > 127
        ber = np.mean(extracted != expected)
        ok = ber <= THRESHOLD
        all_ok &= ok
        print(f"{name:<16}{ber:>8.3f}  {'OK' if ok else 'FAILED'}")

    raise SystemExit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
