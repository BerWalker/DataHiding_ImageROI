# DWT Image Watermarking

This project embeds a grayscale secret image in a color cover image using a
two-level Haar discrete wavelet transform (DWT). The secret image is resized to
50x50 pixels and converted to black and white before embedding. Each bit is
spread across multiple image coefficients using a fixed key, so extraction
does not require the original cover image.

## Requirements

- Python 3
- Packages listed in `requirements.txt`

Install the dependencies from the project root:

```bash
python -m pip install -r requirements.txt
```

## Embed and extract a watermark

The sample cover and secret images are in `images/input/`. Embed the watermark
and then extract it with:

```bash
python main.py --hide --input-image images/input/cover.png --secret-image images/input/secret.png --output-image images/output/stego.png
python main.py --input-image images/output/stego.png --output-image images/output/recovered_secret.png
```

The extracted image is a binary 50x50 version of the secret. Use PNG for the
watermarked image because lossy formats such as JPEG can damage the watermark.

The watermark size, key, and strength are configured by `WM_SIZE`, `KEY`, and
`STRENGTH` in `main.py`.

## Test robustness

Run the robustness script to test the watermark with Gaussian noise, scribbles,
and downscaling/upscaling:

```bash
python test.py --cover images/input/cover.png --secret images/input/secret.png --out-dir images/output
```

The script writes the watermarked image, each attacked image, and each
extracted watermark to the output directory. It reports the bit error rate
(BER); an attack passes when the BER is at most 0.20. The script exits with a
non-zero status if any attack fails.

## Analyze image texture

`lbp.py` compares pixel and Local Binary Pattern (LBP) statistics and
histograms for a cover image and its watermarked version:

```bash
python lbp.py --cover images/input/cover.png --stego images/output/stego.png --output images/output/histograms.png
```
