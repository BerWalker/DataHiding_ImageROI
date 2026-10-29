# Image Steganography

This project hides a grayscale secret image in the least significant bits (LSBs)
of a selected color channel in a cover image. The secret image is resized to
the configured region of interest (ROI) and converted to black and white before
embedding.

## Requirements

- Python 3
- Packages listed in `requirements.txt`

Install the dependencies:

```powershell
python -m pip install -r requirements.txt
```

## Run a smoke test

The sample cover and secret images are in `images/input/`. Run these commands
from the project root:

```powershell
python main.py --hide --input-image images/input/cover.jpg --secret-image images/input/secret.png --output-image images/output/stego.png
python main.py --input-image images/output/stego.png --output-image images/output/recovered_secret.png
```

The first command embeds the secret image. The second extracts it. Compare
`images/output/recovered_secret.png` with the original secret image after
resizing and thresholding; the recovered image is the binary version used for
embedding.

## Command reference

Hide an image:

```powershell
python main.py --hide -i <cover-image> -s <secret-image> -o <stego-output.png>
```

Extract an image:

```powershell
python main.py -i <stego-image> -o <recovered-image.png>
```

Use PNG for the stego image because lossy formats such as JPEG can destroy the
embedded bits. The ROI position, size, and color channel are configured by
`ROI_X`, `ROI_Y`, `ROI_WIDTH`, `ROI_HEIGHT`, and `CHANNEL` in `main.py`.
