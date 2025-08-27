# PyQt Click-to-Segment Labeling App (SAM2-ready)

## Features
- Click on an object to segment and preview a translucent mask
- Prompt for class label and extract polygon from the mask
- Save annotations to JSON (LabelMe-like)
- SAM2 checkpoint support (path default: `/home/tao/sam2.1_hiera_base_plus.pt`)
- Fallback classical segmentation (flood fill) if SAM2 is not installed

## Setup

1) Create venv and install base deps
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) (Optional) Install SAM2
- SAM2 is not yet on PyPI. Install from source:
```bash
pip install git+https://github.com/facebookresearch/segment-anything-2.git
```
- Place your checkpoint at `/home/tao/sam2.1_hiera_base_plus.pt` (default used by the app). You can change this in code or later add a settings dialog.

3) Run the app
```bash
python run.py
```

## Usage
- File -> Open Image: choose an image.
- Left-click on the object: app segments and overlays a semi-transparent mask.
- Enter a label when prompted. A polygon is extracted and stored.
- File -> Save Annotations: export a JSON similar to LabelMe.

## Notes
- If SAM2 is unavailable, the app falls back to flood fill around the click color.
- For packaging, you can use `pyinstaller`.