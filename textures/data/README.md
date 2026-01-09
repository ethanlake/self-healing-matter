# Data Directory

This directory contains assets for the Neural Cellular Automata texture synthesis demo.

## Structure

### `models/`
Contains JSON-formatted model weights for browser-based inference. These files are converted from PyTorch `.pt` files using `convert_pt_to_json.py`.

- **Format**: JSON with quantized weights, metadata, and noise parameters
- **Count**: 60 texture models
- **Size**: ~50-100 KB per file

### `images/texture/`
Contains reference texture images used as training targets and for display in the demo.

- **Format**: JPG images
- **Count**: 60 textures (matching models)
- **Size**: 10-50 KB per image

### `metadata.json`
A JSON file listing all available texture names. This file is used by the demo to populate the texture selector.

## Adding New Textures

To add a new texture to the demo:

1. Train a model using `streamlined_training_notebook.ipynb` on Google Colab
2. Download the `weights.pt` file
3. Convert to JSON: `python convert_pt_to_json.py weights.pt data/models/texture_name.json`
4. Add the reference texture image to `data/images/texture/texture_name.jpg`
5. Add `"texture_name"` to the `texture_names` array in `metadata.json`

## File Correspondence

Every entry in `metadata.json` should have:
- A corresponding `.json` file in `models/`
- A corresponding `.jpg` file in `images/texture/`

Current count: **60 textures**
