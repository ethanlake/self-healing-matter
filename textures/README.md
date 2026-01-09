# Neural Cellular Automata Texture Synthesis

This directory contains a complete implementation of Neural Cellular Automata (NCA) for texture synthesis, including training code, model conversion utilities, and an interactive WebGL-based demo.

## Overview

Neural Cellular Automata learn to generate textures through local update rules. Each cell in a grid updates based on its neighborhood, creating emergent patterns that match target textures. This implementation uses noise-initialized seeds and perceptual loss for training.

## Quick Start

### Training a Model (Google Colab)

1. Open `streamlined_training_notebook.ipynb` in Google Colab
2. Upload your target texture image (recommended max size: 128x128)
3. Run all cells to train the model
4. Download the `weights.pt` file when training completes

### Adding to Demo

After training, convert and add your model to the browser demo:

```bash
# Convert PyTorch weights to JSON format
python convert_pt_to_json.py weights.pt data/models/my_texture.json

# Add the reference texture image
cp my_texture.jpg data/images/texture/my_texture.jpg

# Add the texture name to metadata
# Edit data/metadata.json and add "my_texture" to the texture_names array
```

### Running the Demo

The demo requires a local web server due to browser security restrictions on loading local files.

**Option 1: Using Python's built-in server (from repository root)**
```bash
# Navigate to repository root (self-healing-matter/)
cd /path/to/self-healing-matter

# Start server (Python 3)
python3 -m http.server 8000

# Open browser to:
# http://localhost:8000/textures/demo/index.html
```

**Option 2: Using Node.js http-server**
```bash
# Install http-server globally (once)
npm install -g http-server

# Navigate to repository root
cd /path/to/self-healing-matter

# Start server
http-server -p 8000

# Open browser to:
# http://localhost:8000/textures/demo/index.html
```

The demo supports:
- Interactive painting and erasing
- Real-time NCA simulation on GPU via WebGL
- Parameter adjustment (noise, timestep, scale)
- Multiple texture models

## Directory Structure

```
textures/
├── README.md                              # This file
├── streamlined_training_notebook.ipynb    # Primary training interface
├── models.py                              # NCA model definitions (NoiseNCA, PENCA)
├── convert_pt_to_json.py                  # Weight conversion utility
├── loss.py                                # Perceptual loss functions
├── texture_generator.py                   # Texture generation utilities
├── download_textures.py                   # Texture dataset download script
├── requirements.txt                       # Python dependencies
├── analysis/                              # Analysis tools
│   ├── lyapunov.py                        # Lyapunov exponent analysis
│   ├── time_crystal.py                    # Time crystal analysis
│   ├── calibrate.py                       # Model calibration utilities
│   └── visualize.py                       # Visualization utilities
├── demo/                                  # Interactive WebGL demo
│   ├── index.html                         # Demo page
│   ├── demo.js                            # Demo controller
│   ├── noiseNCA.js                        # WebGL NCA implementation
│   ├── style.css                          # Demo styling
│   ├── dat.gui.min.js                     # UI library
│   └── twgl.min.js                        # WebGL library
├── data/                                  # Model assets
│   ├── README.md                          # Asset documentation
│   ├── metadata.json                      # Texture list (58 textures)
│   ├── models/                            # JSON model weights (58 files)
│   │   ├── bubbly_0101.json
│   │   ├── flames.json
│   │   └── ...
│   └── images/texture/                    # Reference images (58 files)
│       ├── bubbly_0101.jpg
│       ├── flames.jpg
│       └── ...
├── trained_models/                        # PyTorch model weights (.pt files)
│   ├── Vanilla-NCA/                       # Basic NCA models (48 textures)
│   ├── Noise-NCA/                         # Noise-seeded models (65 textures)
│   └── PE-NCA/                            # Positional encoding models (48 textures)
├── static/                                # Web assets (CSS, JS, fonts)
│   ├── css/
│   ├── js/
│   └── images/
└── utils/                                 # Python utilities
    ├── misc.py
    └── video_utils.py
```

## Model Architectures

### NoiseNCA (Standard)
The primary model architecture used in this project:
- **Perception**: 4 filters (identity, sobel_x, sobel_y, laplacian)
- **Processing**: 2-layer MLP with ReLU activation
- **Seed**: Random noise initialization (configurable noise level)
- **Parameters**: ~5,856 (12 channels, 96 hidden units)

### PENCA (Positional Encoding NCA)
Variant with 2D positional encoding as conditional channels:
- Same perception as NoiseNCA
- Additional 2D grid coordinates as input
- Enables position-dependent pattern generation

## Training Details

- **Loss Function**: RelaxedOTLoss (perceptual texture loss based on VGG features)
- **Optimizer**: Adam with learning rate 1e-3
- **Learning Rate Schedule**: MultiStepLR (decay at steps 1000, 2000)
- **Batch Size**: 4
- **Training Iterations**: 3000
- **Pool Size**: 256 states (for batch diversity)
- **Step Range**: Random 32-96 steps per iteration

## File Formats

### PyTorch Weights (.pt)
Training produces PyTorch state_dict files containing:
- `w1.weight`: First MLP layer weights [fc_dim, 4*chn, 1, 1]
- `w1.bias`: First MLP layer bias [fc_dim]
- `w2.weight`: Second MLP layer weights [chn, fc_dim, 1, 1]
- `noise_level`: Seed noise level (scalar)

These are stored in `trained_models/` and serve as the source files for JSON conversion.

### JSON Weights
Browser-compatible format with:
- Quantized weights (normalized to [0, 1])
- Scale and center for dequantization
- Layer shapes and layouts for GPU texture storage
- Noise level and metadata

## Dependencies

### Python
```
torch
torchvision
numpy
requests
moviepy
wandb
tqdm
```

Install with: `pip install -r requirements.txt`

### JavaScript
All required libraries are included:
- `twgl.min.js` - WebGL utilities
- `dat.gui.min.js` - UI controls

## Analysis Tools

The `analysis/` directory contains specialized tools for studying NCA dynamics:
- **lyapunov.py**: Compute Lyapunov exponents for stability analysis
- **time_crystal.py**: Detect time-crystal behavior in NCA patterns
- **calibrate.py**: Calibration utilities for model parameters
- **visualize.py**: Advanced visualization tools

## Browser Demo Features

- **Interactive Painting**: Click and drag to paint patterns
- **Model Selection**: Choose from 58 pre-trained textures
- **Parameter Control**:
  - `dt`: Timestep (integration step size)
  - `dx/dy`: Pattern scale factors
  - `epsilon`: Noise level during simulation
  - Rotation angle, alignment mode
- **Visualization Modes**:
  - RGB output (first 3 channels)
  - Individual hidden channels
  - Grayscale mode
- **Performance**: Real-time GPU-accelerated simulation via WebGL2

## Tips for Training

1. **Image Size**: Keep target images around 128x128 for faster training
2. **Noise Level**: Start with 0.1; increase for more chaotic patterns
3. **Training Duration**: 3000 iterations usually sufficient; watch loss curve
4. **Pattern Scale**: If patterns are too large/small, adjust `dx`/`dy` in demo
5. **Overflow**: If you see artifacts, the model may be unstable; try retraining with lower learning rate

## Credits

Based on the self-organizing texture synthesis work using Neural Cellular Automata. This implementation extends the original NCA concept with noise-based initialization and improved training stability.

## License

Apache License 2.0 (see noiseNCA.js header for full license text)
