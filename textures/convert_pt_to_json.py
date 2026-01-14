#!/usr/bin/env python3
"""
Convert PyTorch .pt model files to JSON format for NoiseNCA demo.

Usage:
    python convert_pt_to_json.py <input.pt> [output.json] [--noise-level FLOAT] [--pos-emb]

This script follows the conversion convention from export.ipynb.
It expects PyTorch state_dict with keys like 'w1.weight', 'w1.bias', 'w2.weight', etc.
"""

import os
import json
import numpy as np
import torch
import argparse
import sys


def tile2d(a, w=None):
    """
    Tile a 3D array into a 2D grid.
    
    Args:
        a: 3D numpy array of shape [n, h, w, ...]
        w: Width of the grid (if None, uses sqrt of n)
        
    Returns:
        2D tiled array
    """
    a = np.asarray(a)
    if w is None:
        w = int(np.ceil(np.sqrt(len(a))))
    th, tw = a.shape[1:3]
    pad = (w - len(a)) % w
    a = np.pad(a, [(0, pad)] + [(0, 0)] * (a.ndim - 1), 'constant')
    h = len(a) // w
    a = a.reshape([h, w] + list(a.shape[1:]))
    a = np.rollaxis(a, 2, 1).reshape([th * h, tw * w] + list(a.shape[4:]))
    return a


def detect_model_type(state_dict, chn=12):
    """
    Detect if model is RD or NCA based on w1 input dimensions.

    Args:
        state_dict: PyTorch state dictionary
        chn: Number of output channels (default 12)

    Returns:
        'rd' for RD models, 'nca' for NCA models
    """
    if 'w1.weight' not in state_dict:
        raise ValueError("Could not find 'w1.weight' in state_dict")

    w1_shape = state_dict['w1.weight'].shape  # [fc_dim, in_channels, 1, 1]
    input_dim = w1_shape[1]

    # RD models: w1 takes raw state directly (chn channels)
    # The laplacian diffusion is computed separately, not as perception input
    if input_dim == chn:
        print(f"Detected RD model: input_dim={input_dim} = {chn} (state-only, no perception)")
        return 'rd'
    elif input_dim == 4 * chn:
        print(f"Detected NCA model: input_dim={input_dim} = 4 × {chn} (identity + sobel_x + sobel_y + laplacian)")
        return 'nca'
    else:
        print(f"Warning: Unknown model type with input_dim={input_dim}, channel_n={chn}, assuming NCA")
        return 'nca'


@torch.no_grad()
def torch_model_to_np(state_dict):
    """
    Convert PyTorch state_dict to numpy arrays following the notebook convention.

    Returns:
        List of numpy arrays: [layer1_params, layer2_params, ..., noise_level]
    """
    layers = []

    # Layer 1: w1.weight and w1.bias
    if 'w1.weight' not in state_dict:
        raise ValueError("Could not find 'w1.weight' in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")

    w1 = state_dict['w1.weight'][:, :, 0, 0].detach().cpu().numpy()
    # w1 shape: [fc_dim, in_channels] where in_channels = chn*4 for NCA or chn*2 for RD

    b1 = state_dict['w1.bias'][:, None].detach().cpu().numpy()
    # b1 shape: [fc_dim, 1]

    # Layer 2: w2.weight (no bias)
    if 'w2.weight' not in state_dict:
        raise ValueError("Could not find 'w2.weight' in state_dict. "
                        f"Available keys: {list(state_dict.keys())[:10]}...")

    w2 = state_dict['w2.weight'][:, :, 0, 0].detach().cpu().numpy()
    # w2 shape: [chn, fc_dim] (after extracting [:, :, 0, 0] from [chn, fc_dim, 1, 1])

    # Concatenate and transpose: [chn, fc_dim] -> [chn+1, fc_dim] after adding bias
    layer1_params = np.concatenate([w1, b1], axis=1).T
    # layer1_params shape: [chn+1, fc_dim]
    layer1_params = layer1_params[None, ...]  # Add batch dimension: [1, chn+1, fc_dim]
    layers.append(layer1_params)

    # w2 shape after transpose: [in_channels, out_channels]
    w2 = w2.T
    w2 = w2[None, ...]  # Add batch dimension: [1, in_channels, out_channels]
    layers.append(w2)

    # Noise level
    if 'noise_level' in state_dict:
        noise_level = state_dict['noise_level'].item()
    else:
        noise_level = 0.1  # Default
    layers.append(noise_level)

    return layers


def export_np_models_to_json(np_models, metadata):
    """
    Export numpy models in a form that the JavaScript can read.

    Args:
        np_models: List from torch_model_to_np: [layer1, layer2, ..., noise_level]
        metadata: Dict with 'model_names', 'pos_emb' (bool), and 'model_type' (str)

    Returns:
        Dictionary in JSON format
    """
    models_js = {
        'model_names': metadata['model_names'],
        'layers': [],
        'noise_level': np_models[-1],
        'model_type': metadata.get('model_type', 'nca')  # Add model type to JSON
    }

    for i, layer in enumerate(np_models[:-1]):
        # layer shape: [n, c_in, fc_dim] where n=1 for single model
        shape = layer[0].shape  # Original shape: [c_in, fc_dim]
        layer = np.array(layer)  # shape: [n, c_in, fc_dim]

        if i == 0:
            # First layer: handle pos_emb and filter arrangement
            c = 1  # Bias
            if metadata['pos_emb']:
                c += 2  # Positional embedding channels

            filter_channels = layer.shape[1] - c  # Number of filter input channels

            if metadata.get('model_type') == 'rd':
                # RD model: takes raw state directly (no perception filters)
                # filter_channels == chn (state channels only)
                # No rearrangement needed - weights are already in correct order
                chn = filter_channels
                print(f"RD model layer 0: {chn} state channels (no perception filters)")

            else:
                # Standard NCA 4-filter model
                chn = filter_channels // 4
                if filter_channels % 4 != 0:
                    raise ValueError(f"Expected NCA 4-filter model but filter_channels ({filter_channels}) is not divisible by 4")

                # Rearrange filter channels (4 filters: id, sobelx, sobely, lap)
                s = layer[:, :-c].shape  # [n, 4 * chn, fc_dim]
                # Reconstruct the layer to avoid in-place assignment issues
                rearranged_filters = (layer[:, :-c]
                                .reshape(s[0], chn, 4, s[2])  # [n, chn, 4, fc_dim]
                                .transpose(0, 2, 1, 3)  # [n, 4, chn, fc_dim]
                                .reshape(s))  # [n, 4 * chn, fc_dim]
                layer = np.concatenate([rearranged_filters, layer[:, -c:]], axis=1)  # [n, 4 * chn + c, fc_dim]
                print(f"NCA model layer 0: {chn} channels × 4 filters = {filter_channels} filter inputs")

            # Update shape to reflect dimensions
            shape = [layer.shape[1], layer.shape[2]]  # [c_in, fc_dim]
        
        # Pad channels to be multiple of 4 (WebGL requirement)
        s = layer.shape  # [n, c_in, fc_dim]
        layer = np.pad(layer, ((0, 0), (0, 0), (0, (4 - s[2]) % 4)), mode='constant')
        # After padding: [n, c_in, padded_fc_dim] where padded_fc_dim is multiple of 4
        # Update shape to reflect fc_dim padding
        shape = [shape[0], layer.shape[2]]  # [c_in, padded_fc_dim]
        
        # Reshape for RGBA packing: [n, c_in, padded_fc_dim] -> [n, c_in, padded_fc_dim//4, 4]
        layer = layer.reshape(s[0], s[1], -1, 4)  # [n, c_in, fc_dim//4, 4]
        
        # Tile into 2D grid
        n, ht, wt = layer.shape[:3]  # n=1, ht=c_in, wt=fc_dim//4
        w = 1
        while w < n and w * wt < (n + w - 1) // w * ht:
            w += 1
        layer = tile2d(layer, w)
        layout = (w, (n + w - 1) // w)
        
        # Compute scale and center for quantization
        scale = float(layer.max() - layer.min())
        center = float(-layer.min() / scale) if scale > 0 else 0.0
        
        # Normalize to [0, 1]
        layer_normalized = layer - layer.min()
        if scale > 0:
            layer_normalized = layer_normalized / scale
        else:
            layer_normalized = np.zeros_like(layer_normalized)
        
        # Store normalized float values (not quantized to uint8)
        layer_flatten = layer_normalized.flatten()
        
        # Create layer dictionary
        layer_js = {
            'scale': scale,
            'center': center,
            'data_flatten': list(map(float, layer_flatten)),
            'data_shape': list(layer.shape),  # Final tiled shape
            'shape': list(shape),  # Original shape [c_in, fc_dim]
            'layout': list(layout),
            'pos_emb': (i == 0) and metadata['pos_emb'],
            'bias': (i == 0),  # Only first layer has bias
        }
        models_js['layers'].append(layer_js)
    
    return models_js


def convert_pt_to_json(pt_path, output_path=None, noise_level=None, pos_emb=False, dt=None, growing_mode=False):
    """
    Convert a PyTorch .pt file to JSON format for NoiseNCA.

    Args:
        pt_path: Path to input .pt file
        output_path: Path to output .json file (default: same name as input)
        noise_level: Noise level value (if not in state_dict)
        pos_emb: Whether the model uses positional embedding
        dt: Timestep for RD models (if not in model)
        growing_mode: Enable Growing NCA mode (normalized Sobel, stochastic updates)

    Returns:
        Path to the created JSON file
    """
    print(f"Loading PyTorch model from: {pt_path}")
    state_dict = torch.load(pt_path, map_location='cpu')

    # Extract training parameters if present (saved by rd_training.ipynb)
    training_params = None
    if '_training_params' in state_dict:
        training_params = state_dict['_training_params']
        print(f"Found training parameters in weights file:")
        for key, value in training_params.items():
            print(f"  - {key}: {value}")
        # Remove from state_dict so it doesn't interfere with model loading
        del state_dict['_training_params']

    # Detect model type (RD or NCA)
    # Get channel count from w2 output dimension
    w2_shape = state_dict['w2.weight'].shape  # [chn, fc_dim, 1, 1]
    chn = w2_shape[0]

    # Use model_type from training params if available, otherwise detect
    if training_params and 'model_type' in training_params:
        model_type = training_params['model_type']
        print(f"Using model_type from training params: {model_type}")
    else:
        model_type = detect_model_type(state_dict, chn)

    # Convert to numpy format
    np_params = torch_model_to_np(state_dict)

    # Override noise_level if provided via command line
    if noise_level is not None:
        np_params[-1] = noise_level

    # Extract model name from filename
    model_name = os.path.splitext(os.path.basename(pt_path))[0]

    # Export to JSON format
    metadata = {
        'model_names': [model_name],
        'pos_emb': pos_emb,
        'model_type': model_type
    }

    js_models = export_np_models_to_json(np_params, metadata)

    # Add training hyperparameters to JSON if present
    if training_params:
        # Add hyperparameters that the demo needs
        if 'dt' in training_params:
            js_models['dt'] = training_params['dt']
        if 'noise_level' in training_params:
            # Use training noise_level unless overridden by command line
            if noise_level is None:
                js_models['noise_level'] = training_params['noise_level']
        if 'diff_coef' in training_params:
            js_models['diff_coef'] = training_params['diff_coef']
        if 'reaction_rate' in training_params:
            js_models['reaction_rate'] = training_params['reaction_rate']
        if 'diffusion_rate' in training_params:
            js_models['diffusion_rate'] = training_params['diffusion_rate']
        if 'seed_type' in training_params:
            js_models['seed_type'] = training_params['seed_type']
        if 'growing_mode' in training_params:
            js_models['growing_mode'] = training_params['growing_mode']
        if 'fire_rate' in training_params:
            js_models['fire_rate'] = training_params['fire_rate']
    else:
        # For older models without _training_params, try to extract from state_dict
        # The diff_coef is stored as a registered buffer in the model
        if 'diff_coef' in state_dict:
            js_models['diff_coef'] = state_dict['diff_coef'].cpu().tolist()
            print(f"Extracted diff_coef from state_dict buffer: {js_models['diff_coef']}")

    # Add dt if specified via command line (for older models without training params)
    if dt is not None:
        js_models['dt'] = dt
        print(f"Using dt from command line: {dt}")

    # Add growing_mode if specified via command line (for Growing NCA models from nca_experiments.ipynb)
    if growing_mode:
        js_models['growing_mode'] = True
        js_models['fire_rate'] = 0.5  # Default fire_rate for Growing NCA
        print(f"Growing NCA mode enabled: normalized Sobel filters, fire_rate=0.5")

    # Determine output path
    if output_path is None:
        output_path = os.path.splitext(pt_path)[0] + '.json'

    # Save JSON
    print(f"Saving JSON to: {output_path}")
    with open(output_path, 'w') as f:
        json.dump(js_models, f, indent=2)

    print("Conversion complete!")
    print(f"  - Model name: {model_name}")
    print(f"  - Model type: {model_type}")
    print(f"  - Layers: {len(js_models['layers'])}")
    print(f"  - Noise level: {js_models['noise_level']}")
    if 'dt' in js_models:
        print(f"  - dt: {js_models['dt']}")
    if 'diff_coef' in js_models:
        print(f"  - diff_coef: {js_models['diff_coef']}")
    print(f"  - Positional embedding: {pos_emb}")

    return output_path


def main():
    parser = argparse.ArgumentParser(
        description='Convert PyTorch .pt model files to JSON format for NoiseNCA demo'
    )
    parser.add_argument('input', help='Input .pt file path')
    parser.add_argument('output', nargs='?', help='Output .json file path (optional)')
    parser.add_argument('--noise-level', type=float, help='Noise level value (if not in model)')
    parser.add_argument('--pos-emb', action='store_true',
                       help='Model uses positional embedding (default: False)')
    parser.add_argument('--dt', type=float, help='Timestep value for RD models (default from model or 0.5)')
    parser.add_argument('--growing-mode', action='store_true',
                       help='Enable Growing NCA mode (normalized Sobel filters, stochastic fire_rate=0.5)')

    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"Error: Input file not found: {args.input}")
        sys.exit(1)

    try:
        convert_pt_to_json(args.input, args.output, args.noise_level, args.pos_emb, args.dt, args.growing_mode)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
