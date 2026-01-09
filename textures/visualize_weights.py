#!/usr/bin/env python3
"""
Visualize weight distributions in NCA models.

Usage:
    python visualize_weights.py <weights_file.pt>
    python visualize_weights.py trained_models/Vanilla-NCA/bubbly_0117/weights.pt

This script loads a PyTorch .pt file and plots histograms of:
- W1 weights (first dense layer)
- W2 weights (second dense layer)
- Biases (from first layer)

Requirements:
    pip install torch matplotlib numpy

Or if you have a virtual environment:
    source venv/bin/activate
    pip install -r requirements.txt
"""

import sys
import os

try:
    import torch
except ImportError:
    print("ERROR: PyTorch is not installed.")
    print("\nPlease install PyTorch:")
    print("  pip install torch")
    print("\nOr install all requirements:")
    print("  pip install -r requirements.txt")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt
except ImportError:
    print("ERROR: matplotlib is not installed.")
    print("\nPlease install matplotlib:")
    print("  pip install matplotlib")
    sys.exit(1)

try:
    import numpy as np
except ImportError:
    print("ERROR: numpy is not installed.")
    print("\nPlease install numpy:")
    print("  pip install numpy")
    sys.exit(1)


def load_weights(pt_file):
    """Load weights from .pt file."""
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Weight file not found: {pt_file}")

    state_dict = torch.load(pt_file, map_location='cpu')
    return state_dict


def extract_weight_components(state_dict):
    """Extract W1, W2, and biases from state_dict."""
    # W1: [fc_dim, in_channels, 1, 1] -> squeeze to [fc_dim, in_channels]
    w1 = state_dict['w1.weight'][:, :, 0, 0].numpy()

    # Bias: [fc_dim]
    b1 = state_dict['w1.bias'].numpy()

    # W2: [out_channels, fc_dim, 1, 1] -> squeeze to [out_channels, fc_dim]
    w2 = state_dict['w2.weight'][:, :, 0, 0].numpy()

    return w1, b1, w2


def plot_weight_distributions(w1, b1, w2, pt_file):
    """Plot histograms of weight magnitudes."""
    # Flatten weights
    w1_flat = w1.flatten()
    w2_flat = w2.flatten()
    b1_flat = b1.flatten()

    # Compute absolute values for weights (but not biases)
    w1_abs = np.abs(w1_flat)
    w2_abs = np.abs(w2_flat)

    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f'Weight Distributions: {os.path.basename(pt_file)}', fontsize=14, fontweight='bold')

    # Plot W1 weights (absolute values)
    axes[0].hist(w1_abs, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel('|Weight Value|')
    axes[0].set_ylabel('Frequency')
    axes[0].set_title(f'W1 Weights (Layer 0)\n{len(w1_flat)} weights')
    axes[0].axvline(np.mean(w1_abs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(w1_abs):.4f}')
    axes[0].axvline(np.median(w1_abs), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(w1_abs):.4f}')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # Plot W2 weights (absolute values)
    axes[1].hist(w2_abs, bins=50, color='forestgreen', alpha=0.7, edgecolor='black')
    axes[1].set_xlabel('|Weight Value|')
    axes[1].set_ylabel('Frequency')
    axes[1].set_title(f'W2 Weights (Layer 1)\n{len(w2_flat)} weights')
    axes[1].axvline(np.mean(w2_abs), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(w2_abs):.4f}')
    axes[1].axvline(np.median(w2_abs), color='orange', linestyle='--', linewidth=2, label=f'Median: {np.median(w2_abs):.4f}')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    # Plot biases (actual values, not absolute)
    axes[2].hist(b1_flat, bins=30, color='darkorange', alpha=0.7, edgecolor='black')
    axes[2].set_xlabel('Bias Value')
    axes[2].set_ylabel('Frequency')
    axes[2].set_title(f'Biases (Layer 0)\n{len(b1_flat)} biases')
    axes[2].axvline(np.mean(b1_flat), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(b1_flat):.4f}')
    axes[2].axvline(np.median(b1_flat), color='darkred', linestyle='--', linewidth=2, label=f'Median: {np.median(b1_flat):.4f}')
    axes[2].legend()
    axes[2].grid(alpha=0.3)

    axes[0].set_yscale('log')
    axes[1].set_yscale('log')

    plt.tight_layout()

    # Print statistics
    print("\n" + "="*60)
    print(f"Weight Statistics for: {os.path.basename(pt_file)}")
    print("="*60)

    print("\nW1 (Layer 0) Weights:")
    print(f"  Count:    {len(w1_flat)}")
    print(f"  Shape:    {w1.shape}")
    print(f"  Mean(|w|): {np.mean(w1_abs):.6f}")
    print(f"  Std(|w|):  {np.std(w1_abs):.6f}")
    print(f"  Min(|w|):  {np.min(w1_abs):.6f}")
    print(f"  Max(|w|):  {np.max(w1_abs):.6f}")
    print(f"  Median(|w|): {np.median(w1_abs):.6f}")

    # Count near-zero weights
    near_zero_threshold = 0.01
    near_zero_w1 = np.sum(w1_abs < near_zero_threshold)
    print(f"  Near-zero (|w|<{near_zero_threshold}): {near_zero_w1} ({100*near_zero_w1/len(w1_flat):.2f}%)")

    print("\nW2 (Layer 1) Weights:")
    print(f"  Count:    {len(w2_flat)}")
    print(f"  Shape:    {w2.shape}")
    print(f"  Mean(|w|): {np.mean(w2_abs):.6f}")
    print(f"  Std(|w|):  {np.std(w2_abs):.6f}")
    print(f"  Min(|w|):  {np.min(w2_abs):.6f}")
    print(f"  Max(|w|):  {np.max(w2_abs):.6f}")
    print(f"  Median(|w|): {np.median(w2_abs):.6f}")

    near_zero_w2 = np.sum(w2_abs < near_zero_threshold)
    print(f"  Near-zero (|w|<{near_zero_threshold}): {near_zero_w2} ({100*near_zero_w2/len(w2_flat):.2f}%)")

    print("\nBiases (Layer 0):")
    print(f"  Count:    {len(b1_flat)}")
    print(f"  Shape:    {b1.shape}")
    print(f"  Mean:     {np.mean(b1_flat):.6f}")
    print(f"  Std:      {np.std(b1_flat):.6f}")
    print(f"  Min:      {np.min(b1_flat):.6f}")
    print(f"  Max:      {np.max(b1_flat):.6f}")
    print(f"  Median:   {np.median(b1_flat):.6f}")

    print("\nTotal prunable weights: {} (W1 + W2)".format(len(w1_flat) + len(w2_flat)))
    print("Biases (excluded from pruning): {}".format(len(b1_flat)))
    print("="*60 + "\n")

    plt.show()


def main():
    if len(sys.argv) < 2:
        print("Usage: python visualize_weights.py <weights_file.pt>")
        print("\nExample:")
        print("  python visualize_weights.py trained_models/Noise-NCA/bubbly_0101.pt")
        sys.exit(1)

    pt_file = sys.argv[1]

    print(f"Loading weights from: {pt_file}")
    state_dict = load_weights(pt_file)

    print("Extracting weight components...")
    w1, b1, w2 = extract_weight_components(state_dict)

    print("Plotting distributions...")
    plot_weight_distributions(w1, b1, w2, pt_file)


if __name__ == '__main__':
    main()
