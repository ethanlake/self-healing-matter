#!/usr/bin/env python3
"""
Calibration script for NCA models with directional advection.
This script loads a trained model and displays real-time visualization with an additional
advection term controlled by sliders for angle (theta) and velocity (v).

The update rule is:
    s -> s + dt * (original model update) + dt * (grad_theta s * dx) * v

where grad_theta is the Sobel filter rotated by angle theta.

Usage:
    python calibrate.py --model_type Noise-NCA --texture bubbly_0101 --dt 0.1
"""

import argparse
import os
import sys
import yaml
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# Fix for matplotlib compatibility issue
try:
    import matplotlib.cbook
    if not hasattr(matplotlib.cbook, "_Stack"):
        class _Stack(list):
            def push(self, item):
                self.append(item)
                return item
            def pop(self):
                return super().pop() if self else None
            def current(self):
                return self[-1] if self else None
        matplotlib.cbook._Stack = _Stack
except:
    pass

from models import NCA, NoiseNCA, PENCA


def get_nca_model(config, texture_name, device):
    """Create an NCA model instance based on config and texture name."""
    model_type = config['model']['type']
    attr = config['model']['attr'].copy()
    attr['device'] = device
    
    if model_type == 'NCA':
        return NCA(**attr)
    elif model_type == 'NoiseNCA':
        noise_levels = config['model']['noise_levels']
        if texture_name in noise_levels:
            noise_level = noise_levels[texture_name]
        else:
            noise_level = noise_levels['default']
        return NoiseNCA(noise_level=1.0, **attr)
    elif model_type == 'PENCA':
        return PENCA(**attr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def zoom(img, scale=4):
    """Zoom an image by repeating pixels."""
    img = np.repeat(img, scale, 0)
    img = np.repeat(img, scale, 1)
    return img


def create_rotated_sobel_kernel(theta, device):
    """
    Create a Sobel-like gradient kernel rotated by angle theta.
    
    The standard Sobel x-kernel detects horizontal gradients (gradient in x direction).
    Rotating by theta gives a gradient in the direction of angle theta.
    
    Args:
        theta: Rotation angle in radians
        device: PyTorch device
    
    Returns:
        Rotated 3x3 Sobel kernel as a tensor
    """
    # Standard Sobel kernels for x and y gradients
    sobel_x = torch.tensor([
        [-1, 0, 1],
        [-2, 0, 2],
        [-1, 0, 1]
    ], dtype=torch.float32, device=device)
    
    sobel_y = torch.tensor([
        [-1, -2, -1],
        [0, 0, 0],
        [1, 2, 1]
    ], dtype=torch.float32, device=device)
    
    # Rotated gradient kernel: grad_theta = cos(theta) * sobel_x + sin(theta) * sobel_y
    # This gives the gradient in the direction of angle theta
    rotated_kernel = np.cos(theta) * sobel_x + np.sin(theta) * sobel_y
    
    return rotated_kernel


def apply_rotated_gradient(s, theta, device):
    """
    Apply rotated Sobel gradient to all channels of the state tensor.
    
    Args:
        s: State tensor of shape [b, chn, h, w]
        theta: Rotation angle in radians
        device: PyTorch device
    
    Returns:
        Gradient tensor of same shape as s
    """
    b, chn, h, w = s.shape
    
    # Create rotated kernel
    kernel = create_rotated_sobel_kernel(theta, device)
    
    # Reshape kernel for conv2d: [out_channels, in_channels, kH, kW]
    # We apply the same kernel to each channel independently
    kernel = kernel.view(1, 1, 3, 3)
    
    # Apply convolution to each channel
    # Use padding='same' equivalent with circular padding for seamless textures
    s_padded = F.pad(s, (1, 1, 1, 1), mode='circular')
    
    # Apply kernel to each channel separately
    grad = torch.zeros_like(s)
    for c in range(chn):
        grad[:, c:c+1, :, :] = F.conv2d(s_padded[:, c:c+1, :, :], kernel)
    
    return grad


def visualize_with_advection(model, device, dt=0.1, dx=1.0, height=128, width=128):
    """
    Visualize NCA dynamics with additional advection term controlled by sliders.
    
    The update is:
        s -> s + dt * (model update) + dt * (grad_theta s * dx) * v
    
    Args:
        model: NCA model
        device: PyTorch device
        dt: Time step
        dx: Spatial step (used in advection term)
        height, width: Grid dimensions
    """
    model = model.to(device)
    model.eval()
    
    # Setup matplotlib figure
    plt.ion()
    
    zoomed_height = height * 2
    zoomed_width = width * 2
    aspect_ratio = zoomed_width / zoomed_height
    
    plot_height = 6
    plot_width = plot_height * aspect_ratio
    fig_width = plot_width + 0.5
    fig_height = plot_height + 2.0  # Extra space for sliders
    
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))
    
    # Set window title
    try:
        fig.canvas.manager.set_window_title('NCA Calibration with Advection')
    except:
        pass
    
    ax.set_title(f'NCA Visualization (dt={dt})', fontsize=14)
    ax.axis('off')
    ax.set_aspect('equal')
    
    # Adjust layout to make room for sliders
    fig.subplots_adjust(bottom=0.25)
    
    # Create sliders for theta and v
    # Theta slider: angle in degrees (converted to radians internally)
    ax_theta = plt.axes([0.15, 0.12, 0.7, 0.03])
    slider_theta = Slider(ax_theta, 'θ (degrees)', -180, 180, valinit=0, valstep=1)
    
    # V slider: velocity
    ax_v = plt.axes([0.15, 0.06, 0.7, 0.03])
    slider_v = Slider(ax_v, 'v (velocity)', -2.0, 2.0, valinit=0.0, valstep=0.01)
    
    # Use lists to allow modification in nested functions
    theta_var = [0.0]  # In radians
    v_var = [0.0]
    
    def update_theta(val):
        theta_var[0] = np.deg2rad(slider_theta.val)
        fig.canvas.draw_idle()
    
    def update_v(val):
        v_var[0] = slider_v.val
        fig.canvas.draw_idle()
    
    slider_theta.on_changed(update_theta)
    slider_v.on_changed(update_v)
    
    # Reset button
    ax_reset = plt.axes([0.8, 0.17, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    reset_flag = [False]
    
    def on_reset(event):
        reset_flag[0] = True
    
    button_reset.on_clicked(on_reset)
    
    # Annotation for current values
    annotation = [None]
    
    try:
        with torch.no_grad():
            s = model.seed(1, height, width).to(device)
            
            im = None
            step = 0
            update_interval = 1
            
            print("Visualization running. Use sliders to adjust theta (angle) and v (velocity). Close window to exit.")
            while True:
                if not plt.fignum_exists(fig.number):
                    print("\nWindow closed. Exiting...")
                    break
                
                # Check for reset
                if reset_flag[0]:
                    s = model.seed(1, height, width).to(device)
                    reset_flag[0] = False
                
                # Get current slider values
                theta = theta_var[0]
                v = v_var[0]
                
                # Standard model update
                s[:] = model(s, dt=dt)
                
                step += 1
                
                # Update display
                if step % update_interval == 0:
                    # Display shifted state: s + v * grad_theta(s) * dx
                    if abs(v) > 1e-6:
                        grad_s = apply_rotated_gradient(s, theta, device)
                        s_display = s + v * grad_s * dx
                    else:
                        s_display = s
                    
                    img = model.to_rgb(s_display[0]).permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    img_zoomed = zoom(img, 2)
                    
                    if im is None:
                        im = ax.imshow(img_zoomed, extent=[0, img_zoomed.shape[1], 0, img_zoomed.shape[0]], aspect='equal')
                        # Add annotation
                        annotation_text = f'θ={np.rad2deg(theta):.0f}°, v={v:.2f}'
                        annotation[0] = ax.annotate(annotation_text,
                                                    xy=(0.02, 0.98), xycoords='axes fraction',
                                                    fontsize=12, color='white',
                                                    bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                                    verticalalignment='top')
                    else:
                        im.set_array(img_zoomed)
                        im.set_extent([0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                        # Update annotation
                        annotation_text = f'θ={np.rad2deg(theta):.0f}°, v={v:.2f}'
                        annotation[0].set_text(annotation_text)
                    
                    plt.draw()
                    fig.canvas.flush_events()
                    plt.pause(0.01)
        
        plt.ioff()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        plt.close('all')
        raise


def main():
    parser = argparse.ArgumentParser(description='NCA Calibration with Advection')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Noise-NCA', 'PE-NCA', 'Vanilla-NCA'],
                        help='Type of model (Noise-NCA, PE-NCA, or Vanilla-NCA)')
    parser.add_argument('--texture', type=str, required=True,
                        help='Texture name (e.g., bubbly_0101, flames, etc.)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (default: 0.1)')
    parser.add_argument('--dx', type=float, default=1.0,
                        help='Spatial step for advection term (default: 1.0)')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of the visualization (default: 128)')
    parser.add_argument('--width', type=int, default=128,
                        help='Width of the visualization (default: 128)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use (default: cuda if available, else cpu)')
    
    args = parser.parse_args()
    
    # Map model type to config file
    config_map = {
        'Noise-NCA': 'configs/Noise-NCA.yml',
        'PE-NCA': 'configs/PE-NCA.yml',
        'Vanilla-NCA': 'configs/Vanilla-NCA.yml'
    }
    
    config_path = config_map[args.model_type]
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")
    
    # Load config
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Construct model path
    model_path = os.path.join('trained_models', args.model_type, args.texture, 'weights.pt')
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found: {model_path}")
    
    # Create model
    device = torch.device(args.device)
    model = get_nca_model(config, args.texture, device)
    
    # Load weights
    print(f"Loading model from: {model_path}")
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    
    # Run visualization
    print(f"Running calibration visualization...")
    print(f"Update rule: s -> s + dt*(model update) + dt*(grad_theta(s)*dx)*v")
    try:
        visualize_with_advection(model, device, args.dt, args.dx, args.height, args.width)
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
        plt.close('all')
        sys.exit(0)


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nProgram interrupted. Exiting...")
        plt.close('all')
        sys.exit(0)

