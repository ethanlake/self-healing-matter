#!/usr/bin/env python3
"""
Visualization script for trained NCA models.
This script loads a trained model and displays real-time visualization of the dynamics using matplotlib.

Usage examples:
    # Visualize with varying dt (time step)
    python visualize.py --model_type Noise-NCA --texture bubbly_0101 --dt 0.1
    
    # Custom dimensions
    python visualize.py --model_type PE-NCA --texture grid_0002 --height 256 --width 256
"""

import argparse
import os
import sys
import yaml
import zlib
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons, Button

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


def compute_compressed_size(img):
    """
    Compute the size of a compressed representation of the image using zlib.
    This serves as a proxy for entropy.
    
    Args:
        img: Image array of shape (H, W, 3) or (H, W) with values in [0, 1]
    
    Returns:
        Compressed size in bytes (int)
    """
    # Convert image to uint8 format for compression
    if img.max() <= 1.0:
        img_uint8 = (img * 255).astype(np.uint8)
    else:
        img_uint8 = img.astype(np.uint8)
    
    # Flatten the image to a byte array
    img_bytes = img_uint8.tobytes()
    
    # Compress using zlib and return the size
    compressed = zlib.compress(img_bytes, level=9)  # level 6 is a good balance of speed/size
    return len(compressed)


def rgb_to_grayscale(img):
    """
    Convert RGB image to grayscale.
    
    Args:
        img: RGB image array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        Grayscale image array of shape (H, W)
    """
    if len(img.shape) == 3:
        return np.mean(img, axis=2)
    else:
        return img


def compute_power_spectrum(img):
    """
    Compute the power spectrum (FFT of autocorrelation) efficiently.
    The FFT of autocorrelation is |FFT(image)|^2, so we compute that directly.
    
    Args:
        img: RGB image array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        Power spectrum in log scale with zero frequency centered
    """
    # Convert to grayscale for FFT computation
    gray = rgb_to_grayscale(img)
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    
    # Power spectrum = |FFT|^2 (this is the FFT of autocorrelation)
    power = np.abs(fft) ** 2
    
    # Shift zero frequency to center
    power_shifted = np.fft.fftshift(power)
    
    # Log scale for better visualization (add small epsilon to avoid log(0))
    log_power = np.log(power_shifted + 1e-10)
    
    return log_power


def compute_angular_average_2d(data):
    """
    Compute angular average of a 2D array (radial average).
    
    Args:
        data: 2D array with zero frequency at center
    
    Returns:
        radii: Array of radial distances
        values: Angular averaged values
    """
    h, w = data.shape
    center_y, center_x = h // 2, w // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    r = r.astype(int)
    
    # Get unique radii and compute averages
    r_flat = r.flatten()
    data_flat = data.flatten()
    
    # Get unique radii
    r_unique = np.unique(r_flat)
    r_max = int(np.max(r_unique))
    
    # Compute angular average for each radius
    values = []
    radii = []
    for r_val in range(r_max + 1):
        mask = (r_flat == r_val)
        if np.any(mask):
            avg_value = np.mean(data_flat[mask])
            values.append(avg_value)
            radii.append(r_val)
    
    return np.array(radii), np.array(values)


def compute_Sq(img):
    """
    Compute S(q): angular average of the squared Fourier transform of autocorrelation.
    Since power spectrum = |FFT|^2 = FFT(autocorrelation), we can use it directly.
    
    Args:
        img: RGB image array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        q: Array of wavenumbers (radial frequencies)
        S_q: Angular averaged power spectrum
    """
    # Convert to grayscale
    gray = rgb_to_grayscale(img)
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    
    # Power spectrum = |FFT|^2
    power = np.abs(fft) ** 2
    
    # Shift zero frequency to center
    power_shifted = np.fft.fftshift(power)
    
    # Compute angular average
    q, S_q = compute_angular_average_2d(power_shifted)
    
    return q, S_q


def compute_Sr(img):
    """
    Compute S(r): angular average of the autocorrelation function.
    
    Args:
        img: RGB image array of shape (H, W, 3) with values in [0, 1]
    
    Returns:
        r: Array of radial distances
        S_r: Angular averaged autocorrelation
    """
    # Convert to grayscale
    gray = rgb_to_grayscale(img)
    
    # Compute FFT
    fft = np.fft.fft2(gray)
    
    # Power spectrum = |FFT|^2 (FFT of autocorrelation)
    power = np.abs(fft) ** 2
    
    # IFFT to get autocorrelation
    autocorr = np.fft.ifft2(power)
    autocorr = np.real(autocorr)  # Take real part
    
    # Shift zero frequency to center
    autocorr_shifted = np.fft.fftshift(autocorr)
    
    # Compute angular average
    r, S_r = compute_angular_average_2d(autocorr_shifted)
    
    return r, S_r


def visualize_varying_dt(model, device, dt=0.1, height=128, width=128, show_fft=False, show_grayscale=False, 
                         structure_fact=None, noise_strength=0.0, show_entropy=False, show_hidden=False,
                         blend_model=None, blend_strength=0.0):
    """
    Visualize NCA dynamics with varying dt (time step) using matplotlib.
    Similar to the "#@title Varying dt" block in the notebook.
    
    Args:
        show_fft: If True, display power spectrum (FFT of autocorrelation) in a separate plot
        show_grayscale: If True, display grayscale image used for FFT computation
        structure_fact: If 'Sq', show S(q) plot below FFT. If 'Sr', show S(r) plot below FFT. If None, show neither.
        noise_strength: If > 0, add noise to all channel values (RGB + hidden) after each update. Noise is drawn from [-0.5*ep, +0.5*ep] where ep is noise_strength.
        show_entropy: If True, compute and display compressed file size (entropy proxy) every 5 frames.
        show_hidden: If True, display a hidden channel as grayscale alongside RGB. A dropdown allows selecting which channel.
        blend_model: If not None, a second model to blend with the base model.
        blend_strength: Blend strength for second model. 0 = only base model, 1 = only blend model.
    """
    model = model.to(device)
    model.eval()
    
    # Get number of channels and hidden channels
    total_channels = model.chn
    num_hidden_channels = total_channels - 3  # RGB channels are first 3
    
    # Setup hidden channel visualization
    # Use a list to allow modification in nested functions
    selected_hidden_channel = [0]  # Default to first hidden channel (0-indexed)
    
    if show_hidden:
        if show_fft:
            print(f"Warning: Hidden channel visualization is only available in real-space (not with FFT). Ignoring --show_hidden.")
            show_hidden = False
        else:
            print(f"Model has {total_channels} total channels ({num_hidden_channels} hidden channels)")
            print(f"Displaying hidden channel 0 (absolute index 3) by default")
    
    # Setup matplotlib figure with subplots
    plt.ion()  # Turn on interactive mode
    
    # Determine number of plots and which ones to show
    # Hidden channel only shown in real-space (not with FFT), and shows RGB + hidden channel side-by-side
    # show_hidden is already set above (and possibly disabled if show_fft is True)
    show_rgb = not show_grayscale  # Show RGB unless grayscale is enabled (hidden channel doesn't replace RGB)
    show_S_plot = structure_fact is not None and show_fft  # Only show S(q)/S(r) if FFT is also shown
    
    # Setup subplot layout
    if show_fft and show_S_plot:
        # Use 1x3 horizontal layout: image, FFT, S(q)/S(r) side by side
        # Note: hidden channel is NOT shown with FFT
        zoomed_height = height * 2
        zoomed_width = width * 2
        aspect_ratio = zoomed_width / zoomed_height
        
        # Calculate figure dimensions to match plot aspect ratios
        # Each plot should have the same height, and width proportional to aspect ratio
        plot_height = 6  # Base height in inches
        plot_width = plot_height * aspect_ratio  # Width for each square plot
        
        # Total figure dimensions: 3 plots side by side
        fig_width = plot_width * 3 + 1  # Add some space for labels/margins
        fig_height = plot_height + 1.5  # Add space for titles
        
        fig, (ax1, ax2, ax_S) = plt.subplots(1, 3, figsize=(fig_width, fig_height))
        
        # Left plot - RGB or Grayscale (no hidden channel with FFT)
        if show_rgb:
            ax1.set_title(f'NCA Visualization (dt={dt})', fontsize=14)
        else:  # show_grayscale
            ax1.set_title(r'${\sf image}$', fontsize=14)
        ax1.axis('off')
        ax1.set_aspect('equal')
        
        # Middle plot - FFT
        ax2.set_title(r'$|S(\mathbf{q})|^2$', fontsize=14)
        ax2.axis('off')
        ax2.set_aspect('equal')
        
        # Right plot - S(q) or S(r)
        if structure_fact == 'Sq':
            ax_S.set_xlabel(r'$q$')
            ax_S.set_ylabel(r'$\widetilde{S}(q)$')
        else:  # 'Sr'
            ax_S.set_xlabel(r'$r$')
            ax_S.set_ylabel(r'$\widetilde{S}(r)$')
        ax_S.grid(True, alpha=0.3)
        # Aspect ratio will be set after first data is plotted
        
        ax3 = None
    else:
        # Layout without S(q)/S(r)
        # Special case: if hidden channel is shown, always show RGB + hidden channel side-by-side
        if show_hidden:
            # RGB + Hidden Channel side-by-side (real-space only, no FFT)
            zoomed_height = height * 2
            zoomed_width = width * 2
            aspect_ratio = zoomed_width / zoomed_height
            
            plot_height = 6  # Base height in inches
            plot_width = plot_height * aspect_ratio  # Width for each square plot
            
            # Total figure dimensions: 2 plots side by side
            fig_width = plot_width * 2 + 0.8  # Add some space for labels/margins
            fig_height = plot_height + 1.5  # Add space for titles
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
            
            # Left plot - RGB
            ax1.set_title(f'RGB (dt={dt})', fontsize=14)
            ax1.axis('off')
            ax1.set_aspect('equal')
            
            # Right plot - Hidden Channel
            ax2.set_title(f'Hidden Channel {selected_hidden_channel[0]} (dt={dt})', fontsize=14)
            ax2.axis('off')
            ax2.set_aspect('equal')
            
            ax3 = None
            ax_S = None
        else:
            # Original layout without hidden channel
            num_plots = (1 if show_rgb else 0) + (1 if show_grayscale else 0) + (1 if show_fft else 0)
            
            if num_plots == 1:
                # Single plot
                zoomed_height = height * 2
                zoomed_width = width * 2
                aspect_ratio = zoomed_width / zoomed_height
                
                # Calculate figure dimensions to match plot aspect ratio
                plot_height = 8  # Base height in inches
                plot_width = plot_height * aspect_ratio  # Width for the plot
                
                fig, ax1 = plt.subplots(figsize=(plot_width + 0.5, plot_height + 1.5))
                if show_rgb:
                    ax1.set_title(r'${\sf image}$', fontsize=14)
                elif show_grayscale:
                    ax1.set_title(r'${\sf image}$', fontsize=14)
                else:  # show_fft only (shouldn't happen, but handle it)
                    ax1.set_title(fr'$|S(\mathbf{q})|^2$', fontsize=14)
                ax1.axis('off')
                ax1.set_aspect('equal')
                ax2 = None
                ax3 = None
                ax_S = None
            elif num_plots == 2:
                # Two plots
                zoomed_height = height * 2
                zoomed_width = width * 2
                aspect_ratio = zoomed_width / zoomed_height
                
                # Calculate figure dimensions to match plot aspect ratios
                plot_height = 6  # Base height in inches
                plot_width = plot_height * aspect_ratio  # Width for each square plot
                
                # Total figure dimensions: 2 plots side by side
                fig_width = plot_width * 2 + 0.8  # Add some space for labels/margins
                fig_height = plot_height + 1.5  # Add space for titles
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(fig_width, fig_height))
                
                if show_rgb and show_fft:
                    # RGB + FFT
                    ax1.set_title(r'${\sf image}$', fontsize=14)
                    ax2.set_title(r'$S(\mathbf{q})$', fontsize=14)
                    ax3 = None
                elif show_grayscale and show_fft:
                    # Grayscale + FFT
                    ax1.set_title(r'${\sf image}$', fontsize=14)
                    ax2.set_title(r'$S(\mathbf{q})$', fontsize=14)
                    ax3 = None
                else:
                    # Shouldn't happen, but handle it
                    ax3 = None
                ax1.axis('off')
                ax1.set_aspect('equal')
                ax2.axis('off')
                ax2.set_aspect('equal')
                ax_S = None
    
    im = None
    im_gray = None
    im_hidden = None
    im_fft = None
    line_S = None
    step = 0
    update_interval = 4 #max(1, int(8 / dt))  # Update display every N steps
    
    # Use a list to hold noise_strength so it can be modified by slider
    noise_strength_var = [noise_strength]
    noise_annotation = [None]  # Use list to allow modification in nested scopes
    entropy_annotation = [None]  # Use list to allow modification in nested scopes
    entropy_frame_counter = 0  # Counter for entropy calculation (every 5 frames)
    current_compressed_size = [0]  # Store current compressed size
    
    # Adjust figure to make room for slider at the bottom
    # Apply tight_layout first if needed
    if show_fft and show_S_plot:
        plt.tight_layout(pad=1.0, rect=[0, 0.08, 1, 1])  # Leave bottom 8% for slider
    else:
        # For other layouts, adjust bottom margin
        num_plots = (1 if show_rgb else 0) + (1 if show_grayscale else 0) + (1 if show_fft else 0)
        if num_plots == 2:
            plt.tight_layout(pad=1.0, rect=[0, 0.08, 1, 1])
        elif num_plots == 1:
            plt.tight_layout(pad=1.0, rect=[0, 0.08, 1, 1])
        else:
            fig.subplots_adjust(bottom=0.15)
    
    # Create slider for noise strength (centered below all plots)
    ax_slider = plt.axes([0.25, 0.02, 0.5, 0.03])  # [left, bottom, width, height] in figure coordinates
    slider = Slider(ax_slider, 'Noise Strength', 0.0, 1.0, valinit=noise_strength, valstep=0.01)
    
    # Slider update function
    def update_noise(val):
        noise_strength_var[0] = slider.val
        # Update annotation immediately
        if noise_annotation[0] is not None:
            noise_annotation[0].set_text(f'Noise: {noise_strength_var[0]:.3f}')
            fig.canvas.draw_idle()
    
    slider.on_changed(update_noise)
    
    # Create dropdown (RadioButtons) for hidden channel selection if show_hidden is enabled
    radio_buttons = None
    if show_hidden and num_hidden_channels > 0:
        # Create labels for each hidden channel
        channel_labels = [str(i) for i in range(num_hidden_channels)]
        
        # Position the radio buttons on the right side of the figure
        ax_radio = plt.axes([0.92, 0.3, 0.07, 0.4])  # [left, bottom, width, height]
        radio_buttons = RadioButtons(ax_radio, channel_labels, active=0)
        ax_radio.set_title('Hidden\nChannel', fontsize=10)
        
        def update_hidden_channel(label):
            selected_hidden_channel[0] = int(label)
            fig.canvas.draw_idle()
        
        radio_buttons.on_clicked(update_hidden_channel)
    
    # Reset button
    ax_reset = plt.axes([0.8, 0.06, 0.1, 0.04])
    button_reset = Button(ax_reset, 'Reset')
    reset_flag = [False]
    
    def on_reset(event):
        reset_flag[0] = True
    
    button_reset.on_clicked(on_reset)
    
    try:
        with torch.no_grad():
            s = model.seed(1, height, width).to(device)
            
            print("Visualization running. Use the slider at the bottom to adjust noise strength. Close the window to exit.")
            while True:
                # Check if window is still open
                if not plt.fignum_exists(fig.number):
                    print("\nWindow closed. Exiting...")
                    break
                
                # Check for reset
                if reset_flag[0]:
                    s = model.seed(1, height, width).to(device)
                    reset_flag[0] = False
                
                # Compute update: blend two models if blend_strength > 0
                if blend_model is not None and blend_strength > 0:
                    # new = old + dt * ((1-blend_strength) * update_base + blend_strength * update_blend)
                    # model(s, dt=1) gives: old + 1 * update, so update = model(s, dt=1) - old
                    update_base = model(s, dt=1.0) - s
                    update_blend = blend_model(s, dt=1.0) - s
                    blended_update = (1 - blend_strength) * update_base + blend_strength * update_blend
                    s[:] = s + dt * blended_update
                else:
                    s[:] = model(s, dt=dt)
                
                # Add noise to all channels if noise_strength > 0
                current_noise = noise_strength_var[0]
                if current_noise > 0:
                    with torch.no_grad():
                        # Generate noise for all channels (RGB + hidden channels)
                        # Noise is drawn from uniform distribution [-0.5*ep, +0.5*ep]
                        noise = (torch.rand_like(s) - 0.5) * current_noise
                        s = s + noise
                
                step += 1
                
                # Update display periodically
                if step % update_interval == 0:
                    img = model.to_rgb(s[0]).permute(1, 2, 0).cpu().numpy()
                    img = np.clip(img, 0, 1)
                    img_zoomed = zoom(img, 2)
                    
                    # Compute entropy (compressed size) if enabled (every 10 frames)
                    # Use grayscale image for compression calculation
                    if show_entropy:
                        entropy_frame_counter += 1
                        if entropy_frame_counter % 10 == 0:
                            gray_img = rgb_to_grayscale(img_zoomed)
                            current_compressed_size[0] = compute_compressed_size(gray_img)
                    
                    # Update RGB plot if enabled (only if grayscale is not shown)
                    if show_rgb:
                        if im is None:
                            im = ax1.imshow(img_zoomed, extent=[0, img_zoomed.shape[1], 0, img_zoomed.shape[0]], 
                                           aspect='equal')
                            # Add noise strength annotation
                            current_noise = noise_strength_var[0]
                            annotation_text = f'Hidden Channels: {num_hidden_channels}'
                            if current_noise > 0:
                                annotation_text += f'\nNoise: {current_noise:.3f}'
                            if show_entropy:
                                annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                            noise_annotation[0] = ax1.annotate(annotation_text, 
                                                       xy=(0.02, 0.98), xycoords='axes fraction',
                                                       fontsize=12, color='white',
                                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                                       verticalalignment='top')
                            if show_entropy:
                                entropy_annotation[0] = noise_annotation[0]  # Reuse same annotation
                        else:
                            im.set_array(img_zoomed)
                            im.set_extent([0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                            # Update noise strength annotation
                            if noise_annotation[0] is not None:
                                current_noise = noise_strength_var[0]
                                annotation_text = f'Hidden Channels: {num_hidden_channels}'
                                if current_noise > 0:
                                    annotation_text += f'\nNoise: {current_noise:.3f}'
                                if show_entropy:
                                    annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                                noise_annotation[0].set_text(annotation_text)
                    
                    # Update grayscale plot if enabled
                    if show_grayscale:
                        gray_img = rgb_to_grayscale(img_zoomed)
                        gray_plot_ax = ax1  # Grayscale is in ax1 when it's the only/main plot
                        if im_gray is None:
                            im_gray = gray_plot_ax.imshow(gray_img, cmap='gray', aspect='equal',
                                                          extent=[0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                            # Add noise strength annotation
                            current_noise = noise_strength_var[0]
                            annotation_text = f'Hidden Channels: {num_hidden_channels}'
                            if current_noise > 0:
                                annotation_text += f'\nNoise: {current_noise:.3f}'
                            if show_entropy:
                                annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                            noise_annotation[0] = gray_plot_ax.annotate(annotation_text, 
                                                                   xy=(0.02, 0.98), xycoords='axes fraction',
                                                                   fontsize=12, color='white',
                                                                   bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                                                   verticalalignment='top')
                            if show_entropy:
                                entropy_annotation[0] = noise_annotation[0]  # Reuse same annotation
                        else:
                            im_gray.set_array(gray_img)
                            im_gray.set_extent([0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                            im_gray.set_clim(vmin=gray_img.min(), vmax=gray_img.max())
                            # Update noise strength annotation
                            if noise_annotation[0] is not None:
                                current_noise = noise_strength_var[0]
                                annotation_text = f'Hidden Channels: {num_hidden_channels}'
                                if current_noise > 0:
                                    annotation_text += f'\nNoise: {current_noise:.3f}'
                                if show_entropy:
                                    annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                                noise_annotation[0].set_text(annotation_text)
                    
                    # Update hidden channel plot if enabled (always shown with RGB side-by-side)
                    if show_hidden:
                        # Get current selected hidden channel (0-indexed from first hidden channel)
                        current_hidden_ch = selected_hidden_channel[0]
                        hidden_channel_idx = current_hidden_ch + 3  # Absolute index (RGB are 0,1,2)
                        
                        # Extract hidden channel from state tensor
                        hidden_channel_data = s[0, hidden_channel_idx, :, :].cpu().numpy()
                        # Normalize to [0, 1] range (state values are typically in [-1, 1])
                        hidden_channel_data = (hidden_channel_data + 1.0) / 2.0
                        hidden_channel_data = np.clip(hidden_channel_data, 0, 1)
                        # Zoom the hidden channel
                        hidden_channel_zoomed = zoom(hidden_channel_data, 2)
                        
                        # Hidden channel is always in ax2 when shown (ax1 is RGB)
                        hidden_plot_ax = ax2
                        if im_hidden is None:
                            im_hidden = hidden_plot_ax.imshow(hidden_channel_zoomed, cmap='gray', aspect='equal',
                                                              extent=[0, hidden_channel_zoomed.shape[1], 0, hidden_channel_zoomed.shape[0]])
                            # Add annotation with number of hidden channels
                            current_noise = noise_strength_var[0]
                            annotation_text = f'Hidden Channels: {num_hidden_channels}\nChannel: {current_hidden_ch}'
                            if current_noise > 0:
                                annotation_text += f'\nNoise: {current_noise:.3f}'
                            if show_entropy:
                                annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                            # Create separate annotation for hidden channel plot
                            hidden_annotation = hidden_plot_ax.annotate(annotation_text, 
                                                                       xy=(0.02, 0.98), xycoords='axes fraction',
                                                                       fontsize=12, color='white',
                                                                       bbox=dict(boxstyle='round', facecolor='black', alpha=0.7),
                                                                       verticalalignment='top')
                        else:
                            im_hidden.set_array(hidden_channel_zoomed)
                            im_hidden.set_extent([0, hidden_channel_zoomed.shape[1], 0, hidden_channel_zoomed.shape[0]])
                            im_hidden.set_clim(vmin=hidden_channel_zoomed.min(), vmax=hidden_channel_zoomed.max())
                            # Update annotation with current channel
                            if hidden_annotation is not None:
                                current_noise = noise_strength_var[0]
                                annotation_text = f'Hidden Channels: {num_hidden_channels}\nChannel: {current_hidden_ch}'
                                if current_noise > 0:
                                    annotation_text += f'\nNoise: {current_noise:.3f}'
                                if show_entropy:
                                    annotation_text += f'\nEntropy: {current_compressed_size[0]} bytes'
                                hidden_annotation.set_text(annotation_text)
                    
                    # Update FFT plot if enabled
                    if show_fft:
                        power_spectrum = compute_power_spectrum(img_zoomed)
                        # FFT plot position: ax2 if RGB or grayscale is shown, ax1 if FFT only
                        fft_plot_ax = ax2 if (show_rgb or show_grayscale) else ax1
                        if im_fft is None:
                            im_fft = fft_plot_ax.imshow(power_spectrum, cmap='viridis', aspect='equal', 
                                                extent=[0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                            # cbar = fig.colorbar(im_fft,fraction=0.046, pad=0.04, ax = fft_plot_ax) #needed to get the spacing to the colorbar right --- the last ax allows one to place a separate cbar on each axis 
                            # cbar.minorticks_on()
                            # cbar.set_label('')
                        else:
                            im_fft.set_array(power_spectrum)
                            im_fft.set_clim(vmin=power_spectrum.min(), vmax=power_spectrum.max())
                            im_fft.set_extent([0, img_zoomed.shape[1], 0, img_zoomed.shape[0]])
                    
                    # Update S(q) or S(r) plot if enabled
                    if show_S_plot and ax_S is not None:
                        zoomed_height = img_zoomed.shape[0]
                        zoomed_width = img_zoomed.shape[1]
                        aspect_ratio = zoomed_width / zoomed_height
                        
                        if structure_fact == 'Sq':
                            q, S_q = compute_Sq(img_zoomed)
                            q = q[1:]; S_q = (S_q[1:]) 
                            S_q /= S_q.max()
                            if line_S is None:
                                line_S, = ax_S.plot(q, S_q, 'b-', linewidth=1.5)
                                ax_S.set_xlim(0, q.max())
                                ax_S.set_ylim(S_q.min() * 0.9, S_q.max() * 1.1)
                            else:
                                line_S.set_data(q, S_q)
                                ax_S.set_xlim(0, q.max())
                                ax_S.set_ylim(S_q.min() * 0.9, S_q.max() * 1.1)
                        else:  # 'Sr'
                            r, S_r = compute_Sr(img_zoomed)
                            S_r /= S_r.max()
                            # Limit to half the image width
                            max_r = zoomed_width / 2
                            mask = r <= max_r
                            r_filtered = r[mask]
                            S_r_filtered = S_r[mask]
                            
                            if line_S is None:
                                line_S, = ax_S.plot(r_filtered, S_r_filtered, 'r-', linewidth=1.5)
                                ax_S.set_xlim(0, max_r)
                                ax_S.set_ylim(S_r_filtered.min() * 0.9, S_r_filtered.max() * 1.1)
                            else:
                                line_S.set_data(r_filtered, S_r_filtered)
                                ax_S.set_xlim(0, max_r)
                                ax_S.set_ylim(S_r_filtered.min() * 0.9, S_r_filtered.max() * 1.1)
                        
                        # Maintain aspect ratio to match image plots
                        ax_S.set_aspect(aspect_ratio / ax_S.get_data_ratio())
                    
                    plt.draw()
                    fig.canvas.flush_events()  # Process any pending events (including keyboard)
                    plt.pause(0.01)  # Small pause to allow GUI to update
        
        plt.ioff()  # Turn off interactive mode
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
        plt.close('all')  # Close all matplotlib figures
        raise  # Re-raise to exit the program


def main():
    parser = argparse.ArgumentParser(description='Visualize trained NCA models')
    parser.add_argument('--model_type', type=str, required=True,
                        choices=['Noise-NCA', 'PE-NCA', 'Vanilla-NCA'],
                        help='Type of model (Noise-NCA, PE-NCA, or Vanilla-NCA)')
    parser.add_argument('--texture', type=str, required=True,
                        help='Texture name (e.g., bubbly_0101, flames, etc.)')
    parser.add_argument('--dt', type=float, default=0.1,
                        help='Time step (default: 0.1)')
    parser.add_argument('--height', type=int, default=128,
                        help='Height of the visualization (default: 128)')
    parser.add_argument('--width', type=int, default=128,
                        help='Width of the visualization (default: 128)')
    parser.add_argument('--show_fft', action='store_true',
                        help='Show power spectrum (FFT of autocorrelation) in a separate plot')
    parser.add_argument('--show_grayscale', action='store_true',
                        help='Show grayscale image used for FFT computation in a separate plot')
    parser.add_argument('--structure_fact', type=str, choices=['Sq', 'Sr'], default=None,
                        help='Show S(q) or S(r) plot below FFT. Requires --show_fft. S(q) is angular average of power spectrum, S(r) is angular average of autocorrelation.')
    parser.add_argument('--noise_strength', type=float, default=0.0,
                        help='Noise strength (ep) for all channels. After each update, all channel values (RGB + hidden) are changed by random numbers in [-0.5*ep, +0.5*ep]. Default: 0.0 (no noise)')
    parser.add_argument('--show_entropy', action='store_true',
                        help='Compute and display compressed file size (entropy proxy) every 5 frames. Uses zlib compression.')
    parser.add_argument('--show_hidden', action='store_true',
                        help='Display a hidden channel as grayscale alongside RGB. Use dropdown to select which hidden channel.')
    parser.add_argument('--blend_texture', type=str, default='bubbly_0101',
                        help='Second texture to blend with (default: bubbly_0101)')
    parser.add_argument('--blend_strength', type=float, default=0.0,
                        help='Blend strength for second texture. 0 = only base texture, 1 = only blend texture. Default: 0.0')
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
    
    # Load blend model if blend_strength > 0
    blend_model = None
    if args.blend_strength > 0:
        blend_model_path = os.path.join('trained_models', args.model_type, args.blend_texture, 'weights.pt')
        if not os.path.exists(blend_model_path):
            raise FileNotFoundError(f"Blend model weights not found: {blend_model_path}")
        blend_model = get_nca_model(config, args.blend_texture, device)
        print(f"Loading blend model from: {blend_model_path}")
        blend_state_dict = torch.load(blend_model_path, map_location=device)
        blend_model.load_state_dict(blend_state_dict)
        print(f"Blending {args.texture} with {args.blend_texture} (strength={args.blend_strength})")
    
    # Validate arguments
    if args.structure_fact is not None and not args.show_fft:
        raise ValueError("--structure_fact requires --show_fft to be enabled")
    
    # Run visualization
    print(f"Running visualization...")
    try:
        visualize_varying_dt(model, device, args.dt, args.height, args.width, args.show_fft, args.show_grayscale, args.structure_fact, args.noise_strength, args.show_entropy, args.show_hidden, blend_model, args.blend_strength)
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

