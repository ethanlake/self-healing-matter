#!/usr/bin/env python3
"""
Lyapunov exponent analysis for NCA models.
This script measures the sensitivity of NCA evolutions to changes in initial conditions
by tracking the divergence between perturbed and unperturbed trajectories.
"""

import argparse
import os
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
from tqdm import tqdm

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
        return NoiseNCA(noise_level=noise_level, **attr)
    elif model_type == 'PENCA':
        return PENCA(**attr)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def apply_circular_noise(s, center_x, center_y, radius, noise_strength, device):
    """
    Apply noise to pixels within a circle of specified radius.
    
    Args:
        s: State tensor of shape [b, chn, h, w]
        center_x, center_y: Center coordinates of the circle
        radius: Radius of the circle
        noise_strength: Strength of noise to apply
        device: PyTorch device
    
    Returns:
        Modified state tensor
    """
    b, chn, h, w = s.shape
    s_perturbed = s.clone()
    
    # Create coordinate grids
    y_coords, x_coords = torch.meshgrid(
        torch.arange(h, device=device, dtype=torch.float32),
        torch.arange(w, device=device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Compute distance from center
    distances = torch.sqrt((x_coords - center_x)**2 + (y_coords - center_y)**2)
    
    # Create mask for pixels within circle
    mask = distances <= radius
    mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, h, w]
    mask = mask.expand(b, chn, -1, -1)  # [b, chn, h, w]
    
    # Generate noise
    noise = (torch.rand_like(s) - 0.5) * noise_strength
    
    # Apply noise only within the circle
    s_perturbed = torch.where(mask, s + noise, s)
    
    return s_perturbed


def compute_pixel_distance(s1, s2):
    """
    Compute pixel-wise distance between two states using only RGB channels.
    
    Args:
        s1, s2: State tensors of shape [b, chn, h, w]
    
    Returns:
        Average pixel-wise distance (scalar) computed using only the first 3 channels (RGB)
    """
    # Compute L2 distance per pixel using only RGB channels (first 3 channels)
    diff = s1[:, :3, :, :] - s2[:, :3, :, :]  # Only use RGB channels
    pixel_distances = torch.norm(diff, dim=1)  # [b, h, w] - L2 norm over RGB channels
    return pixel_distances.mean().item()


def run_lyapunov_analysis(model, device, height, width, dx, dy, dt, 
                          tquench, tevolve, radius, noise_strength, n_runs, show_realtime=False, epsilon=0.0):
    """
    Run Lyapunov exponent analysis.
    
    Args:
        model: NCA model
        device: PyTorch device
        height, width: Grid dimensions
        dx, dy: Spatial scaling parameters
        dt: Time step
        tquench: Time to evolve before perturbation
        tevolve: Time to evolve after perturbation
        radius: Radius of circular perturbation
        noise_strength: Strength of noise in perturbation
        n_runs: Number of independent runs to average over
        show_realtime: If True, display real-time visualization of original and perturbed evolutions
        epsilon: Per-step noise strength. Same noise is applied to both copies. Default: 0.0 (no noise)
    
    Returns:
        times: Array of time steps after perturbation
        distances: Array of averaged pixel-wise distances
    """
    model = model.to(device)
    model.eval()
    
    # Center of perturbation (middle of grid)
    center_x = width / 2.0
    center_y = height / 2.0
    
    # Number of time steps
    steps_quench = int(tquench / dt)
    steps_evolve = int(tevolve / dt)
    
    # Set up real-time visualization if requested
    fig_realtime = None
    circle_patch = None
    frame_skip = 4  # Update visualization every N frames to speed up
    if show_realtime:
        plt.ion()  # Turn on interactive mode
        fig_realtime, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        ax1.set_title('Unperturbed Evolution', fontsize=14)
        ax1.axis('off')
        ax2.set_title('Perturbed Evolution', fontsize=14)
        ax2.axis('off')
        im1 = None
        im2 = None
        print("Real-time visualization enabled. Showing first run only (after perturbation).")
    
    # Storage for distances
    all_distances = []
    
    print(f"Running {n_runs} independent runs...")
    for run in tqdm(range(n_runs), desc="Runs"):
        with torch.no_grad():
            # Step 1: Evolve from random initial state for tquench (no visualization)
            s = model.seed(1, height, width).to(device)
            for _ in range(steps_quench):
                s = model(s, dx=dx, dy=dy, dt=dt)
                # Apply per-step noise if epsilon > 0
                if epsilon > 0:
                    noise = (torch.rand_like(s) - 0.5) * epsilon
                    s = s + noise
            
            # Step 2: Save state at tquench as s0
            s0 = s.clone()
            
            # Step 3: Create two copies
            s_unperturbed = s0.clone()
            s_perturbed = s0.clone()
            
            # Apply circular noise perturbation
            s_perturbed = apply_circular_noise(
                s_perturbed, center_x, center_y, radius, noise_strength, device
            )
            
            # Draw red circle at perturbation location (only for first run with visualization)
            if show_realtime and run == 0:
                # Check if window is still open
                if not plt.fignum_exists(fig_realtime.number):
                    print("\nVisualization window closed. Exiting...")
                    return np.array([]), np.array([])  # Return empty arrays to signal early exit
                
                # Show the state right after perturbation (at t - t_pert = 0)
                img_unperturbed = model.to_rgb(s_unperturbed[0]).permute(1, 2, 0).cpu().numpy()
                img_perturbed = model.to_rgb(s_perturbed[0]).permute(1, 2, 0).cpu().numpy()
                img_unperturbed = np.clip(img_unperturbed, 0, 1)
                img_perturbed = np.clip(img_perturbed, 0, 1)
                
                # Initialize images if not already created
                if im1 is None:
                    im1 = ax1.imshow(img_unperturbed, aspect='equal')
                    im2 = ax2.imshow(img_perturbed, aspect='equal')
                else:
                    im1.set_array(img_unperturbed)
                    im2.set_array(img_perturbed)
                
                # Add red circle to show perturbation location
                if circle_patch is None:
                    circle_patch = Circle((center_x, center_y), radius, fill=False, 
                                         edgecolor='red', linewidth=2)
                    ax2.add_patch(circle_patch)
                else:
                    # Update circle position if needed (shouldn't change, but just in case)
                    circle_patch.center = (center_x, center_y)
                    circle_patch.radius = radius
                
                ax1.set_title(f'Unperturbed Evolution (t - t_pert = 0.0)', fontsize=14)
                ax2.set_title(f'Perturbed Evolution (t - t_pert = 0.0, perturbation applied)', fontsize=14)
                
                plt.draw()
                plt.pause(0.01)  # Small pause to allow GUI to update
            
            # Step 4: Evolve both for tevolve time and track distances
            run_distances = []
            for step in range(steps_evolve):
                # Check if window is still open (only for first run with visualization)
                if show_realtime and run == 0:
                    if not plt.fignum_exists(fig_realtime.number):
                        print("\nVisualization window closed. Exiting...")
                        return np.array([]), np.array([])  # Return empty arrays to signal early exit
                
                # Evolve both states
                s_unperturbed = model(s_unperturbed, dx=dx, dy=dy, dt=dt)
                s_perturbed = model(s_perturbed, dx=dx, dy=dy, dt=dt)
                
                # Apply per-step noise if epsilon > 0
                # IMPORTANT: Same noise is applied to both copies
                if epsilon > 0:
                    noise = (torch.rand_like(s_unperturbed) - 0.5) * epsilon
                    s_unperturbed = s_unperturbed + noise
                    s_perturbed = s_perturbed + noise
                
                # Compute pixel-wise distance
                dist = compute_pixel_distance(s_unperturbed, s_perturbed)
                run_distances.append(dist)
                
                # Update real-time visualization (only for first run, and only every N frames)
                if show_realtime and run == 0 and step % frame_skip == 0:
                    # Convert to RGB images
                    img_unperturbed = model.to_rgb(s_unperturbed[0]).permute(1, 2, 0).cpu().numpy()
                    img_perturbed = model.to_rgb(s_perturbed[0]).permute(1, 2, 0).cpu().numpy()
                    img_unperturbed = np.clip(img_unperturbed, 0, 1)
                    img_perturbed = np.clip(img_perturbed, 0, 1)
                    
                    # Update images
                    im1.set_array(img_unperturbed)
                    im2.set_array(img_perturbed)
                    
                    # Red circle remains visible throughout evolution
                    # (it's already added above)
                    
                    # Time relative to perturbation (positive after perturbation)
                    t_rel = (step + 1) * dt
                    ax1.set_title(f'Unperturbed Evolution (t - t_pert = {t_rel:.1f})', fontsize=14)
                    ax2.set_title(f'Perturbed Evolution (t - t_pert = {t_rel:.1f}, dist={dist:.4f})', fontsize=14)
                    
                    plt.draw()
                    plt.pause(0.01)  # Small pause to allow GUI to update
            
            all_distances.append(run_distances)
    
    # Step 5: Average distances over all runs
    if len(all_distances) == 0:
        raise ValueError("No distance data collected! Check that steps_evolve > 0.")
    
    all_distances = np.array(all_distances)  # [n_runs, steps_evolve]
    
    if all_distances.size == 0:
        raise ValueError(f"No distance data! steps_evolve={steps_evolve}, n_runs={n_runs}")
    
    avg_distances = np.mean(all_distances, axis=0)
    
    # Create time array
    times = np.arange(steps_evolve) * dt
    
    # Validate data
    if len(times) != len(avg_distances):
        raise ValueError(f"Time and distance arrays have different lengths: {len(times)} vs {len(avg_distances)}")
    
    # Close real-time visualization if it was open
    if show_realtime and fig_realtime is not None:
        plt.close(fig_realtime)
        plt.ioff()  # Turn off interactive mode
    
    return times, avg_distances


def main():
    parser = argparse.ArgumentParser(description='Lyapunov exponent analysis for NCA models')
    parser.add_argument('--model_type', type=str, default='Noise-NCA',
                        choices=['Noise-NCA', 'PE-NCA', 'Vanilla-NCA'],
                        help='Type of model (Noise-NCA, PE-NCA, or Vanilla-NCA)')
    parser.add_argument('--texture', type=str, default='spiralled_0124',
                        help='Texture name (e.g., spiralled_0124, bubbly_0101, etc.)')
    parser.add_argument('--height', type=int, default=128,
                        help='Grid height (default: 128)')
    parser.add_argument('--width', type=int, default=128,
                        help='Grid width (default: 128)')
    parser.add_argument('--dx', type=float, default=.5,
                        help='X spatial scaling (default: .5)')
    parser.add_argument('--dy', type=float, default=.5,
                        help='Y spatial scaling (default: .5)')
    parser.add_argument('--dt', type=float, default=0.5,
                        help='Time step (default: 0.5)')
    parser.add_argument('--tquench', type=float, default=50.0,
                        help='Time to evolve before perturbation (default: 100.0)')
    parser.add_argument('--tevolve', type=float, default=200.0,
                        help='Time to evolve after perturbation (default: 200.0)')
    parser.add_argument('--radius', type=float, default=10.0,
                        help='Radius of circular perturbation (default: 5.0)')
    parser.add_argument('--noise_strength', type=float, default=1.,
                        help='Strength of noise in perturbation (default: 1)')
    parser.add_argument('--n_runs', type=int, default=2,
                        help='Number of independent runs to average over (default: 10)')
    parser.add_argument('--show_realtime', action='store_true',
                        help='Show real-time visualization of original and perturbed evolutions side-by-side (first run only)')
    parser.add_argument('--epsilon', type=float, default=0.0,
                        help='Strength of per-step noise applied to all channels. Same noise is applied to both copies. Default: 0.0 (no noise)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output plot filename (default: auto-generated)')
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
    
    # Run analysis
    print(f"\nRunning Lyapunov analysis...")
    print(f"Parameters:")
    print(f"  Grid size: {args.height}x{args.width}")
    print(f"  dx={args.dx}, dy={args.dy}, dt={args.dt}")
    print(f"  tquench={args.tquench}, tevolve={args.tevolve}")
    print(f"  Perturbation: radius={args.radius}, noise_strength={args.noise_strength}")
    print(f"  Per-step noise (epsilon): {args.epsilon}")
    print(f"  Averaging over {args.n_runs} runs\n")
    
    times, distances = run_lyapunov_analysis(
        model, device, args.height, args.width,
        args.dx, args.dy, args.dt,
        args.tquench, args.tevolve,
        args.radius, args.noise_strength, args.n_runs,
        args.show_realtime, args.epsilon
    )
    
    # Check if analysis was terminated early (window closed)
    if len(times) == 0 or len(distances) == 0:
        if args.show_realtime:
            print("Analysis terminated early because visualization window was closed.")
        else:
            print("ERROR: No data to plot! Check that steps_evolve > 0.")
        return
    
    # Debug: Check data
    print(f"\nData summary:")
    print(f"  Times shape: {times.shape}, min={times.min():.2f}, max={times.max():.2f}")
    print(f"  Distances shape: {distances.shape}, min={distances.min():.4f}, max={distances.max():.4f}")
    
    # Plot results
    fig = plt.figure(figsize=(10, 6))
    plt.plot(times, distances, 'b-', linewidth=2)
    plt.xlabel('Time after perturbation', fontsize=12)
    plt.ylabel('Average pixel-wise distance', fontsize=12)
    plt.title(f'Lyapunov Analysis: {args.model_type} - {args.texture}\n'
              f'Perturbation: r={args.radius}, noise={args.noise_strength} | '
              f'Averaged over {args.n_runs} runs', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save plot (disabled)
    # if args.output is None:
    #     output_file = f"lyapunov_{args.model_type}_{args.texture}_r{args.radius}_n{args.n_runs}.png"
    # else:
    #     output_file = args.output
    # 
    # plt.savefig(output_file, dpi=150, bbox_inches='tight')
    # print(f"\nPlot saved to: {output_file}")
    
    # Display plot in window
    print("Displaying plot. Close the window to exit.")
    plt.show(block=True)


if __name__ == '__main__':
    main()

