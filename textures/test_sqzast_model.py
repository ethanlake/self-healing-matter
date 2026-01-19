#!/usr/bin/env python3
"""
Test a trained SQZAST NCA model by running the dynamics exactly as done in the training notebook.

SQZAST models have a theta channel (channel 3 by default) with FIXED dynamics:
- The theta channel evolves under the SQZAST update rule (not learned)
- All other channels evolve under the learned network dynamics
- Theta acts as a local orientation field that rotates gradient perception

Usage:
    python3 test_sqzast_model.py --weights weights.pt --ep 0.0
    python3 test_sqzast_model.py --weights weights.pt --ep 0.04 --steps 1024
    python3 test_sqzast_model.py --weights weights.pt --ep 0.0 --show-theta
"""

import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.colors as mcolors
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


def depthwise_conv(x, filters):
    """filters: [filter_n, h, w]"""
    b, ch, h, w = x.shape
    y = x.reshape(b * ch, 1, h, w)
    y = torch.nn.functional.pad(y, [1, 1, 1, 1], "circular")
    y = torch.nn.functional.conv2d(y, filters[:, None])
    return y.reshape(b, -1, h, w)


def sqzast_theta_update(theta, dt=1.0):
    """
    Fixed SQZAST update rule for the theta (orientation) channel.

    1. Random binary mask selects sites to update (probability 0.5)
    2. Each selected site picks a random direction m in {1,2,3,4} (N,S,E,W)
    3. For m=1 (North): dtheta = theta_north - theta; theta += dt * dtheta * step(-sin(2π*dtheta))
    4. For m=2 (South): dtheta = theta_south - theta; theta += dt * dtheta * step(-sin(2π*dtheta))
    5. For m=3 (East):  dtheta = theta_east - theta;  theta += dt * dtheta * step(+sin(2π*dtheta))
    6. For m=4 (West):  dtheta = theta_west - theta;  theta += dt * dtheta * step(+sin(2π*dtheta))

    Note: theta is in [0,1] representing [0, 2π] radians, so we use sin(2π*dtheta)
    to match the angular interpretation used in the perception rotation.

    Args:
        theta: [b, 1, h, w] tensor of theta values
        dt: time step

    Returns:
        Updated theta tensor
    """
    b, _, h, w = theta.shape
    device = theta.device

    # Generate random binary mask (probability 0.5 for each site)
    mask = (torch.rand(b, 1, h, w, device=device) > 0.5).float()

    # Generate random direction for each site: 1=N, 2=S, 3=E, 4=W
    directions = torch.randint(1, 5, (b, 1, h, w), device=device)

    # Get neighbors using circular (periodic) boundary conditions
    # North neighbor: roll by -1 along height axis (row above)
    theta_north = torch.roll(theta, shifts=-1, dims=2)
    # South neighbor: roll by +1 along height axis (row below)
    theta_south = torch.roll(theta, shifts=1, dims=2)
    # East neighbor: roll by +1 along width axis (column to the right)
    theta_east = torch.roll(theta, shifts=1, dims=3)
    # West neighbor: roll by -1 along width axis (column to the left)
    theta_west = torch.roll(theta, shifts=-1, dims=3)

    # Compute dtheta for each direction
    dtheta_north = theta_north - theta
    dtheta_south = theta_south - theta
    dtheta_east = theta_east - theta
    dtheta_west = theta_west - theta

    # step function: step(x) = 0 if x < 0, 1 otherwise
    # For N,S: use step(-sin(2*pi*dtheta))
    # For E,W: use step(+sin(2*pi*dtheta))
    # Note: theta is in [0,1] representing [0, 2pi] radians, so we need 2*pi*dtheta
    two_pi = 2 * np.pi
    step_north = (torch.sin(two_pi * dtheta_north) <= 0).float()  # step(-sin(2*pi*dtheta))
    step_south = (torch.sin(two_pi * dtheta_south) <= 0).float()  # step(-sin(2*pi*dtheta))
    step_east = (torch.sin(two_pi * dtheta_east) >= 0).float()    # step(+sin(2*pi*dtheta))
    step_west = (torch.sin(two_pi * dtheta_west) >= 0).float()    # step(+sin(2*pi*dtheta))

    # Compute updates for each direction
    update_north = dt * dtheta_north * step_north
    update_south = dt * dtheta_south * step_south
    update_east = dt * dtheta_east * step_east
    update_west = dt * dtheta_west * step_west

    # Select the update based on direction
    update = torch.where(directions == 1, update_north,
             torch.where(directions == 2, update_south,
             torch.where(directions == 3, update_east,
                         update_west)))  # directions == 4

    # Apply mask and update theta
    theta_new = theta + mask * update

    return theta_new


class SqzastNCA(torch.nn.Module):
    """
    SQZAST Neural Cellular Automata.

    Similar to Rotation-Invariant NCA, but:
    - Channel 3 (theta) has FIXED dynamics defined by the SQZAST update rule
    - The theta channel is NOT learned, only the other channels are
    - Theta still acts as a local orientation field that rotates gradient perception
    - The bare value of theta is zeroed in the perception input (only gradients reach the network)
    """
    def __init__(self, chn=12, fc_dim=96, noise_level=1.0, theta_channel=3):
        super().__init__()
        self.chn = chn
        self.theta_channel = theta_channel  # First hidden channel after RGB (0,1,2)
        self.register_buffer("noise_level", torch.tensor([noise_level]))

        # Note: perception produces 4 features per channel, but we only learn
        # updates for non-theta channels. However, we still need full perception
        # because theta affects the rotated gradients for all channels.
        self.w1 = torch.nn.Conv2d(chn * 4, fc_dim, 1, bias=True)
        self.w2 = torch.nn.Conv2d(fc_dim, chn, 1, bias=False)

        torch.nn.init.xavier_normal_(self.w1.weight, gain=0.2)
        torch.nn.init.zeros_(self.w2.weight)

        with torch.no_grad():
            ident = torch.tensor([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.0]])
            sobel_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
            lap_x = torch.tensor([[0.5, 0.0, 0.5], [2.0, -6.0, 2.0], [0.5, 0.0, 0.5]])
            self.filters = torch.stack([ident, sobel_x, sobel_x.T, lap_x, lap_x.T])

    def perception(self, s, dx=1.0, dy=1.0):
        """
        Compute rotation-invariant perception.

        For each channel, outputs 4 features:
        - identity (zeroed for theta channel)
        - rotated gradient x: cos(2*pi*theta)*sobel_x + sin(2*pi*theta)*sobel_y
        - rotated gradient y: -sin(2*pi*theta)*sobel_x + cos(2*pi*theta)*sobel_y
        - laplacian (rotation-invariant)
        """
        # Apply all 5 filters via depthwise conv
        z = depthwise_conv(s, self.filters)  # [b, 5*chn, h, w]

        # Extract theta (channel 3) and compute rotation
        theta = s[:, self.theta_channel:self.theta_channel+1, :, :]  # [b, 1, h, w]
        cos_t = torch.cos(2 * np.pi * theta)  # [b, 1, h, w]
        sin_t = torch.sin(2 * np.pi * theta)  # [b, 1, h, w]

        # Extract filter outputs (interleaved: id, sx, sy, lx, ly for each channel)
        ident_all = z[:, 0::5]    # [b, chn, h, w]
        sobel_x_all = z[:, 1::5]  # [b, chn, h, w]
        sobel_y_all = z[:, 2::5]  # [b, chn, h, w]
        lap_x_all = z[:, 3::5]    # [b, chn, h, w]
        lap_y_all = z[:, 4::5]    # [b, chn, h, w]

        # Rotate gradients using theta
        rotated_x = cos_t * sobel_x_all + sin_t * sobel_y_all
        rotated_y = -sin_t * sobel_x_all + cos_t * sobel_y_all

        # Combine laplacians (rotation-invariant)
        laplacian = lap_x_all + lap_y_all

        # Zero out bare theta value (channel 3 identity)
        # Only theta gradients should reach the network
        ident_all = ident_all.clone()
        ident_all[:, self.theta_channel, :, :] = 0.0

        # Stack features: [identity, rotated_x, rotated_y, laplacian] for each channel
        # Result: [b, 4*chn, h, w]
        perception = torch.stack([ident_all, rotated_x, rotated_y, laplacian], dim=2)
        return perception.reshape(s.shape[0], -1, s.shape[2], s.shape[3])

    def forward(self, s, dx=1.0, dy=1.0, dt=1.0, noise=None):
        # Add noise to all channels (including theta)
        if noise is not None:
            # Support both scalar and per-batch noise
            if isinstance(noise, torch.Tensor) and noise.ndim >= 1:
                if noise.ndim == 1:
                    noise = noise.reshape(-1, 1, 1, 1)
            s = s + torch.randn_like(s) * noise

        # Compute perception and learned dynamics
        z = self.perception(s, dx, dy)
        delta_s = self.w2(torch.relu(self.w1(z)))

        # Apply learned dynamics to all channels
        s_new = s + delta_s * dt

        # Override theta channel with fixed SQZAST dynamics
        # Get theta from the noisy input state (before learned update)
        theta_old = s[:, self.theta_channel:self.theta_channel+1, :, :]

        # Apply the fixed SQZAST update rule to theta
        theta_new = sqzast_theta_update(theta_old, dt=dt)

        # Replace theta channel in the output
        s_new = s_new.clone()
        s_new[:, self.theta_channel:self.theta_channel+1, :, :] = theta_new

        return s_new

    def seed(self, n, h=128, w=128, noise_level=None):
        """
        Create initial seed states.

        Args:
            n: Number of samples (or list of per-sample noise levels)
            h: Height
            w: Width
            noise_level: Noise level(s) for initial state
        """
        if isinstance(n, (list, np.ndarray, torch.Tensor)):
            noise_level = n
            n = len(noise_level)

        if noise_level is None:
            nl = self.noise_level.item()
            return (torch.rand(n, self.chn, h, w) - 0.5) * nl
        elif isinstance(noise_level, (int, float)):
            return (torch.rand(n, self.chn, h, w) - 0.5) * noise_level
        else:
            if isinstance(noise_level, np.ndarray):
                noise_level = torch.tensor(noise_level, dtype=torch.float32)
            noise_level = noise_level.reshape(-1, 1, 1, 1)
            return (torch.rand(n, self.chn, h, w) - 0.5) * noise_level


def to_rgb(s):
    return s[..., :3, :, :] + 0.5


def to_theta(s, theta_channel=3):
    """Extract theta channel and map to [0, 1] for visualization."""
    return (s[..., theta_channel:theta_channel+1, :, :] + 0.5).clamp(0, 1)


def theta_to_hsv_rgb(theta_vals):
    """Convert theta values to RGB using HSV color wheel."""
    # theta_vals shape: [h, w] with values in [0, 1]
    h, w = theta_vals.shape
    hsv = np.zeros((h, w, 3))
    hsv[..., 0] = theta_vals  # hue
    hsv[..., 1] = 1.0         # saturation
    hsv[..., 2] = 1.0         # value
    return mcolors.hsv_to_rgb(hsv)


def main():
    parser = argparse.ArgumentParser(description='Test a trained SQZAST NCA model')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the .pt weights file')
    parser.add_argument('--epi', type=float, default=0.0,
                        help='Initial noise strength (epsilon)')
    parser.add_argument('--epf', type=float, default=None,
                        help='Final noise strength (defaults to epi)')
    parser.add_argument('--hold_frac', type=float, default=0.0,
                        help='Fraction of steps to hold at epi/epf at start/end')
    parser.add_argument('--steps', type=int, default=256,
                        help='Number of steps to run (default: 256)')
    parser.add_argument('--size', type=int, default=128,
                        help='Grid size (default: 128)')
    parser.add_argument('--save', type=str, default=None,
                        help='Save animation to file (e.g., output.mp4)')
    parser.add_argument('--fps', type=int, default=30,
                        help='Frames per second for animation (default: 30)')
    parser.add_argument('--skip', type=int, default=1,
                        help='Show every Nth frame (default: 1)')
    parser.add_argument('--show-theta', action='store_true',
                        help='Show theta field alongside RGB output')
    parser.add_argument('--theta-channel', type=int, default=3,
                        help='Theta channel index (default: 3)')

    args = parser.parse_args()

    # Default epf to epi if not specified
    if args.epf is None:
        args.epf = args.epi

    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load model
    print(f"Loading weights from: {args.weights}")
    state_dict = torch.load(args.weights, map_location=device)

    # Handle checkpoint vs raw state dict
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']

    # Extract training params if present
    training_params = {}
    if '_training_params' in state_dict:
        training_params = state_dict['_training_params']
        print(f"Found training parameters:")
        for k, v in training_params.items():
            print(f"  {k}: {v}")

    # Get theta channel from training params or args
    theta_channel = training_params.get('theta_channel', args.theta_channel)
    print(f"Theta channel: {theta_channel}")

    # Filter out non-model keys (like _training_params)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}

    model = SqzastNCA(theta_channel=theta_channel)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    print(f"Running {args.steps} steps with noise={args.epi} -> {args.epf} (hold_frac={args.hold_frac})")
    print(f"SQZAST model: theta channel {theta_channel} has FIXED dynamics")

    # Initialize state
    with torch.no_grad():
        s = model.seed(1, args.size, args.size).to(device)

    # Compute noise level as a function of time
    def get_noise_level(t, steps, epi, epf, hold_frac):
        hold_steps = hold_frac * steps
        if t < hold_steps:
            return epi
        elif t > steps - hold_steps:
            return epf
        else:
            # Linear interpolation in the middle region
            t_start = hold_steps
            t_end = steps - hold_steps
            alpha = (t - t_start) / (t_end - t_start)
            return epi + alpha * (epf - epi)

    # Collect frames
    frames = []
    theta_frames = []
    noise_levels = []
    with torch.no_grad():
        for step in range(args.steps):
            noise = get_noise_level(step, args.steps, args.epi, args.epf, args.hold_frac)
            s = model(s, noise=noise)
            if step % args.skip == 0:
                rgb = to_rgb(s)[0].permute(1, 2, 0).cpu().numpy()
                rgb = np.clip(rgb, 0, 1)
                frames.append(rgb)

                # Also capture theta for visualization
                theta_val = to_theta(s, theta_channel)[0, 0].cpu().numpy()
                theta_frames.append(theta_to_hsv_rgb(theta_val))

                noise_levels.append(noise)
            if step % 100 == 0:
                print(f"  Step {step}/{args.steps}")

    print(f"Collected {len(frames)} frames")

    # Create animation
    if args.show_theta:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        ax1.axis('off')
        ax2.axis('off')

        im1 = ax1.imshow(frames[0])
        im2 = ax2.imshow(theta_frames[0])

        ax1.set_title('RGB Output', fontsize=16)
        ax2.set_title(r'Theta Field ($\theta$)', fontsize=16)

        title = fig.suptitle(r'$\epsilon = %.2f$' % noise_levels[0], fontsize=24)

        def update(frame_idx):
            im1.set_array(frames[frame_idx])
            im2.set_array(theta_frames[frame_idx])
            title.set_text(r'$\epsilon = %.2f$' % noise_levels[frame_idx])
            return [im1, im2, title]
    else:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.axis('off')

        im = ax.imshow(frames[0])
        title = ax.set_title(r'$\epsilon = %.2f$' % noise_levels[0], fontsize=32, pad=20)

        def update(frame_idx):
            im.set_array(frames[frame_idx])
            title.set_text(r'$\epsilon = %.2f$' % noise_levels[frame_idx])
            return [im, title]

    ani = animation.FuncAnimation(
        fig, update, frames=len(frames),
        interval=1000 / args.fps, blit=False
    )

    if args.save:
        print(f"Saving animation to: {args.save}")
        fig.tight_layout(pad=0.1)
        if args.save.endswith('.mp4'):
            try:
                ani.save(args.save, fps=args.fps, dpi=100, writer='ffmpeg')
            except Exception as e:
                print(f"ffmpeg not available ({e}), falling back to gif")
                gif_path = args.save.replace('.mp4', '.gif')
                ani.save(gif_path, fps=args.fps, dpi=100, writer='pillow')
                print(f"Saved as: {gif_path}")
        else:
            ani.save(args.save, fps=args.fps, dpi=100, writer='pillow')
        print("Done!")
    else:
        plt.show()


if __name__ == '__main__':
    main()
