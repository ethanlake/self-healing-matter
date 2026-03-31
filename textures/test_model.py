#!/usr/bin/env python3
"""
Test a trained NCA model by running the dynamics exactly as done in the training notebook.

Usage:
    python3 test_model.py --weights weights.pt --ep 0.0
    python3 test_model.py --weights weights.pt --ep 0.04 --steps 1024
"""

import argparse
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
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


def merge_lap(z):
    """Merge lap_x and lap_y into a single laplacian filter"""
    b, c, h, w = z.shape  # [b, 5 * chn, h, w]
    z = torch.stack([
        z[:, ::5],
        z[:, 1::5],
        z[:, 2::5],
        z[:, 3::5] + z[:, 4::5]
    ], dim=2)  # [b, chn, 4, h, w]
    return z.reshape(b, -1, h, w)  # [b, 4 * chn, h, w]


class NoiseNCA(torch.nn.Module):
    def __init__(self, chn=12, fc_dim=96, noise_level=0.1):
        super().__init__()
        self.chn = chn
        self.register_buffer("noise_level", torch.tensor([noise_level]))
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
        z = depthwise_conv(s, self.filters)  # [b, 5 * chn, h, w]
        if isinstance(dx, float) and dx == 1.0 and isinstance(dy, float) == 1.0:
            return merge_lap(z)

        if not isinstance(dx, torch.Tensor) or dx.ndim != 3:
            dx = torch.tensor([dx], device=s.device)[:, None, None]
        if not isinstance(dy, torch.Tensor) or dy.ndim != 3:
            dy = torch.tensor([dy], device=s.device)[:, None, None]

        scale = 1.0 / torch.stack([torch.ones_like(dx), dx, dy, dx ** 2, dy ** 2], dim=1)
        scale = torch.tile(scale, (1, self.chn, 1, 1))
        z = z * scale
        return merge_lap(z)

    def forward(self, s, dx=1.0, dy=1.0, dt=1.0, noise=None):
        if noise is not None:
            # Support both scalar and per-batch noise
            if isinstance(noise, torch.Tensor) and noise.ndim >= 1:
                # Per-batch noise: reshape to [b, 1, 1, 1] for broadcasting
                if noise.ndim == 1:
                    noise = noise.reshape(-1, 1, 1, 1)
            s = s + torch.randn_like(s) * noise
        z = self.perception(s, dx, dy)
        delta_s = self.w2(torch.relu(self.w1(z)))
        return s + delta_s * dt

    def seed(self, n, h=128, w=128):
        return (torch.rand(n, self.chn, h, w) - 0.5) * self.noise_level


def to_rgb(s):
    return s[..., :3, :, :] + 0.5


def main():
    parser = argparse.ArgumentParser(description='Test a trained NCA model')
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
    parser.add_argument('--epl', type=float, default=0.0,
                        help='Noise level at left edge (default: 0)')
    parser.add_argument('--epr', type=float, default=0.0,
                        help='Noise level at right edge (default: 0)')
    parser.add_argument('--aspect', type=float, default=1.0,
                        help='Aspect ratio for rectangular grid (height = size * aspect, default: 1)')
    parser.add_argument('--open', action='store_true',
                        help='Use open boundary conditions (clamp boundaries to zero)')
    parser.add_argument('--dx', type=float, default=1.0,
                        help='Spatial scale factor for dx and dy (default: 1.0)')
    parser.add_argument('--dt', type=float, default=None,
                        help='Time step scale factor (default: dx^2)')
    parser.add_argument('--last_half', action='store_true',
                        help='Only save the second half of the animation (requires --save)')

    args = parser.parse_args()

    # Default dt to dx^2 if not specified
    if args.dt is None:
        args.dt = args.dx ** 2

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

    # Extract training params if present to get channel count
    training_params = {}
    if '_training_params' in state_dict:
        training_params = state_dict['_training_params']
        print(f"Found training parameters: {training_params}")

    # Infer channel count from weights or training params
    channel_n = training_params.get('channel_n', None)
    if channel_n is None:
        # Infer from w2.weight shape: [chn, fc_dim, 1, 1]
        if 'w2.weight' in state_dict:
            channel_n = state_dict['w2.weight'].shape[0]
            print(f"Inferred channel count from weights: {channel_n}")
        else:
            channel_n = 12  # Default fallback
    else:
        print(f"Using channel count from training params: {channel_n}")

    # Filter out non-model keys (like _training_params)
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_')}

    model = NoiseNCA(chn=channel_n)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    # Compute grid dimensions
    grid_h = args.size
    grid_w = int(args.size * args.aspect)

    # Check if using spatial noise gradient
    use_spatial_noise = (args.epl != 0.0 or args.epr != 0.0)

    if use_spatial_noise:
        print(f"Running {args.steps} steps with spatial noise: epl={args.epl}, epr={args.epr}")
    else:
        print(f"Running {args.steps} steps with noise={args.epi} -> {args.epf} (hold_frac={args.hold_frac})")

    if args.aspect != 1.0:
        print(f"Using rectangular grid: {grid_h} x {grid_w} (aspect={args.aspect})")

    if args.open:
        print("Using open boundary conditions (boundaries clamped to zero)")

    if args.dx != 1.0 or args.dt != 1.0:
        print(f"Using dx=dy={args.dx}, dt={args.dt} (noise scaled by sqrt(dt)={math.sqrt(args.dt):.4f})")

    # Initialize state
    with torch.no_grad():
        s = model.seed(1, grid_h, grid_w).to(device)
        # Initialize boundaries to zero if using open boundary conditions
        if args.open:
            s[:, :, 0, :] = 0.0   # Top row
            s[:, :, -1, :] = 0.0  # Bottom row
            s[:, :, :, 0] = 0.0   # Left column
            s[:, :, :, -1] = 0.0  # Right column

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

    # Create spatial noise field if using epl/epr
    if use_spatial_noise:
        # Create a tensor that linearly interpolates from epl to epr across width
        # Shape: [1, 1, 1, grid_w] for broadcasting with [b, c, h, w]
        x_coords = torch.linspace(0, 1, grid_w, device=device)
        spatial_noise = args.epl + (args.epr - args.epl) * x_coords
        spatial_noise = spatial_noise.reshape(1, 1, 1, grid_w)  # [1, 1, 1, w]

    # Helper function to apply open boundary conditions
    def apply_open_boundaries(state):
        """Clamp all boundary pixels to zero."""
        state[:, :, 0, :] = 0.0   # Top row
        state[:, :, -1, :] = 0.0  # Bottom row
        state[:, :, :, 0] = 0.0   # Left column
        state[:, :, :, -1] = 0.0  # Right column
        return state

    # Collect frames
    frames = []
    noise_levels = []
    with torch.no_grad():
        for step in range(args.steps):
            if use_spatial_noise:
                # Use spatially varying noise (scaled by sqrt(dt))
                noise_scale = math.sqrt(args.dt)
                s = s + torch.randn_like(s) * spatial_noise * noise_scale
                z = model.perception(s, dx=args.dx, dy=args.dx)
                delta_s = model.w2(torch.relu(model.w1(z)))
                s = s + delta_s * args.dt
                noise_for_display = (args.epl + args.epr) / 2  # Average for display
            else:
                noise = get_noise_level(step, args.steps, args.epi, args.epf, args.hold_frac)
                # Scale noise by sqrt(dt)
                scaled_noise = noise * math.sqrt(args.dt)
                s = model(s, dx=args.dx, dy=args.dx, dt=args.dt, noise=scaled_noise)
                noise_for_display = noise

            # Apply open boundary conditions if enabled
            if args.open:
                s = apply_open_boundaries(s)

            if step % args.skip == 0:
                rgb = to_rgb(s)[0].permute(1, 2, 0).cpu().numpy()
                rgb = np.clip(rgb, 0, 1)
                frames.append(rgb)
                noise_levels.append(noise_for_display)
            if step % 100 == 0:
                print(f"  Step {step}/{args.steps}")

    print(f"Collected {len(frames)} frames")

    # Use only the second half of frames if --last_half is specified
    if args.last_half:
        half_idx = len(frames) // 2
        frames = frames[half_idx:]
        noise_levels = noise_levels[half_idx:]
        print(f"Using last half: {len(frames)} frames")

    # Create animation with appropriate figure size for aspect ratio
    fig_width = 6 * args.aspect
    fig_height = 6
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.axis('off')

    im = ax.imshow(frames[0])

    if use_spatial_noise:
        title_text = r'$\epsilon_L = %.2f, \epsilon_R = %.2f$' % (args.epl, args.epr)
        if args.open:
            title_text += ' (open BC)'
        title = ax.set_title(title_text, fontsize=24, pad=20)
    else:
        title_text = r'$\epsilon = %.2f$' % noise_levels[0]
        if args.open:
            title = ax.set_title(title_text + ' (open BC)', fontsize=28, pad=20)
        else:
            title = ax.set_title(title_text, fontsize=32, pad=20)

    def update(frame_idx):
        im.set_array(frames[frame_idx])
        if not use_spatial_noise:
            new_title = r'$\epsilon = %.2f$' % noise_levels[frame_idx]
            if args.open:
                new_title += ' (open BC)'
            title.set_text(new_title)
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
