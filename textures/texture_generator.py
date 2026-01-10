"""
Generate synthetic texture images for NCA training.
"""

import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter
import argparse
import os


def generate_vertical_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform vertical black and white stripes.
    
    num_stripes should be even to have equal black and white stripes.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for x in range(size):
        if smooth:
            # Sinusoidal variation: 0.5 + 0.5*cos gives [0, 1]
            val = 0.5 + 0.5 * np.cos(2 * np.pi * x * num_stripes / size)
            img[:, x, :] = int(val * 255)
        else:
            stripe_idx = int(x * num_stripes / size)
            if stripe_idx % 2 == 0:
                img[:, x, :] = 255
    
    return img


def generate_horizontal_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform horizontal black and white stripes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        if smooth:
            val = 0.5 + 0.5 * np.cos(2 * np.pi * y * num_stripes / size)
            img[y, :, :] = int(val * 255)
        else:
            stripe_idx = int(y * num_stripes / size)
            if stripe_idx % 2 == 0:
                img[y, :, :] = 255
    
    return img


def generate_diagonal_stripes(size=256, num_stripes=10, smooth=False):
    """Generate an image with uniform diagonal (45 degree) black and white stripes."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    for y in range(size):
        for x in range(size):
            diag_pos = (x + y) / (2 * size - 2)
            if smooth:
                val = 0.5 + 0.5 * np.cos(2 * np.pi * diag_pos * num_stripes)
                img[y, x, :] = int(val * 255)
            else:
                stripe_idx = int(diag_pos * num_stripes)
                if stripe_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_bullseye(size=256, num_stripes=10, smooth=False):
    """Generate an image with concentric black and white rings (bullseye pattern)."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = size / 2
    max_radius = size / 2 * np.sqrt(2)
    
    for y in range(size):
        for x in range(size):
            r = np.sqrt((x - center + 0.5)**2 + (y - center + 0.5)**2)
            if smooth:
                val = 0.5 + 0.5 * np.cos(2 * np.pi * r * num_stripes / max_radius)
                img[y, x, :] = int(val * 255)
            else:
                ring_idx = int(r * num_stripes / max_radius)
                if ring_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_burst(size=256, num_rays=10, smooth=False):
    """Generate an image with alternating black and white rays focusing to the center.
    
    num_rays should be even to have equal black and white rays.
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    center = size / 2
    
    for y in range(size):
        for x in range(size):
            # Calculate angle from center (in radians)
            dx = x - center + 0.5
            dy = y - center + 0.5
            angle = np.arctan2(dy, dx)
            # Normalize angle to [0, 2π]
            if angle < 0:
                angle += 2 * np.pi
            
            if smooth:
                # Sinusoidal variation based on angle
                val = 0.5 + 0.5 * np.cos(angle * num_rays)
                img[y, x, :] = int(val * 255)
            else:
                # Map angle to ray index
                ray_idx = int(angle * num_rays / (2 * np.pi)) % num_rays
                if ray_idx % 2 == 0:
                    img[y, x, :] = 255
    
    return img


def generate_french_flag(size=256, num_stripes=None, smooth=False):
    """Generate a French flag pattern with three vertical stripes: blue, white, red."""
    img = np.zeros((size, size, 3), dtype=np.uint8)
    
    # French flag colors (RGB)
    blue = np.array([0, 85, 164], dtype=np.uint8)
    white = np.array([255, 255, 255], dtype=np.uint8)
    red = np.array([239, 65, 53], dtype=np.uint8)
    
    # Divide width into three equal parts
    third = size / 3
    
    for y in range(size):
        for x in range(size):
            if x < third:
                # Left third: blue
                img[y, x, :] = blue
            elif x < 2 * third:
                # Middle third: white
                img[y, x, :] = white
            else:
                # Right third: red
                img[y, x, :] = red
    
    return img


def generate_japanese_flag(size=256, num_stripes=None, smooth=False):
    """Generate a Japanese flag pattern: white background with red circle in center."""
    img = np.ones((size, size, 3), dtype=np.uint8) * 255  # White background

    # Japanese flag colors (RGB)
    red = np.array([188, 0, 45], dtype=np.uint8)  # Standard Japanese red

    center = size / 2
    # Red circle diameter is typically 3/5 of the flag height
    radius = (size * 3) / 10

    for y in range(size):
        for x in range(size):
            # Calculate distance from center
            dx = x - center + 0.5
            dy = y - center + 0.5
            dist = np.sqrt(dx**2 + dy**2)

            if dist <= radius:
                img[y, x, :] = red

    return img


def generate_mono_polka(size=256, n_dots=5, num_stripes=None, smooth=False, border=False):
    """Generate a square grid of orange polka dots on white background.

    Args:
        size: Image size (width and height, always square)
        n_dots: Number of polka dots per side of the grid (e.g., 5 means 5x5 grid)
        num_stripes: Not used, kept for API consistency
        smooth: Not used, kept for API consistency
        border: If True, add a thin darker border around each dot

    The polka dots are arranged in a grid with spacing equal to their diameter,
    and positioned so they don't get cut off at boundaries.
    """
    # White background - ensure square
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Orange color (RGB)
    orange = np.array([255, 165, 0], dtype=np.uint8)
    dark_orange = np.array([180, 100, 0], dtype=np.uint8)  # Darker orange for border

    # Calculate radius from number of dots
    radius = size / (2.0 * (2 * n_dots - 1))

    # Calculate grid parameters
    diameter = 2 * radius
    spacing = diameter  # Distance between centers equals diameter

    # Calculate total span and centering offset
    total_span = n_dots * diameter + (n_dots - 1) * spacing
    offset = (size - total_span) / 2

    # Border thickness as fraction of radius
    border_thickness = 0.08 * radius if border else 0.0

    # Draw polka dots
    for i in range(n_dots):
        for j in range(n_dots):
            # Center of this dot
            cx = offset + i * (diameter + spacing) + radius
            cy = offset + j * (diameter + spacing) + radius

            # Draw the dot
            for y in range(size):
                for x in range(size):
                    dx = x - cx + 0.5
                    dy = y - cy + 0.5
                    dist = np.sqrt(dx**2 + dy**2)

                    if border and dist <= radius and dist > radius - border_thickness:
                        # Draw border
                        img[y, x, :] = dark_orange
                    elif dist <= radius - border_thickness:
                        # Draw main dot
                        img[y, x, :] = orange
                    elif not border and dist <= radius:
                        # No border, just draw dot
                        img[y, x, :] = orange

    return img


def generate_random_polka(size=256, n_dots=5, num_stripes=None, smooth=False, border=False):
    """Generate a square grid of randomly colored polka dots on white background.

    Args:
        size: Image size (width and height, always square)
        n_dots: Number of polka dots per side of the grid (e.g., 5 means 5x5 grid)
        num_stripes: Not used, kept for API consistency
        smooth: Not used, kept for API consistency
        border: If True, add a thin darker border around each dot

    Each dot is randomly assigned either orange or sky blue color.
    The polka dots are arranged in a grid with spacing equal to their diameter,
    and positioned so they don't get cut off at boundaries.
    """
    # White background - ensure square
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Orange and sky blue colors (RGB)
    orange = np.array([255, 165, 0], dtype=np.uint8)
    sky_blue = np.array([135, 206, 235], dtype=np.uint8)
    dark_orange = np.array([180, 100, 0], dtype=np.uint8)
    dark_blue = np.array([70, 130, 180], dtype=np.uint8)

    # Calculate radius from number of dots
    radius = size / (2.0 * (2 * n_dots - 1))

    # Calculate grid parameters
    diameter = 2 * radius
    spacing = diameter  # Distance between centers equals diameter

    # Calculate total span and centering offset
    total_span = n_dots * diameter + (n_dots - 1) * spacing
    offset = (size - total_span) / 2

    # Border thickness as fraction of radius
    border_thickness = 0.08 * radius if border else 0.0

    # Draw polka dots
    for i in range(n_dots):
        for j in range(n_dots):
            # Center of this dot
            cx = offset + i * (diameter + spacing) + radius
            cy = offset + j * (diameter + spacing) + radius

            # Randomly choose orange or sky blue
            if np.random.rand() < 0.5:
                color = orange
                border_color = dark_orange
            else:
                color = sky_blue
                border_color = dark_blue

            # Draw the dot
            for y in range(size):
                for x in range(size):
                    dx = x - cx + 0.5
                    dy = y - cy + 0.5
                    dist = np.sqrt(dx**2 + dy**2)

                    if border and dist <= radius and dist > radius - border_thickness:
                        # Draw border
                        img[y, x, :] = border_color
                    elif dist <= radius - border_thickness:
                        # Draw main dot
                        img[y, x, :] = color
                    elif not border and dist <= radius:
                        # No border, just draw dot
                        img[y, x, :] = color

    return img


def generate_checkered_polka(size=256, n_dots=5, num_stripes=None, smooth=False, border=False):
    """Generate a checkered square grid of orange and sky blue polka dots on white background.

    Args:
        size: Image size (width and height, always square)
        n_dots: Number of polka dots per side of the grid (e.g., 5 means 5x5 grid)
        num_stripes: Not used, kept for API consistency
        smooth: Not used, kept for API consistency
        border: If True, add a thin darker border around each dot

    Dots at position (i,j) are orange if i+j is even, sky blue if i+j is odd.
    The polka dots are arranged in a grid with spacing equal to their diameter,
    and positioned so they don't get cut off at boundaries.
    """
    # White background - ensure square
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # Orange and sky blue colors (RGB)
    orange = np.array([255, 165, 0], dtype=np.uint8)
    sky_blue = np.array([135, 206, 235], dtype=np.uint8)
    dark_orange = np.array([180, 100, 0], dtype=np.uint8)
    dark_blue = np.array([70, 130, 180], dtype=np.uint8)  # Darker sky blue

    # Calculate radius from number of dots
    radius = size / (2.0 * (2 * n_dots - 1))

    # Calculate grid parameters
    diameter = 2 * radius
    spacing = diameter  # Distance between centers equals diameter

    # Calculate total span and centering offset
    total_span = n_dots * diameter + (n_dots - 1) * spacing
    offset = (size - total_span) / 2

    # Border thickness as fraction of radius
    border_thickness = 0.08 * radius if border else 0.0

    # Draw polka dots
    for i in range(n_dots):
        for j in range(n_dots):
            # Center of this dot
            cx = offset + i * (diameter + spacing) + radius
            cy = offset + j * (diameter + spacing) + radius

            # Choose color based on checkered pattern
            if (i + j) % 2 == 0:
                color = orange
                border_color = dark_orange
            else:
                color = sky_blue
                border_color = dark_blue

            # Draw the dot
            for y in range(size):
                for x in range(size):
                    dx = x - cx + 0.5
                    dy = y - cy + 0.5
                    dist = np.sqrt(dx**2 + dy**2)

                    if border and dist <= radius and dist > radius - border_thickness:
                        # Draw border
                        img[y, x, :] = border_color
                    elif dist <= radius - border_thickness:
                        # Draw main dot
                        img[y, x, :] = color
                    elif not border and dist <= radius:
                        # No border, just draw dot
                        img[y, x, :] = color

    return img


def save_texture(img, filename, quality=95):
    """Save image in appropriate format based on file extension."""
    pil_img = Image.fromarray(img)
    if filename.lower().endswith('.png'):
        pil_img.save(filename, 'PNG')
    else:
        pil_img.save(filename, 'JPEG', quality=quality)


def main():
    parser = argparse.ArgumentParser(description='Generate synthetic texture images')
    parser.add_argument('--pattern', type=str, default='vertical_stripes',
                        choices=['vertical_stripes', 'horizontal_stripes', 'diagonal_stripes', 'bullseye', 'burst', 'french_flag', 'japanese_flag', 'mono_polka', 'checkered_polka', 'random_polka'],
                        help='Pattern type to generate')
    parser.add_argument('--output', type=str, default='generated_texture.jpg',
                        help='Output filename')
    parser.add_argument('--size', type=int, default=256,
                        help='Image size (square)')
    parser.add_argument('--num_stripes', type=int, default=10,
                        help='Number of stripes (for stripe patterns)')
    parser.add_argument('--n', type=int, default=5,
                        help='Number of polka dots per side of grid (for mono_polka pattern). Default: 5 (creates 5x5 grid)')
    parser.add_argument('--smoothing_rad', type=float, default=0.0,
                        help='Gaussian smoothing radius. If non-zero, apply Gaussian blur with this sigma value.')
    parser.add_argument('--smooth', action='store_true',
                        help='Use sinusoidal variation instead of sharp stripes')
    parser.add_argument('--border', action='store_true',
                        help='Add a thin darker border around each polka dot (for polka patterns only)')

    args = parser.parse_args()
    
    if args.pattern == 'vertical_stripes':
        img = generate_vertical_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'horizontal_stripes':
        img = generate_horizontal_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'diagonal_stripes':
        img = generate_diagonal_stripes(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'bullseye':
        img = generate_bullseye(size=args.size, num_stripes=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'burst':
        img = generate_burst(size=args.size, num_rays=args.num_stripes, smooth=args.smooth)
    elif args.pattern == 'french_flag':
        img = generate_french_flag(size=args.size, smooth=args.smooth)
    elif args.pattern == 'japanese_flag':
        img = generate_japanese_flag(size=args.size, smooth=args.smooth)
        # Use PNG for Japanese flag to avoid JPEG artifacts on circle edges
        if not args.output.lower().endswith('.png'):
            args.output = args.output.rsplit('.', 1)[0] + '.png'
    elif args.pattern == 'mono_polka':
        img = generate_mono_polka(size=args.size, n_dots=args.n, smooth=args.smooth, border=args.border)
        # Use PNG for polka dots to avoid JPEG artifacts on circle edges
        if not args.output.lower().endswith('.png'):
            args.output = args.output.rsplit('.', 1)[0] + '.png'
    elif args.pattern == 'checkered_polka':
        img = generate_checkered_polka(size=args.size, n_dots=args.n, smooth=args.smooth, border=args.border)
        # Use PNG for polka dots to avoid JPEG artifacts on circle edges
        if not args.output.lower().endswith('.png'):
            args.output = args.output.rsplit('.', 1)[0] + '.png'
    elif args.pattern == 'random_polka':
        img = generate_random_polka(size=args.size, n_dots=args.n, smooth=args.smooth, border=args.border)
        # Use PNG for polka dots to avoid JPEG artifacts on circle edges
        if not args.output.lower().endswith('.png'):
            args.output = args.output.rsplit('.', 1)[0] + '.png'

    # Apply Gaussian smoothing if requested
    if args.smoothing_rad > 0:
        # Convert RGB to HSV for better color preservation
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray(img)
        hsv_img = pil_img.convert('HSV')
        hsv_array = np.array(hsv_img).astype(np.float32)

        # Smooth Saturation and Value channels, but NOT Hue
        # This prevents color distortion from averaging hue (which is circular)
        h_channel = hsv_array[:, :, 0]  # Hue - don't smooth (circular values)
        s_channel = hsv_array[:, :, 1]  # Saturation - smooth this
        v_channel = hsv_array[:, :, 2]  # Value - smooth this

        # Apply Gaussian filter to Saturation and Value channels
        s_smoothed = gaussian_filter(s_channel, sigma=args.smoothing_rad, mode='reflect')
        v_smoothed = gaussian_filter(v_channel, sigma=args.smoothing_rad, mode='reflect')

        # Reconstruct HSV with smoothed S and V
        hsv_smoothed = np.stack([h_channel, s_smoothed, v_smoothed], axis=2)

        # Clip and convert back to uint8
        hsv_smoothed = np.clip(hsv_smoothed, 0, 255).astype(np.uint8)

        # Convert back to RGB
        hsv_pil = PILImage.fromarray(hsv_smoothed, mode='HSV')
        img = np.array(hsv_pil.convert('RGB'))
        print(f"Applied Gaussian smoothing in HSV space (S and V channels) with sigma={args.smoothing_rad}")

    save_texture(img, args.output)
    print(f"Generated {args.pattern} texture: {args.output}")


if __name__ == '__main__':
    main()

