#!/usr/bin/env python3
"""
Add a trained model and its target image to the demo.

Usage:
    python3 add_model.py --weights weight_file.pt --image image.png --model_name my_model
    python3 add_model.py --weights weight_file.pt --image image.jpg --model_name my_model --model_type Noise-NCA
    python3 add_model.py --weights weight_file.pt --image image.png --model_name my_model --category objects

    # Noise-controlled models (2 or 3 textures):
    python3 add_model.py --weights weights.pt --model_name my_nc_model --noise_controlled --images img1.png img2.png
    python3 add_model.py --weights weights.pt --model_name my_nc_model --noise_controlled --images img1.png img2.png img3.png

Categories:
    textures (default): Models trained on texture images with uniform noise initialization
    objects: Models trained on object images with center seed initialization (Growing NCA style)
"""

import argparse
import os
import shutil
import subprocess
import sys
import json
from PIL import Image

# Valid categories and their metadata keys
VALID_CATEGORIES = {
    'textures': 'texture_names',
    'objects': 'object_names',
}


def confirm_overwrite(filepath):
    """Ask user if they want to overwrite an existing file."""
    if not os.path.exists(filepath):
        return True

    while True:
        response = input(f"File '{filepath}' already exists. Overwrite? (y/n): ").strip().lower()
        if response in ['y', 'yes']:
            return True
        elif response in ['n', 'no']:
            return False
        else:
            print("Please answer 'y' or 'n'")


def create_hybrid_image(image_paths):
    """
    Create a hybrid image from 2 or 3 input images.

    For 2 images: left half from image 1, right half from image 2
    For 3 images: left third from image 1, center third from image 2, right third from image 3

    All images are resized to match the first image's dimensions.

    Args:
        image_paths: List of 2 or 3 image paths

    Returns:
        PIL Image object of the hybrid image
    """
    if len(image_paths) not in [2, 3]:
        raise ValueError(f"Expected 2 or 3 images, got {len(image_paths)}")

    # Load and convert all images to RGB
    images = []
    for path in image_paths:
        img = Image.open(path)
        if img.mode != 'RGB':
            img = img.convert('RGB')
        images.append(img)

    # Use first image's size as reference
    width, height = images[0].size

    # Resize other images to match
    for i in range(1, len(images)):
        if images[i].size != (width, height):
            images[i] = images[i].resize((width, height), Image.Resampling.LANCZOS)

    # Create output image
    hybrid = Image.new('RGB', (width, height))

    if len(images) == 2:
        # Left half from image 1, right half from image 2
        left_width = width // 2
        right_width = width - left_width

        # Crop left half from image 1
        left_part = images[0].crop((0, 0, left_width, height))
        # Crop right half from image 2
        right_part = images[1].crop((left_width, 0, width, height))

        hybrid.paste(left_part, (0, 0))
        hybrid.paste(right_part, (left_width, 0))

    else:  # 3 images
        # Left third from image 1, center third from image 2, right third from image 3
        third_width = width // 3
        remainder = width - (third_width * 3)

        # Distribute remainder to make widths sum to total
        left_width = third_width
        center_width = third_width + remainder  # Give remainder to center
        right_width = third_width

        left_end = left_width
        center_end = left_end + center_width

        # Crop each third from corresponding image
        left_part = images[0].crop((0, 0, left_width, height))
        center_part = images[1].crop((left_end, 0, center_end, height))
        right_part = images[2].crop((center_end, 0, width, height))

        hybrid.paste(left_part, (0, 0))
        hybrid.paste(center_part, (left_end, 0))
        hybrid.paste(right_part, (center_end, 0))

    return hybrid


def main():
    parser = argparse.ArgumentParser(description='Add a trained model and image to the demo')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the .pt weights file')
    parser.add_argument('--image', type=str, default=None,
                        help='Path to the target image (.png or .jpg) - required unless --noise_controlled')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for the model (used for filenames)')
    parser.add_argument('--model_type', type=str, default='Noise-NCA',
                        help='Model type for folder organization (default: Noise-NCA)')
    parser.add_argument('--category', type=str, default='textures',
                        choices=list(VALID_CATEGORIES.keys()),
                        help='Model category: textures (uniform noise init) or objects (center seed init)')
    parser.add_argument('--noise_controlled', action='store_true',
                        help='Noise-controlled model with multiple textures. Use --images instead of --image.')
    parser.add_argument('--images', type=str, nargs='+',
                        help='Paths to 2 or 3 target images for noise-controlled models')

    args = parser.parse_args()

    # Validate argument combinations
    if args.noise_controlled:
        if args.image:
            print("Warning: --image is ignored when --noise_controlled is used. Use --images instead.")
        if not args.images:
            print("Error: --noise_controlled requires --images with 2 or 3 image paths")
            sys.exit(1)
        if len(args.images) not in [2, 3]:
            print(f"Error: --images requires 2 or 3 images, got {len(args.images)}")
            sys.exit(1)
        # Validate all images exist
        for img_path in args.images:
            if not os.path.exists(img_path):
                print(f"Error: Image file '{img_path}' not found")
                sys.exit(1)
            img_ext = os.path.splitext(img_path)[1].lower()
            if img_ext not in ['.png', '.jpg', '.jpeg']:
                print(f"Error: Image must be .png or .jpg, got '{img_ext}' for '{img_path}'")
                sys.exit(1)
    else:
        if not args.image:
            print("Error: --image is required (or use --noise_controlled with --images)")
            sys.exit(1)

    # Validate input files exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found")
        sys.exit(1)

    # Validate single image mode (already validated noise_controlled mode above)
    if not args.noise_controlled:
        if not os.path.exists(args.image):
            print(f"Error: Image file '{args.image}' not found")
            sys.exit(1)

        # Get image extension
        image_ext = os.path.splitext(args.image)[1].lower()
        if image_ext not in ['.png', '.jpg', '.jpeg']:
            print(f"Error: Image must be .png or .jpg, got '{image_ext}'")
            sys.exit(1)

    # Define target paths based on category
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Image folder depends on category
    image_subdir = 'texture' if args.category == 'textures' else 'object'
    image_dir = os.path.join(script_dir, 'data', 'images', image_subdir)
    # Always save as .jpg for consistency with demo
    image_target = os.path.join(image_dir, f"{args.model_name}.jpg")

    # Weights folder: trained_models/<category>/<model_type>/<model_name>/
    # e.g., trained_models/Objects/Noise-NCA/my_model/ or trained_models/Noise-NCA/my_model/
    if args.category == 'objects':
        weights_dir = os.path.join(script_dir, 'trained_models', 'Objects', args.model_type, args.model_name)
    else:
        weights_dir = os.path.join(script_dir, 'trained_models', args.model_type, args.model_name)
    weights_target = os.path.join(weights_dir, 'weights.pt')

    models_dir = os.path.join(script_dir, 'data', 'models')
    json_target = os.path.join(models_dir, f"{args.model_name}.json")

    metadata_file = os.path.join(script_dir, 'data', 'metadata.json')

    convert_script = os.path.join(script_dir, 'convert_pt_to_json.py')

    # Check if convert script exists
    if not os.path.exists(convert_script):
        print(f"Error: Conversion script '{convert_script}' not found")
        sys.exit(1)

    # Check if metadata file exists
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file '{metadata_file}' not found")
        sys.exit(1)

    # Check for existing files and ask for confirmation
    if not confirm_overwrite(image_target):
        print("Aborted: Image file already exists")
        sys.exit(0)

    if not confirm_overwrite(weights_target):
        print("Aborted: Weights file already exists")
        sys.exit(0)

    if not confirm_overwrite(json_target):
        print("Aborted: JSON model file already exists")
        sys.exit(0)

    # Create directories if they don't exist
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(weights_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Create/convert image to JPG
    if args.noise_controlled:
        # Create hybrid image from multiple inputs
        print(f"Creating hybrid image from {len(args.images)} images...")
        try:
            img = create_hybrid_image(args.images)
            img.save(image_target, 'JPEG', quality=95)
            print(f"✓ Hybrid image saved as JPG")
            print(f"  Sources: {', '.join(args.images)}")
        except Exception as e:
            print(f"Error creating hybrid image: {e}")
            sys.exit(1)
    else:
        # Single image mode
        print(f"Converting image to {image_target}...")
        try:
            img = Image.open(args.image)
            # Convert to RGB if necessary (handles RGBA, grayscale, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')
            # Save as JPG with high quality
            img.save(image_target, 'JPEG', quality=95)
            print(f"✓ Image saved as JPG")
        except Exception as e:
            print(f"Error converting image: {e}")
            sys.exit(1)

    # Copy weights
    print(f"Copying weights to {weights_target}...")
    shutil.copy2(args.weights, weights_target)
    print(f"✓ Weights saved")

    # Convert weights to JSON
    print(f"Converting weights to JSON...")
    try:
        # Build command with optional flags
        convert_cmd = ['python3', convert_script, weights_target, json_target]

        # Object models (from nca_experiments.ipynb) use Growing NCA mode
        # with normalized Sobel filters and stochastic fire_rate=0.5
        if args.category == 'objects':
            convert_cmd.append('--growing-mode')
            print("  Adding --growing-mode flag for object model")

        result = subprocess.run(
            convert_cmd,
            check=True,
            capture_output=True,
            text=True
        )
        print(result.stdout)
        print(f"✓ JSON model saved to {json_target}")
    except subprocess.CalledProcessError as e:
        print(f"Error during conversion:")
        print(e.stderr)
        print("\nCleaning up...")
        # Clean up the files we created
        if os.path.exists(image_target):
            os.remove(image_target)
        if os.path.exists(weights_target):
            os.remove(weights_target)
        sys.exit(1)

    # Update metadata.json
    print(f"Updating metadata.json...")
    try:
        with open(metadata_file, 'r') as f:
            metadata = json.load(f)

        # Get the appropriate metadata key based on category
        metadata_key = VALID_CATEGORIES[args.category]

        # Check if model name already exists
        if args.model_name in metadata.get(metadata_key, []):
            print(f"  Model '{args.model_name}' already in {metadata_key}, skipping")
        else:
            # Add model name to the appropriate list
            if metadata_key not in metadata:
                metadata[metadata_key] = []
            metadata[metadata_key].append(args.model_name)

            # Write back to file
            with open(metadata_file, 'w') as f:
                json.dump(metadata, f, indent=2)

            print(f"✓ Added '{args.model_name}' to {metadata_key} in metadata.json")

    except Exception as e:
        print(f"Error updating metadata.json: {e}")
        print("You may need to manually add the model name to metadata.json")

    print(f"\n✓ Successfully added model '{args.model_name}'")
    print(f"  - Category: {args.category}")
    if args.noise_controlled:
        print(f"  - Mode: Noise-controlled ({len(args.images)} textures)")
        print(f"  - Source images: {', '.join(args.images)}")
        print(f"  - Hybrid image: {image_target}")
    else:
        print(f"  - Image: {image_target}")
    print(f"  - Weights: {weights_target}")
    print(f"  - JSON: {json_target}")
    print(f"  - Added to: {metadata_file}")
    if args.category == 'objects':
        print(f"\n  Note: Object models use center seed initialization. Select 'Center Seed' in the demo.")
    if args.noise_controlled:
        print(f"\n  Note: Noise-controlled model. Use the noise slider in the demo to switch between textures.")


if __name__ == '__main__':
    main()
