#!/usr/bin/env python3
"""
Add a trained model and its target image to the demo.

Usage:
    python3 add_model.py --weights weight_file.pt --image image.png --model_name my_model
    python3 add_model.py --weights weight_file.pt --image image.jpg --model_name my_model --model_type Noise-NCA
    python3 add_model.py --weights weight_file.pt --image image.png --model_name my_model --category objects

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


def main():
    parser = argparse.ArgumentParser(description='Add a trained model and image to the demo')
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to the .pt weights file')
    parser.add_argument('--image', type=str, required=True,
                        help='Path to the target image (.png or .jpg)')
    parser.add_argument('--model_name', type=str, required=True,
                        help='Name for the model (used for filenames)')
    parser.add_argument('--model_type', type=str, default='Noise-NCA',
                        help='Model type for folder organization (default: Noise-NCA)')
    parser.add_argument('--category', type=str, default='textures',
                        choices=list(VALID_CATEGORIES.keys()),
                        help='Model category: textures (uniform noise init) or objects (center seed init)')

    args = parser.parse_args()

    # Validate input files exist
    if not os.path.exists(args.weights):
        print(f"Error: Weights file '{args.weights}' not found")
        sys.exit(1)

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

    # Copy/convert image to JPG
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
    print(f"  - Image: {image_target}")
    print(f"  - Weights: {weights_target}")
    print(f"  - JSON: {json_target}")
    print(f"  - Added to: {metadata_file}")
    if args.category == 'objects':
        print(f"\n  Note: Object models use center seed initialization. Select 'Center Seed' in the demo.")


if __name__ == '__main__':
    main()
