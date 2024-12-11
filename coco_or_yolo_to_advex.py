import os
import json
import shutil
import numpy as np
from PIL import Image
import cv2
from pycocotools import mask as coco_mask
from tqdm import tqdm

def convert_coco_to_advex(coco_json_path, images_dir, output_dir):
    """
    Convert COCO format annotations to Advex format.
    Handles both polygon and RLE segmentation formats.
    """
    # Create Advex directory structure
    advex_dir = os.path.join(output_dir, 'AdvexInputsFolder')
    images_output_dir = os.path.join(advex_dir, 'images')
    masks_output_dir = os.path.join(advex_dir, 'masks')
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    # Load COCO annotations
    with open(coco_json_path, 'r') as f:
        coco_data = json.load(f)
    
    # Create image id to annotations mapping
    img_to_anns = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in img_to_anns:
            img_to_anns[img_id] = []
        img_to_anns[img_id].append(ann)
    
    # Process each image with progress bar
    print("Converting COCO format to Advex format...")
    for img_info in tqdm(coco_data['images'], desc="Processing images", unit="img"):
        img_id = img_info['id']
        img_filename = img_info['file_name']
        
        # Copy original image
        src_img_path = os.path.join(images_dir, img_filename)
        if not os.path.exists(src_img_path):
            print(f"Warning: Image {img_filename} not found")
            continue
            
        # Read image to get dimensions
        img = Image.open(src_img_path)
        width, height = img.size
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Fill mask with annotations
        if img_id in img_to_anns:
            for ann in img_to_anns[img_id]:
                if 'segmentation' in ann:
                    if isinstance(ann['segmentation'], dict):  # RLE format
                        rle = ann['segmentation']
                        binary_mask = coco_mask.decode(rle)
                        mask = np.logical_or(mask, binary_mask).astype(np.uint8) * 255
                    else:  # Polygon format
                        for seg in ann['segmentation']:
                            pts = np.array(seg).reshape((-1, 2)).astype(np.int32)
                            cv2.fillPoly(mask, [pts], 255)
                elif 'bbox' in ann:  # Fallback to bbox if no segmentation
                    x, y, w, h = map(int, ann['bbox'])
                    cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
        
        # Save processed image and mask
        dst_img_path = os.path.join(images_output_dir, f'img_{img_id}.png')
        dst_mask_path = os.path.join(masks_output_dir, f'img_{img_id}.png')
        
        # Convert and save image as PNG
        img = img.convert('RGB')
        img.save(dst_img_path, 'PNG')
        
        # Save mask
        mask_img = Image.fromarray(mask)
        mask_img.save(dst_mask_path)

def convert_yolo_to_advex(yolo_dir, output_dir):
    """
    Convert YOLO format annotations to Advex format.
    Reads actual image dimensions instead of assuming fixed size.
    """
    # Create Advex directory structure
    advex_dir = os.path.join(output_dir, 'AdvexInputsFolder')
    images_output_dir = os.path.join(advex_dir, 'images')
    masks_output_dir = os.path.join(advex_dir, 'masks')
    
    os.makedirs(images_output_dir, exist_ok=True)
    os.makedirs(masks_output_dir, exist_ok=True)
    
    images_dir = os.path.join(yolo_dir, 'images')
    labels_dir = os.path.join(yolo_dir, 'labels')
    
    if not os.path.exists(labels_dir):
        print(f"Warning: Labels directory not found at {labels_dir}")
        return
    
    # Get list of valid images
    valid_images = [f for f in os.listdir(images_dir) 
                   if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process each image and its label file with progress bar
    print("Converting YOLO format to Advex format...")
    for img_file in tqdm(valid_images, desc="Processing images", unit="img"):
        # Get corresponding label file
        base_name = os.path.splitext(img_file)[0]
        label_file = os.path.join(labels_dir, f'{base_name}.txt')
        
        if not os.path.exists(label_file):
            print(f"Warning: No label file found for {img_file}")
            continue
        
        # Read image to get dimensions
        img_path = os.path.join(images_dir, img_file)
        try:
            img = Image.open(img_path)
            width, height = img.size
        except Exception as e:
            print(f"Error reading image {img_file}: {e}")
            continue
        
        # Create mask
        mask = np.zeros((height, width), dtype=np.uint8)
        
        # Read YOLO format labels and convert to mask
        try:
            with open(label_file, 'r') as f:
                for line in f:
                    try:
                        class_id, x_center, y_center, w, h = map(float, line.strip().split())
                        
                        # Convert normalized coordinates to pixel coordinates
                        x_center = int(x_center * width)
                        y_center = int(y_center * height)
                        w = int(w * width)
                        h = int(h * height)
                        
                        # Calculate bounding box coordinates
                        x1 = max(0, int(x_center - w/2))
                        y1 = max(0, int(y_center - h/2))
                        x2 = min(width, int(x_center + w/2))
                        y2 = min(height, int(y_center + h/2))
                        
                        # Draw filled rectangle on mask
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                    except ValueError as e:
                        print(f"Error parsing line in {label_file}: {e}")
                        continue
        except Exception as e:
            print(f"Error reading label file {label_file}: {e}")
            continue
        
        # Save processed image and mask
        dst_img_path = os.path.join(images_output_dir, f'{base_name}.png')
        dst_mask_path = os.path.join(masks_output_dir, f'{base_name}.png')
        
        # Convert and save image as PNG
        img = img.convert('RGB')
        img.save(dst_img_path, 'PNG')
        
        # Save mask
        mask_img = Image.fromarray(mask)
        mask_img.save(dst_mask_path)

def verify_advex_structure(input_dir, show_progress=True):
    """
    Verify that the directory structure and files meet Advex requirements.
    """
    # Check main directory structure
    images_dir = os.path.join(input_dir, 'images')
    masks_dir = os.path.join(input_dir, 'masks')
    
    if not os.path.exists(images_dir) or not os.path.exists(masks_dir):
        return False, "Missing 'images' or 'masks' directory"
        
    # Get list of files
    image_files = set(os.listdir(images_dir))
    mask_files = set(os.listdir(masks_dir))
    
    if not image_files:
        return False, "No images found in images directory"
    
    # Check number of files
    if len(image_files) < 10:
        return False, f"Not enough images. Found {len(image_files)}, need at least 10"
    
    # Check matching files
    if image_files != mask_files:
        return False, "Mismatch between image and mask files"
    
    # Verify each image and mask with progress bar
    for img_file in (tqdm(image_files, desc="Verifying files", unit="file") if show_progress else image_files):
        # Check image format
        if not img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            return False, f"Invalid image format for {img_file}"
            
        try:
            # Verify image can be opened
            img_path = os.path.join(images_dir, img_file)
            mask_path = os.path.join(masks_dir, img_file)
            
            img = Image.open(img_path)
            mask = Image.open(mask_path)
            
            # Verify mask is grayscale
            if mask.mode != 'L':
                return False, f"Mask {img_file} is not grayscale"
                
            # Verify image and mask dimensions match
            if img.size != mask.size:
                return False, f"Size mismatch between image and mask for {img_file}"
                
            # Verify mask values are only 0 and 255
            mask_array = np.array(mask)
            unique_values = np.unique(mask_array)
            if not np.array_equal(unique_values, [0]) and not np.array_equal(unique_values, [0, 255]) and not np.array_equal(unique_values, [255]):
                return False, f"Mask {img_file} contains invalid values (should only be 0 and 255)"
                
        except Exception as e:
            return False, f"Error processing {img_file}: {str(e)}"
    
    return True, "Directory structure is valid"

if __name__ == "__main__":
    import argparse
    import time
    
    parser = argparse.ArgumentParser(description='Convert COCO/YOLO format to Advex format')
    parser.add_argument('--format', choices=['coco', 'yolo'], required=True,
                      help='Input format (COCO or YOLO)')
    parser.add_argument('--input-dir', required=True,
                      help='Input directory containing the original format data')
    parser.add_argument('--output-dir', required=True,
                      help='Output directory for Advex format data')
    parser.add_argument('--coco-json', 
                      help='Path to COCO json file (required for COCO format)')
    
    args = parser.parse_args()
    
    try:
        start_time = time.time()
        print(f"\nStarting conversion from {args.format.upper()} format to Advex format...")
        
        if args.format == 'coco':
            if not args.coco_json:
                parser.error("--coco-json is required when format is 'coco'")
            convert_coco_to_advex(args.coco_json, args.input_dir, args.output_dir)
        else:  # YOLO
            convert_yolo_to_advex(args.input_dir, args.output_dir)
        
        # Verify the output
        is_valid, message = verify_advex_structure(
            os.path.join(args.output_dir, 'AdvexInputsFolder'))
            
        elapsed_time = time.time() - start_time
        if is_valid:
            print(f"\nConversion successful: {message}")
            print(f"Total time elapsed: {elapsed_time:.2f} seconds")
        else:
            print(f"\nConversion produced invalid output: {message}")
            
    except Exception as e:
        print(f"Error during conversion: {str(e)}")
