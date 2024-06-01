import os
import sys
import numpy as np
from PIL import Image
from scipy.ndimage import label

def extract_regions_with_corrected_center(image_path, output_folder):
    image = Image.open(image_path).convert("RGBA")
    base_name = os.path.basename(image_path)
    image_output_folder = os.path.join(output_folder, os.path.splitext(base_name)[0])
    if not os.path.exists(image_output_folder):
        os.makedirs(image_output_folder)
    
    mask = image.split()[3].point(lambda p: p > 0 and 255)
    mask_np = np.array(mask)
    labeled, num_features = label(mask_np)
    
    regions = []
    for region_idx in range(1, num_features + 1):
        region_mask = (labeled == region_idx).astype(np.uint8) * 255
        region_image = Image.fromarray(region_mask, mode='L')
        bbox = region_image.getbbox()
        if bbox:
            center = ((bbox[0] + bbox[2]) // 2, (bbox[1] + bbox[3]) // 2)
            regions.append((region_idx, bbox, center))
    
    image_center = (image.width // 2, image.height // 2)
    central_region = min(regions, key=lambda r: (r[2][0] - image_center[0])**2 + (r[2][1] - image_center[1])**2)
    central_center = central_region[2]
    
    smallest_central_region = min(
        [r for r in regions if (r[2][0] - central_center[0], r[2][1] - central_center[1]) == (0, 0)],
        key=lambda r: (r[1][2] - r[1][0]) * (r[1][3] - r[1][1])
    )
    
    for region_idx, bbox, center in regions:
        region = image.crop(bbox)
        region_mask = (labeled == region_idx).astype(np.uint8) * 255
        region_image = Image.fromarray(region_mask, mode='L')
        region_mask_cropped = region_image.crop(bbox)
        region.putalpha(region_mask_cropped)
        
        relative_coords = (center[0] - central_center[0], center[1] - central_center[1])
        filename = f"region_{region_idx}_{relative_coords[0]}x{relative_coords[1]}"
        if relative_coords == (0, 0):
            if region_idx == smallest_central_region[0]:
                filename += "_center"
        filename += ".png"
        
        region.save(os.path.join(image_output_folder, filename))

def process_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for file_name in os.listdir(input_folder):
        if file_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
            image_path = os.path.join(input_folder, file_name)
            extract_regions_with_corrected_center(image_path, output_folder)

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python extract_regions.py <input_folder> <output_folder>")
        sys.exit(1)
    
    input_folder = sys.argv[1]
    output_folder = sys.argv[2]
    process_folder(input_folder, output_folder)
