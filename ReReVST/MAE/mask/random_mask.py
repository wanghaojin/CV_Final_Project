import os
import json
import random
import numpy as np
import math
import re
import cv2
from PIL import Image
"""
Split the image into patches and mask pacthes randomly (remain 100 pathces unmask)
Save the mask (boolean) as an array
Save the effective unmasked patches and finally stitch the unmasked patches to generate an image
"""
# --- img to patches ---
def img2p(image_path, patch_size):
    ori_img = cv2.imread(image_path)
    img_resized = cv2.resize(ori_img, (224, 224), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    img_width, img_height = img.size
    patch_width, patch_height = patch_size

    patches = []
    mask_array = []

    for top in range(0, img_height, patch_height):
        row_mask = []  
        for left in range(0, img_width, patch_width):
            right = min(left + patch_width, img_width)
            bottom = min(top + patch_height, img_height)

            patch = img.crop((left, top, right, bottom))
            patches.append(patch)

            # 初始化所有patches为mask
            row_mask.append(1)

        mask_array.append(row_mask)

    # randomly choose 100 patches unmask, can change
    total_patches = len(patches)
    indices_to_unmask = random.sample(range(total_patches), min(total_patches, 100))
    for idx in indices_to_unmask:
        mask_array[idx // (img_height // patch_height)][idx % (img_width // patch_width)] = 0

    return patches, mask_array

# --- patches to img ---
def p2img(patches, mask_array, patch_size, img_size, output_image_path):
    patch_width, patch_height = patch_size
    img_width, img_height = img_size

    result_img = Image.new('RGB', (img_width, img_height))

    for top in range(0, img_height, patch_height):
        for left in range(0, img_width, patch_width):
            patch = patches[top // patch_height * (img_width // patch_width) + left // patch_width]
            
            if mask_array[top // patch_height][left // patch_width] == 1:
                patch = Image.new('RGB', patch.size, (150, 150, 150))

            result_img.paste(patch, (left, top))

    result_img.save(output_image_path, 'JPEG')

def save_mask_array(mask_array, output_mask_path):
    with open(output_mask_path, 'w') as f:
        json.dump(mask_array, f)

# --- stitch ---
def stitch(patches, patch_size, output_image_path):
    patch_width, patch_height = patch_size
    num_patches = len(patches)
    
    # 确保stitch的patches数为10*10
    num_cols = 10
    num_rows = 10

    stitched_img = Image.new('RGB', (num_cols * patch_width, num_rows * patch_height))

    for idx, patch in enumerate(patches):
        col = idx % num_cols
        row = idx // num_cols
        stitched_img.paste(patch, (col * patch_width, row * patch_height))

    stitched_img.save(output_image_path)

# --- process ---
def process_images(input_dir, output_dir, mask_array_dir, patch_size=(16, 16), stitch_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(mask_array_dir):
        os.makedirs(mask_array_dir)

    if stitch_dir and not os.path.exists(stitch_dir):
        os.makedirs(stitch_dir)
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        patches, mask_array = img2p(image_path, patch_size)
        output_image_path = os.path.join(output_dir, f"{image_file}")
        p2img(patches, mask_array, patch_size, img_size=(224, 224), output_image_path=output_image_path)
        
        img_name_without_ext = os.path.splitext(image_file)[0]
        mask_array_file = os.path.join(mask_array_dir, f"{img_name_without_ext}.json")
        save_mask_array(mask_array, mask_array_file)

        non_mask_patches = [patch for patch, mask in zip(patches, [item for sublist in mask_array for item in sublist]) if mask == 0]

        if stitch_dir:
            stitch_output_image_path = os.path.join(stitch_dir, f"{img_name_without_ext}.jpg")
            stitch(non_mask_patches, patch_size, stitch_output_image_path)

input_dir = ""  
output_dir = ""  
mask_array_dir = "" 
stitch_dir = ""

process_images(input_dir, output_dir, mask_array_dir, patch_size=(16, 16), stitch_dir=stitch_dir)
