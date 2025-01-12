import os
import json
import random
import numpy as np
import math
import re
import cv2
from PIL import Image

# --- 图像分割 ---
def img2p(image_path, patch_size, mask_array):
    ori_img = cv2.imread(image_path)
    img_resized = cv2.resize(ori_img, (224, 224), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    img_width, img_height = img.size
    patch_width, patch_height = patch_size

    patches = []

    for top in range(0, img_height, patch_height):
        for left in range(0, img_width, patch_width):
            right = min(left + patch_width, img_width)
            bottom = min(top + patch_height, img_height)

            patch = img.crop((left, top, right, bottom))
            patches.append(patch)

    return patches, mask_array

# --- patches拼接 ---
def p2img(patches, mask_array, patch_size, img_size, output_image_path):
    patch_width, patch_height = patch_size
    img_width, img_height = img_size

    result_img = Image.new('RGB', (img_width, img_height))

    for top in range(0, img_height, patch_height):
        for left in range(0, img_width, patch_width):
            patch = patches[top // patch_height * (img_width // patch_width) + left // patch_width]
            
            # 检查是否被mask
            if mask_array[top // patch_height][left // patch_width] == 1:
                patch = Image.new('RGB', patch.size, (150, 150, 150))

            result_img.paste(patch, (left, top))

    result_img.save(output_image_path, 'JPEG')

def save_mask_array(mask_array, output_mask_path):
    with open(output_mask_path, 'w') as f:
        json.dump(mask_array, f)

def stitch(patches, patch_size, output_image_path):

    patch_width, patch_height = patch_size
    num_patches = len(patches)
    
    # 计算需要多少行和列
    num_cols = int(math.sqrt(num_patches))  # 排列成正方形
    num_rows = num_patches // num_cols + (1 if num_patches % num_cols else 0)

    stitched_img = Image.new('RGB', (num_cols * patch_width, num_rows * patch_height))

    for idx, patch in enumerate(patches):
        col = idx % num_cols
        row = idx // num_cols
        stitched_img.paste(patch, (col * patch_width, row * patch_height))

    stitched_img.save(output_image_path)

# --- 生成固定的mask数组 ---
def generate_fixed_mask_array(img_size, patch_size, num_unmasked):
    img_width, img_height = img_size
    patch_width, patch_height = patch_size

    num_patches = (img_width // patch_width) * (img_height // patch_height)
    mask_array = [[1] * (img_width // patch_width) for _ in range(img_height // patch_height)]

    # 保留中间64个patches
    for row in range(2, 10):
        for col in range(2, 10):
            mask_array[row][col] = 0

    # 从mask_array中随机选择num_unmasked个设置为0
    indices_to_unmask = random.sample([i for i in range(num_patches) if mask_array[i // (img_height // patch_height)][i % (img_width // patch_width)] == 1], num_unmasked)
    for idx in indices_to_unmask:
        mask_array[idx // (img_height // patch_height)][idx % (img_width // patch_width)] = 0

    return mask_array

# --- 处理图像 ---
def process_images(input_dir, output_dir, mask_array_dir, patch_size=(16, 16), stitch_dir=None):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if not os.path.exists(mask_array_dir):
        os.makedirs(mask_array_dir)

    if stitch_dir and not os.path.exists(stitch_dir):
        os.makedirs(stitch_dir)
        
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 生成固定的mask数组
    fixed_mask_array = generate_fixed_mask_array((224, 224), patch_size, 57)

    for image_file in image_files:
        image_path = os.path.join(input_dir, image_file)

        patches, mask_array = img2p(image_path, patch_size, fixed_mask_array)
        output_image_path = os.path.join(output_dir, f"{image_file}")
        p2img(patches, mask_array, patch_size, img_size=(224, 224), output_image_path=output_image_path)
        
        img_name_without_ext = os.path.splitext(image_file)[0]
        mask_array_file = os.path.join(mask_array_dir, f"{img_name_without_ext}.json")
        save_mask_array(mask_array, mask_array_file)

        non_mask_patches = [patch for patch, mask in zip(patches, [item for sublist in mask_array for item in sublist]) if mask == 0]

        if stitch_dir:
            stitch_output_image_path = os.path.join(stitch_dir, f"{img_name_without_ext}.jpg")
            stitch(non_mask_patches, patch_size, stitch_output_image_path)

input_dir = "/home/sem/Videos/ReReVST-Code/MAE/input_fox"  
output_dir = "/home/sem/Videos/ReReVST-Code/MAE/mask_6457"  
mask_array_dir = "/home/sem/Videos/ReReVST-Code/MAE/mask_6457_array" 
stitch_dir = "/home/sem/Videos/ReReVST-Code/MAE/stitch_6457" 

process_images(input_dir, output_dir, mask_array_dir, patch_size=(16, 16), stitch_dir=stitch_dir)
print("Imgs are all divided to patches and are stitched.")