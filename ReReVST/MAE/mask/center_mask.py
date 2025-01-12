import os
import json
import random
import numpy as np
import math
import re
import cv2
from PIL import Image


"""
将图像分割成patch并mask后再拼接好,是否mask(bool)存为数组.json,将未mask的有效patches保存,并在最后拼接未mask的patches生成图像
"""

# --- 图像分割 ---

def img2p(image_path, patch_size):
    ori_img = cv2.imread(image_path)
    img_resized = cv2.resize(ori_img, (224,224), interpolation=cv2.INTER_AREA)
    img = Image.fromarray(cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB))
    
    img_width, img_height = img.size
    patch_width, patch_height = patch_size

    patches = []
    mask_array = []

    patch_count = 0
    for top in range(0, img_height, patch_height):
        row_mask = []  
        for left in range(0, img_width, patch_width):
            right = min(left + patch_width, img_width)
            bottom = min(top + patch_height, img_height)

            patch = img.crop((left, top, right, bottom))
            
            # 新的mask逻辑，只保留中间49个patches
            row, col = top // patch_height, left // patch_width
            if (row >= 3 and row <= 9) and (col >= 3 and col <= 9):  # 保留中间7x7的patches
                row_mask.append(0)  # no mask
            else:
                row_mask.append(1)  # mask

            patches.append(patch)
            patch_count += 1
        
        mask_array.append(row_mask)

    return patches, mask_array

# --- patches拼接 ---
def p2img(patches, mask_array, patch_size, img_size, output_image_path):
    patch_width, patch_height = patch_size
    img_width, img_height = img_size

    result_img = Image.new('RGB', (img_width, img_height))

    patch_count = 0
    for top in range(0, img_height, patch_height):
        for left in range(0, img_width, patch_width):
            patch = patches[patch_count]
            
            # 检查是否被mask
            row = top // patch_height
            col = left // patch_width
            if mask_array[row][col] == 1:
                patch = Image.new('RGB', patch.size, (150, 150, 150))

            result_img.paste(patch, (left, top))
            patch_count += 1

    result_img.save(output_image_path, 'JPEG')

def save_mask_array(mask_array, output_mask_path):
    #for row in mask_array:
        #print(row)
    with open(output_mask_path, 'w') as f:
        json.dump(mask_array, f)


# --- stitch ---

def extract_number(filename):
    match = re.search(r'(\d+)', filename)
    return int(match.group(1)) if match else 0

def sort_patches(patch_files):
    return sorted(patch_files, key=extract_number)

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

# --- ---

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
        #print(mask_array)
        # save mask
        output_image_path = os.path.join(output_dir, f"{image_file}")
        p2img(patches, mask_array, patch_size, img_size=(224, 224), output_image_path=output_image_path)
        
        # save mask_array
        img_name_without_ext = os.path.splitext(image_file)[0]  # 去掉文件扩展名
        mask_array_file = os.path.join(mask_array_dir, f"{img_name_without_ext}.json")
        save_mask_array(mask_array, mask_array_file)

        # non_mask_patches
        non_mask_patches = [patch for patch, mask in zip(patches, [item for sublist in mask_array for item in sublist]) if mask == 0]

        # stitch
        if stitch_dir:
            stitch_output_image_path = os.path.join(stitch_dir, f"{img_name_without_ext}.jpg")
            stitch(non_mask_patches, patch_size, stitch_output_image_path)
            

input_dir = "/home/sem/Videos/ReReVST-Code/MAE/input_fox"  
output_dir = "/home/sem/Videos/ReReVST-Code/MAE/mask_fc49"  
mask_array_dir = "/home/sem/Videos/ReReVST-Code/MAE/mask_fc49_array" 
stitch_dir = "/home/sem/Videos/ReReVST-Code/MAE/stitch_fc49"

process_images(input_dir, output_dir, mask_array_dir, patch_size=(16, 16), stitch_dir=stitch_dir)
print("Imgs are all divided to patches and are stitched.")
