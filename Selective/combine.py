import os
import numpy as np
import cv2
from PIL import Image

video_dir = "fox/video"
result_dir = "fox/transferred"
mask_path = "fox/video_array.npy"
output_dir = "fox/output"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mask = np.load(mask_path)

frame_names = [
    p for p in os.listdir(video_dir)
    if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
]
frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

output_images = []
for i, frame_name in enumerate(frame_names):
    A_path = os.path.join(video_dir, frame_name)
    B_path = os.path.join(result_dir, frame_name)
    
    A = np.array(Image.open(A_path))
    B = np.array(Image.open(B_path))
    
    A_resized = cv2.resize(A, (B.shape[1], B.shape[0]))
    mask_resized = cv2.resize(mask[i].astype(np.uint8), (B.shape[1], B.shape[0]))
    mask_resized = mask_resized.astype(bool)
    
    C = mask_resized[..., np.newaxis] * B + (1 - mask_resized[..., np.newaxis]) * A_resized
    
    output_image_path = os.path.join(output_dir, f"{i:04d}.jpg")
    Image.fromarray(C.astype(np.uint8)).save(output_image_path)
    output_images.append(output_image_path)

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_out = cv2.VideoWriter('output_video.mp4', fourcc, 30, (B.shape[1], B.shape[0]))

for img_path in output_images:
    img = cv2.imread(img_path)
    video_out.write(img)

video_out.release()
print("视频生成完毕")
