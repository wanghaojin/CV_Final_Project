import cv2
import os
import glob

# 视频的帧率（每秒帧数）
fps = 24
# 输出视频的分辨率
frame_width = 224
frame_height = 224

frame_pattern = "img*.jpg"

# frames_dir = '/home/sem/Videos/ReReVST-Code/MAE/mae_6457'
frames_dir = '../test/result_balabala'
# output_video_path = '/home/sem/Videos/ReReVST-Code/MAE/foxvideo_output/6457_fox.mp4'
output_video_path = '../test/result_balabala.mp4'
# 创建一个VideoWriter对象，使用MP4V编码器
out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

for filename in sorted(glob.glob(os.path.join(frames_dir, frame_pattern))):
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)

# 释放VideoWriter对象
out.release()

print("视频合并完成，已保存为:", output_video_path)