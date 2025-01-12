import cv2
import os
import glob

fps = 24
# Output video resolution
frame_width = 224
frame_height = 224

frame_pattern = "img*.jpg"

frames_dir = ''
output_video_path = ''

out = cv2.VideoWriter(output_video_path, cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_width, frame_height))

for filename in sorted(glob.glob(os.path.join(frames_dir, frame_pattern))):
    frame = cv2.imread(filename)
    frame = cv2.resize(frame, (frame_width, frame_height))
    out.write(frame)

out.release()

print("video saved as:", output_video_path)
