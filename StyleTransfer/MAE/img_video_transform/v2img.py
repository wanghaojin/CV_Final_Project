import cv2
import os

video_path = 'fox.mp4'
output_folder = 'input_fox'

if not os.path.exists(output_folder):
    os.makedirs(output_folder)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_count = 0  

while True:
    ret, frame = cap.read() 

    if not ret:
        print("Can't receive frame (stream end?). Exiting ...")
        break
    
    resized_frame = cv2.resize(frame, (448, 448), interpolation=cv2.INTER_AREA)
    
    frame_path = os.path.join(output_folder, f'img{frame_count:03d}.jpg')
    
    cv2.imwrite(frame_path, resized_frame)
    
    frame_count += 1  

cap.release()
print(f"Video has been converted into {frame_count} frames and saved to {output_folder} folder.")