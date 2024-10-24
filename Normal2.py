import os
import cv2
from ultralytics import YOLO
import pandas as pd

# Load YOLO model
model = YOLO("yolo11s-pose.pt")

# Video path
video_path = r'C:\Rupesh\OneDrive\Documents\charan_project\proglint2\normal2.mp4'

# Check if the video file exists
if not os.path.exists(video_path):
    print(f"Error: Video file not found at '{video_path}'. Please check the file path.")
    exit(1)

cap = cv2.VideoCapture(video_path)

# Check if video opened successfully
if not cap.isOpened():
    print(f"Error: Could not open video file '{video_path}'. The file might be corrupted or in an unsupported format.")
    exit(1)

# Get video properties
frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)

# Check if fps is valid
if fps <= 0:
    print(f"Error: Invalid FPS value ({fps}). The video file might be corrupted or not supported.")
    cap.release()
    exit(1)

seconds = round(frames / fps)
frame_total = min(1000, frames)  # Frames to process, capped at total frames

i, a = 0, 0  # Counters

# Define correct paths
base_dir = r'C:\Rupesh\OneDrive\Documents\charan_project\proglint2'
image_dir = os.path.join(base_dir, 'images')
cropped_dir = os.path.join(base_dir, 'images1')
csv_file_path = os.path.join(base_dir, 'nkeypoint.csv')

# Ensure directories exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

all_data = []

try:
    while cap.isOpened() and i < frame_total:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_MSEC, i * (seconds / frame_total) * 1000)
        flag, frame = cap.read()
        if not flag:
            print(f"Warning: Could not read frame {i}. Stopping processing.")
            break

        # Save full frame image
        image_path = os.path.join(image_dir, f'img_{i}.jpg')
        cv2.imwrite(image_path, frame)

        # Run YOLO detection
        results = model(frame, verbose=False)
        for r in results:
            bound_box = r.boxes.xyxy  # Bounding boxes
            conf = r.boxes.conf.tolist()  # Confidence scores
            keypoints = r.keypoints.xyn.tolist()  # Keypoints

            for index, box in enumerate(bound_box):
                if conf[index] > 0.75:  # Filter by confidence
                    x1, y1, x2, y2 = map(int, box.tolist())
                    # Handle cropping bounds
                    y1, y2 = max(0, y1), min(frame.shape[0], y2)
                    x1, x2 = max(0, x1), min(frame.shape[1], x2)
                    cropped_person = frame[y1:y2, x1:x2]
                    output_path = os.path.join(cropped_dir, f'person_nn_{a}.jpg')
                    cv2.imwrite(output_path, cropped_person)

                    # Collect keypoint data
                    data = {'image_name': f'person_nn_{a}.jpg'}
                    for j, (x, y) in enumerate(keypoints[index]):
                        data[f'x{j}'] = x
                        data[f'y{j}'] = y
                    all_data.append(data)
                    a += 1
        i += 1

    print(f"Total frames processed: {i}, Total cropped images saved: {a}")

except Exception as e:
    print(f"An error occurred during video processing: {str(e)}")

finally:
    cap.release()
    cv2.destroyAllWindows()

# Save keypoint data to CSV
try:
    if all_data:
        df = pd.DataFrame(all_data)
        df.to_csv(csv_file_path, index=False)
        print(f"Keypoint data saved to {csv_file_path}")
    else:
        print("No data to save. Check if the video file contains valid frames.")
except Exception as e:
    print(f"An error occurred while saving the CSV file: {str(e)}")