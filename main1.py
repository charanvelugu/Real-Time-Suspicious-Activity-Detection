import cv2
import os
import pandas as pd
from ultralytics import YOLO
import xgboost as xgb
import numpy as np
import cvzone
import pygame
import time

# Initialize Pygame for audio playback
pygame.mixer.init()

# Load the sound file (path to your sound file)
suspicious_sound = r"C:\Rupesh\OneDrive\Documents\charan_project\yolo11_suspicious_activity-main\fx-police(chosic.com).mp3"

def detect_shoplifting(video_path):
    # Load YOLOv8 model
    model_yolo = YOLO('yolo11s-pose.pt')

    # Load the trained XGBoost model
    model = xgb.Booster()
    model.load_model(r'C:\Rupesh\OneDrive\Documents\charan_project\proglint2\trained_model.json')

    # Open the video
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    print(f"Total Frames: {cap.get(cv2.CAP_PROP_FRAME_COUNT)}")

    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    frame_tot = 0
    count = 0
    
    # Variables for audio control
    audio_duration = 4  # Duration to play audio in seconds
    last_audio_start = 0
    audio_playing = False
    suspicious_detected = False

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("Warning: Frame could not be read. Skipping.")
            break

        count += 1
        if count % 3 != 0:
            continue

        # Resize the frame
        frame = cv2.resize(frame, (1200, 700))

        # Run YOLOv8 on the frame
        results = model_yolo(frame, verbose=False)

        # Visualize the YOLO results on the frame
        annotated_frame = results[0].plot(boxes=False)

        # Reset suspicious flag for current frame
        frame_suspicious = False

        current_time = time.time()

        for r in results:
            bound_box = r.boxes.xyxy
            conf = r.boxes.conf.tolist()
            keypoints = r.keypoints.xyn.tolist()

            print(f'Frame {frame_tot}: Detected {len(bound_box)} bounding boxes')

            for index, box in enumerate(bound_box):
                if conf[index] > 0.55:
                    x1, y1, x2, y2 = box.tolist()

                    if index < len(keypoints):
                        # Prepare data for XGBoost prediction
                        data = {}
                        for j in range(len(keypoints[index])):
                            data[f'x{j}'] = keypoints[index][j][0]
                            data[f'y{j}'] = keypoints[index][j][1]

                        df = pd.DataFrame(data, index=[0])
                        dmatrix = xgb.DMatrix(df)
                        sus = model.predict(dmatrix)
                        binary_predictions = (sus > 0.5).astype(int)
                        print(f'Prediction: {binary_predictions}')

                        if binary_predictions == 0:  # Suspicious
                            frame_suspicious = True
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 5)
                            cvzone.putTextRect(annotated_frame, f"Suspicious", (int(x1), int(y1) + 50), 1, 3)
                        else:  # Normal
                            cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                            cvzone.putTextRect(annotated_frame, f"Normal", (int(x1), int(y1) + 50), 1, 1)

        # Audio control logic
        if frame_suspicious and not audio_playing:
            # Start new audio alert
            pygame.mixer.music.load(suspicious_sound)
            pygame.mixer.music.play(0)  # Play once
            audio_playing = True
            last_audio_start = current_time
            suspicious_detected = True
        
        # Check if we need to stop the audio
        if audio_playing and (current_time - last_audio_start >= audio_duration):
            pygame.mixer.music.stop()
            audio_playing = False
            suspicious_detected = False

        # Show the annotated frame in a window
        cv2.imshow('Frame', annotated_frame)

        # Press 'q' to stop the video early
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    pygame.mixer.music.stop()
    pygame.mixer.quit()

# Call the function with the video path
detect_shoplifting("C:\Rupesh\OneDrive\Documents\charan_project\proglint2\WhatsApp Video 2024-10-24 at 12.01.17 PM.mp4")