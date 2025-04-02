import cv2
import torch
import time
from ultralytics import YOLO
import pygame

# Load the trained YOLO model from local storage
model_path = "#A:\\path\\to\\your\\model"  
model = YOLO(model_path)

# Initialize pygame mixer for sound
pygame.mixer.init()
sound_path = "#A:\\path\\to\\your\\audio"

# Define the monkey class, 0 = F-monkey, 1 = N-monkey
MONKEY_CLASSES = [0, 1] 

# Prompt user to choose input mode
print("Choose input mode:")
print("0 - Enter video file path")
print("1 - Use webcam for real-time inference")

choice = input("Enter 0 or 1: ")

if choice == "0":
    video_path = input("Enter the path to the video file: ")  
    video = cv2.VideoCapture(video_path)
elif choice == "1":
    video = cv2.VideoCapture(0)  # Open webcam
else:
    print("Invalid choice. Exiting.")
    exit()

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

# Get video properties
fps = int(video.get(cv2.CAP_PROP_FPS)) 
frame_width = int(video.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))
resize_dim = (640, 360) if frame_width > 640 else (frame_width, frame_height)

# Process each frame
frame_count = 0 

# Run inference on the frame
while True:
    ret, frame = video.read()
    if not ret:
        break  

    frame_count += 1
    if frame_count % 3 != 0: 
        continue
    frame = cv2.resize(frame, resize_dim)  

    
    results = model(frame)

    monkey_detected = False  

    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])  # Get the detected class ID
            confidence = box.conf[0].item()

            # Play sound if either F-monkey or N-monkey is detected
            if class_id in MONKEY_CLASSES and confidence > 0.25:  
                monkey_detected = True

                # Play sound only if not already playing
                if not pygame.mixer.music.get_busy():
                    pygame.mixer.music.load(sound_path)
                    pygame.mixer.music.play()

                # Draw bounding box and label
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, f"Monkey {confidence:.2f}", (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # Display the frame with detection
    cv2.imshow("Monkey Detection", frame)

   
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


video.release()
cv2.destroyAllWindows()

print("Inference complete. Video closed.")

#if you want to use video in local, use this format
#A:\\path\\to\\your\\video