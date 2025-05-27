"""
Face Detection Test - Camera Only
Test script for face detection without robot communication
"""

import cv2
import numpy as np
import imutils
import time

# Load the face detection model
print("Loading face detection model...")
try:
    pretrained_model = cv2.dnn.readNetFromCaffe("MODELS/deploy.prototxt.txt", "MODELS/res10_300x300_ssd_iter_140000.caffemodel")
    print("Model loaded successfully")
except Exception as e:
    print(f"Failed to load model: {e}")
    print("Make sure the MODELS directory exists with the required files")
    exit(1)

video_resolution = (700, 400)
video_midpoint = (int(video_resolution[0]/2), int(video_resolution[1]/2))

def find_faces_dnn(image):
    """Find faces using DNN model"""
    frame = image.copy()
    frame = imutils.resize(frame, width=video_resolution[0])

    # Get frame dimensions and convert to blob
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
                                 (300, 300), (104.0, 177.0, 123.0))

    # Pass through network
    pretrained_model.setInput(blob)
    detections = pretrained_model.forward()
    
    face_centers = []
    
    # Process detections
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        
        if confidence < 0.4:
            continue
            
        # Get bounding box coordinates
        box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
        (startX, startY, endX, endY) = box.astype("int")
        
        # Calculate face center
        face_center = (int(startX + (endX - startX) / 2), int(startY + (endY - startY) / 2))
        position_from_center = (face_center[0] - video_midpoint[0], face_center[1] - video_midpoint[1])
        face_centers.append(position_from_center)
        
        # Draw detection
        text = "{:.2f}%".format(confidence * 100)
        y = startY - 10 if startY - 10 > 10 else startY + 10
        
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 2)
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 2)
        cv2.line(frame, video_midpoint, face_center, (0, 200, 0), 5)
        cv2.circle(frame, face_center, 4, (0, 200, 0), 3)
    
    # Draw center crosshair
    cv2.line(frame, (video_midpoint[0]-20, video_midpoint[1]), (video_midpoint[0]+20, video_midpoint[1]), (255, 255, 255), 2)
    cv2.line(frame, (video_midpoint[0], video_midpoint[1]-20), (video_midpoint[0], video_midpoint[1]+20), (255, 255, 255), 2)
    
    return face_centers, frame

def main():
    print("Initializing camera...")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera")
        return
    
    # Set camera properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, video_resolution[0])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, video_resolution[1])
    cap.set(cv2.CAP_PROP_FPS, 13)
    
    print("Camera initialized successfully")
    print("Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    fps_timer = time.time()
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Process frame
            face_positions, processed_frame = find_faces_dnn(frame)
            
            # Add info text
            info_text = f"Faces detected: {len(face_positions)}"
            cv2.putText(processed_frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Calculate and display FPS
            frame_count += 1
            if frame_count % 30 == 0:
                current_time = time.time()
                fps = 30 / (current_time - fps_timer)
                fps_timer = current_time
                print(f"FPS: {fps:.1f}, Faces: {len(face_positions)}")
            
            # Show frame
            cv2.imshow('Face Detection Test', processed_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"face_detection_frame_{frame_count}.jpg"
                cv2.imwrite(filename, processed_frame)
                print(f"Saved frame as {filename}")
    
    except KeyboardInterrupt:
        print("\nStopping...")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")

if __name__ == "__main__":
    main() 