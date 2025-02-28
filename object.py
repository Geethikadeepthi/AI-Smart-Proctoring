import cv2
import numpy as np
from ultralytics import YOLO
import ctypes
import logging

# Configure logging
logging.basicConfig(filename="proctoring.log", level=logging.INFO, format="%(asctime)s - %(message)s")

# Function to display a message box
def show_message_box(title, message):
    ctypes.windll.user32.MessageBoxW(0, message, title, 0x40 | 0x1000)

# Load the YOLOv8 model (CPU-only)
model = YOLO("yolov8s.pt")  # Use a smaller model for faster CPU inference

# Define prohibited objects to monitor
PROHIBITED_OBJECTS = {"cell phone", "book", "laptop", "earbuds"}

# Flag to control exam termination
exam_terminated = False

# Function to terminate the exam
def terminate_exam(reason):
    global exam_terminated
    logging.warning(f"Exam terminated: {reason}")
    show_message_box("Exam Terminated", reason)
    exam_terminated = True

# Function to log detected events
def log_event(event):
    logging.info(event)

# Open the video feed from the proctoring window
video_capture = cv2.VideoCapture(0)  # Use 0 for webcam or video file path

# Main loop
while not exam_terminated:
    try:
        # Read a frame from the video feed
        ret, frame = video_capture.read()
        if not ret:
            logging.error("Failed to capture video feed.")
            break

        # Resize the frame for faster processing
        frame = cv2.resize(frame, (640, 480))  # Resize to 640x480

        # Run YOLOv8 inference
        results = model(frame, conf=0.6)  # Set confidence threshold to 0.7

        # Variables to track detections
        face_count = 0
        prohibited_objects_detected = []

        # Process YOLOv8 results
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                confidence = float(box.conf[0])
                label = result.names[int(box.cls[0])]

                # Check for prohibited objects
                if label in PROHIBITED_OBJECTS and confidence > 0.7:
                    prohibited_objects_detected.append(label)

                    # Log the detected object
                    log_event(f"Prohibited object detected: {label} (Confidence: {confidence:.2f})")

                    # Draw bounding box (optional, for debugging)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    cv2.putText(frame, f"{label} ({confidence:.2f})", (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

                # Count faces
                if label == "person" and confidence > 0.7:
                    face_count += 1

        # Log the number of faces detected
        log_event(f"Faces detected: {face_count}")

        # Terminate exam if multiple faces are detected
        if face_count > 1:
            terminate_exam("Multiple faces detected.")

        # Terminate exam if prohibited objects are detected
        if prohibited_objects_detected:
            terminate_exam(f"Prohibited objects detected: {', '.join(prohibited_objects_detected)}")

        # Display the proctoring window with bounding boxes
        cv2.imshow("Proctoring Window", frame)

        # Add a small delay to reduce CPU usage
        key = cv2.waitKey(1)
        if key == 27:  # Press 'Esc' to exit
            break

    except Exception as e:
        logging.error(f"Error in main loop: {e}")
        break

# Release resources
video_capture.release()
cv2.destroyAllWindows()