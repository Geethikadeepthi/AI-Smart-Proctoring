# from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# import csv
# import os
# import subprocess
# from werkzeug.utils import secure_filename
# import threading
# import time
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import mediapipe as mp
# from collections import deque
# from noise import NoiseDetector  # Import the NoiseDetector class
# import logging

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Ensure the data directory exists
# if not os.path.exists('data'):
#     os.makedirs('data')

# # Initialize YOLOv8 model
# yolo_model = YOLO("yolov8s.pt")  # Ensure this model file is present
# PROHIBITED_OBJECTS = {"cell phone", "book", "laptop", "earbuds"}  # YOLO class names

# # Ensure CSV files exist
# if not os.path.exists('data/signup.csv'):
#     with open('data/signup.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Name', 'Username', 'Email', 'Password', 'Image'])

# if not os.path.exists('data/login.csv'):
#     with open('data/login.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Username', 'Password'])

# # Global variables for proctoring
# proctoring_warnings = 0
# max_warnings = 3
# cheat_threshold = 0.70
# exam_active = False

# # Load CNN model for object detection
# # cnn_model = load_model(r'C:\Users\GEETHIKA\OneDrive\Desktop\final prjct\object_detection.h5', compile=False)

# # Initialize NoiseDetector
# noise_detector = NoiseDetector(threshold=0.7)

# # RL Agent for dynamic threshold adjustment
# class RLAgent:
#     def __init__(self):
#         self.thresholds = {
#             'sound': 20,
#             'head_pose': 15,
#             'object_confidence': 0.7
#         }

#     def update_thresholds(self, state, reward):
#         # Simple RL logic (replace with a proper RL algorithm)
#         if reward > 0:
#             self.thresholds['sound'] -= 1  # Increase sensitivity
#             self.thresholds['head_pose'] -= 1
#         else:
#             self.thresholds['sound'] += 1  # Decrease sensitivity
#             self.thresholds['head_pose'] += 1

# # Initialize RL Agent
# rl_agent = RLAgent()

# # Head Pose Detection (MediaPipe)
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # YOLO-based Object Detection
# def detect_objects(frame, confidence_threshold):
#     """Detect prohibited objects and count persons using YOLOv8."""
#     results = yolo_model(frame, conf=confidence_threshold)
#     prohibited_confidences = []
#     person_count = 0
    
#     for result in results:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             label = yolo_model.names[cls_id]
#             confidence = float(box.conf[0])
            
#             if label in PROHIBITED_OBJECTS:
#                 prohibited_confidences.append(confidence)
#             elif label == "person":
#                 person_count += 1
                
#     return prohibited_confidences, person_count

# # Proctoring Logic
# def proctoring_checks():
#     global proctoring_warnings, exam_active
#      # Initialize NoiseDetector
#     noise_detector = NoiseDetector(threshold=0.7)
#     # Start noise detection in a separate thread
#     noise_thread = threading.Thread(target=noise_detector.detect)
#     noise_thread.start()

#     cap = cv2.VideoCapture(0)
#     while exam_active:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # YOLO Object Detection
#         prohibited_confidences, person_count = detect_objects(frame, rl_agent.thresholds['object_confidence'])

#         # Head Pose Detection
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)
#         head_pose_alert = False

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Example: Check if the head is tilted
#                 nose_tip = face_landmarks.landmark[1]  # Nose tip landmark
#                 left_eye = face_landmarks.landmark[33]  # Left eye landmark
#                 right_eye = face_landmarks.landmark[263]  # Right eye landmark
#                 eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y]) 

#                 # If the distance is too small or too large, the head is tilted
#                 if eye_distance < 0.1 or eye_distance > 0.3:
#                     head_pose_alert = True

#         # Noise detection alert
#         noise_alert = noise_detector.detect()  # Assume detect() returns True if noise is detected
#         # Check for multiple persons
#         multiple_persons_alert = person_count > 1
#         # RL-based Decision Making
#         state = {
#             'sound_level': np.random.random(),  # Replace with actual sound data
#             'head_pose_alert': head_pose_alert,
#             'object_confidences': prohibited_confidences  # Convert to list for JSON serialization
#         }
#         reward = 1 if head_pose_alert else -1  # Replace with actual reward logic
#         rl_agent.update_thresholds(state, reward)

#         # Violation checks
#         violation = any([
#             head_pose_alert,
#             len(prohibited_confidences) > 0,
#             noise_alert,
#             multiple_persons_alert
#         ])

#         if violation:
#             proctoring_warnings += 1
#             flash(f"Warning {proctoring_warnings}/{max_warnings}: Suspicious activity detected!", "warning")
#             if proctoring_warnings >= max_warnings:
#                 flash("Exam terminated due to multiple violations!", "error")
#                 exam_active = False
#                 break

#     # Stop noise detection
#     noise_detector.stop()
#     noise_thread.join()
#     cap.release()

# @app.route('/')
# def frontend():
#     return render_template('frontend.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         # Check if username and password match
#         with open('data/login.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[0] == username and row[1] == password:
#                     with open('data/signup.csv', 'r') as signup_file:
#                         signup_reader = csv.reader(signup_file)
#                         for user_row in signup_reader:
#                             if user_row[1] == username:
#                                 session['loggedInUser'] = user_row[0] 
#                                 session['loggedInUsername'] = user_row[1]  
#                                 session['loggedInEmail'] = user_row[2]  
#                                 session['loggedInImage'] = user_row[4]  
#                                 return redirect(url_for('dashboard'))

#         flash("Invalid username or password", "error")
#         return redirect(url_for('login'))

#     return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form['name']
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']
#         image = request.files['image']

#         if password != confirm_password:
#             flash("Password mismatched", "error")
#             return redirect(url_for('signup'))

#         with open('data/signup.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[1] == username:
#                     flash("Username already exists", "error")
#                     return redirect(url_for('signup'))
#                 if row[2] == email:
#                     flash("Email already exists", "error")
#                     return redirect(url_for('signup'))

#         # Save image
#         image_filename = secure_filename(image.filename)
#         image_path = os.path.join('static/images', image_filename)
#         image.save(image_path)

#         with open('data/signup.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([name, username, email, password, image_filename])

#         with open('data/login.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([username, password])

#         flash("Successfully registered! Please login.", "success")
#         return redirect(url_for('login'))

#     return render_template('signup.html')

# @app.route('/dashboard')
# def dashboard():
#     if 'loggedInUser' not in session:
#         return redirect(url_for('login'))
#     return render_template('dash.html')

# @app.route('/update_password', methods=['POST'])
# def update_password():
#     if 'loggedInUsername' not in session:
#         return redirect(url_for('login'))

#     username = session['loggedInUsername']
#     new_password = request.json.get('password')

#     try:
#         updated_signup_data = []
#         with open('data/signup.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[1] == username:
#                     row[3] = new_password
#                 updated_signup_data.append(row)

#         with open('data/signup.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(updated_signup_data)

#         updated_login_data = []
#         with open('data/login.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[0] == username:
#                     row[1] = new_password
#                 updated_login_data.append(row)

#         with open('data/login.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(updated_login_data)

#         flash("Password updated successfully!", "success")
#         return {'success': True, 'redirect': url_for('login')}

#     except Exception as e:
#         flash("Failed to update password.", "error")
#         return {'success': False}, 500

# @app.route('/exam')
# def exam():
#     if 'loggedInUser' not in session:
#         return redirect(url_for('login'))
#     return render_template('exam.html')

# @app.route('/start_model', methods=['GET'])
# def start_model():
#     global exam_active
#     exam_active = True
#     try:
#         threading.Thread(target=proctoring_checks).start()  # Start proctoring in a separate thread
#         return jsonify({"message": "Proctoring started successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # @app.route('/proctoring_status', methods=['GET'])
# def check_proctoring():
#     global proctoring_warnings, exam_active

#     if not exam_active:
#         return jsonify({"status": "inactive"})

#     # Simulate proctoring data (replace with actual model outputs)
#     head_pose_alert = False  # Replace with actual head pose detection logic
#     object_confidences = [0.1, 0.2, 0.3]  # Replace with actual object detection confidences

#     # Check for violations
#     if head_pose_alert or any(conf > rl_agent.thresholds['object_confidence'] for conf in object_confidences):
#         proctoring_warnings += 1
#         if proctoring_warnings >= max_warnings:
#             exam_active = False
#             return jsonify({
#                 "status": "terminated",
#                 "message": "Exam terminated due to multiple violations.",
#                 "head_pose_alert": head_pose_alert,
#                 "object_confidences": object_confidences
#             })
#         else:
#             return jsonify({
#                 "status": "warning",
#                 "message": f"Look straight! Warning {proctoring_warnings}/{max_warnings}",
#                 "head_pose_alert": head_pose_alert,
#                 "object_confidences": object_confidences
#             })
#     else:
#         return jsonify({
#             "status": "ok",
#             "head_pose_alert": head_pose_alert,
#             "object_confidences": object_confidences
#         })

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
# import csv
# import os
# import subprocess
# from werkzeug.utils import secure_filename
# import threading
# import time
# import numpy as np
# import cv2
# from ultralytics import YOLO
# import mediapipe as mp
# from collections import deque
# from noise import NoiseDetector  # Import the NoiseDetector class
# import logging

# app = Flask(__name__)
# app.secret_key = 'your_secret_key'

# # Ensure the data directory exists
# if not os.path.exists('data'):
#     os.makedirs('data')

# # Initialize YOLOv8 model
# yolo_model = YOLO("yolov8s.pt")  # Ensure this model file is present
# PROHIBITED_OBJECTS = {"cell phone", "book", "laptop", "earbuds"}  # YOLO class names

# # Ensure CSV files exist
# if not os.path.exists('data/signup.csv'):
#     with open('data/signup.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Name', 'Username', 'Email', 'Password', 'Image'])

# if not os.path.exists('data/login.csv'):
#     with open('data/login.csv', 'w', newline='') as file:
#         writer = csv.writer(file)
#         writer.writerow(['Username', 'Password'])

# # Global variables for proctoring
# proctoring_warnings = 0
# max_warnings = 3
# cheat_threshold = 0.70
# exam_active = False

# # Load CNN model for object detection
# # cnn_model = load_model(r'C:\Users\GEETHIKA\OneDrive\Desktop\final prjct\object_detection.h5', compile=False)

# # Initialize NoiseDetector
# noise_detector = NoiseDetector(threshold=0.7)

# # RL Agent for dynamic threshold adjustment
# class RLAgent:
#     def __init__(self):
#         self.thresholds = {
#             'sound': 20,
#             'head_pose': 15,
#             'object_confidence': 0.7
#         }

#     def update_thresholds(self, state, reward):
#         # Simple RL logic (replace with a proper RL algorithm)
#         if reward > 0:
#             self.thresholds['sound'] -= 1  # Increase sensitivity
#             self.thresholds['head_pose'] -= 1
#         else:
#             self.thresholds['sound'] += 1  # Decrease sensitivity
#             self.thresholds['head_pose'] += 1

# # Initialize RL Agent
# rl_agent = RLAgent()

# # Head Pose Detection (MediaPipe)
# mp_face_mesh = mp.solutions.face_mesh
# face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# # YOLO-based Object Detection
# def detect_objects(frame, confidence_threshold):
#     """Detect prohibited objects and count persons using YOLOv8."""
#     results = yolo_model(frame, conf=confidence_threshold)
#     prohibited_confidences = []
#     person_count = 0
    
#     for result in results:
#         for box in result.boxes:
#             cls_id = int(box.cls[0])
#             label = yolo_model.names[cls_id]
#             confidence = float(box.conf[0])
            
#             if label in PROHIBITED_OBJECTS:
#                 prohibited_confidences.append(confidence)
#             elif label == "person":
#                 person_count += 1
                
#     return prohibited_confidences, person_count

# # Proctoring Logic
# def proctoring_checks():
#     global proctoring_warnings, exam_active
#      # Initialize NoiseDetector
#     noise_detector = NoiseDetector(threshold=0.7)
#     # Start noise detection in a separate thread
#     noise_thread = threading.Thread(target=noise_detector.detect)
#     noise_thread.start()

#     cap = cv2.VideoCapture(0)
#     while exam_active:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # YOLO Object Detection
#         prohibited_confidences, person_count = detect_objects(frame, rl_agent.thresholds['object_confidence'])

#         # Head Pose Detection
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         results = face_mesh.process(frame_rgb)
#         head_pose_alert = False

#         if results.multi_face_landmarks:
#             for face_landmarks in results.multi_face_landmarks:
#                 # Example: Check if the head is tilted
#                 nose_tip = face_landmarks.landmark[1]  # Nose tip landmark
#                 left_eye = face_landmarks.landmark[33]  # Left eye landmark
#                 right_eye = face_landmarks.landmark[263]  # Right eye landmark
#                 eye_distance = np.linalg.norm([left_eye.x - right_eye.x, left_eye.y - right_eye.y]) 

#                 # If the distance is too small or too large, the head is tilted
#                 if eye_distance < 0.1 or eye_distance > 0.3:
#                     head_pose_alert = True

#         # Noise detection alert
#         noise_alert = noise_detector.detect()  # Assume detect() returns True if noise is detected
#         # Check for multiple persons
#         multiple_persons_alert = person_count > 1
#         # RL-based Decision Making
#         state = {
#             'sound_level': np.random.random(),  # Replace with actual sound data
#             'head_pose_alert': head_pose_alert,
#             'object_confidences': prohibited_confidences  # Convert to list for JSON serialization
#         }
#         reward = 1 if head_pose_alert else -1  # Replace with actual reward logic
#         rl_agent.update_thresholds(state, reward)

#         # Violation checks
#         violation = any([
#             head_pose_alert,
#             len(prohibited_confidences) > 0,
#             noise_alert,
#             multiple_persons_alert
#         ])

#         if violation:
#             proctoring_warnings += 1
#             flash(f"Warning {proctoring_warnings}/{max_warnings}: Suspicious activity detected!", "warning")
#             if proctoring_warnings >= max_warnings:
#                 flash("Exam terminated due to multiple violations!", "error")
#                 exam_active = False
#                 break

#     # Stop noise detection
#     noise_detector.stop()
#     noise_thread.join()
#     cap.release()

# @app.route('/')
# def frontend():
#     return render_template('frontend.html')

# @app.route('/login', methods=['GET', 'POST'])
# def login():
#     if request.method == 'POST':
#         username = request.form['username']
#         password = request.form['password']

#         # Check if username and password match
#         with open('data/login.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[0] == username and row[1] == password:
#                     with open('data/signup.csv', 'r') as signup_file:
#                         signup_reader = csv.reader(signup_file)
#                         for user_row in signup_reader:
#                             if user_row[1] == username:
#                                 session['loggedInUser'] = user_row[0] 
#                                 session['loggedInUsername'] = user_row[1]  
#                                 session['loggedInEmail'] = user_row[2]  
#                                 session['loggedInImage'] = user_row[4]  
#                                 return redirect(url_for('dashboard'))

#         flash("Invalid username or password", "error")
#         return redirect(url_for('login'))

#     return render_template('login.html')

# @app.route('/signup', methods=['GET', 'POST'])
# def signup():
#     if request.method == 'POST':
#         name = request.form['name']
#         username = request.form['username']
#         email = request.form['email']
#         password = request.form['password']
#         confirm_password = request.form['confirm_password']
#         image = request.files['image']

#         if password != confirm_password:
#             flash("Password mismatched", "error")
#             return redirect(url_for('signup'))

#         with open('data/signup.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[1] == username:
#                     flash("Username already exists", "error")
#                     return redirect(url_for('signup'))
#                 if row[2] == email:
#                     flash("Email already exists", "error")
#                     return redirect(url_for('signup'))

#         # Save image
#         image_filename = secure_filename(image.filename)
#         image_path = os.path.join('static/images', image_filename)
#         image.save(image_path)

#         with open('data/signup.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([name, username, email, password, image_filename])

#         with open('data/login.csv', 'a', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerow([username, password])

#         flash("Successfully registered! Please login.", "success")
#         return redirect(url_for('login'))

#     return render_template('signup.html')

# @app.route('/dashboard')
# def dashboard():
#     if 'loggedInUser' not in session:
#         return redirect(url_for('login'))
#     return render_template('dash.html')

# @app.route('/update_password', methods=['POST'])
# def update_password():
#     if 'loggedInUsername' not in session:
#         return redirect(url_for('login'))

#     username = session['loggedInUsername']
#     new_password = request.json.get('password')

#     try:
#         updated_signup_data = []
#         with open('data/signup.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[1] == username:
#                     row[3] = new_password
#                 updated_signup_data.append(row)

#         with open('data/signup.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(updated_signup_data)

#         updated_login_data = []
#         with open('data/login.csv', 'r') as file:
#             reader = csv.reader(file)
#             for row in reader:
#                 if row[0] == username:
#                     row[1] = new_password
#                 updated_login_data.append(row)

#         with open('data/login.csv', 'w', newline='') as file:
#             writer = csv.writer(file)
#             writer.writerows(updated_login_data)

#         flash("Password updated successfully!", "success")
#         return {'success': True, 'redirect': url_for('login')}

#     except Exception as e:
#         flash("Failed to update password.", "error")
#         return {'success': False}, 500

# @app.route('/exam')
# def exam():
#     if 'loggedInUser' not in session:
#         return redirect(url_for('login'))
#     return render_template('exam.html')

# @app.route('/start_model', methods=['GET'])
# def start_model():
#     global exam_active
#     exam_active = True
#     try:
#         threading.Thread(target=proctoring_checks).start()  # Start proctoring in a separate thread
#         return jsonify({"message": "Proctoring started successfully"}), 200
#     except Exception as e:
#         return jsonify({"error": str(e)}), 500

# # @app.route('/proctoring_status', methods=['GET'])
# def check_proctoring():
#     global proctoring_warnings, exam_active

#     if not exam_active:
#         return jsonify({"status": "inactive"})

#     # Simulate proctoring data (replace with actual model outputs)
#     head_pose_alert = False  # Replace with actual head pose detection logic
#     object_confidences = [0.1, 0.2, 0.3]  # Replace with actual object detection confidences

#     # Check for violations
#     if head_pose_alert or any(conf > rl_agent.thresholds['object_confidence'] for conf in object_confidences):
#         proctoring_warnings += 1
#         if proctoring_warnings >= max_warnings:
#             exam_active = False
#             return jsonify({
#                 "status": "terminated",
#                 "message": "Exam terminated due to multiple violations.",
#                 "head_pose_alert": head_pose_alert,
#                 "object_confidences": object_confidences
#             })
#         else:
#             return jsonify({
#                 "status": "warning",
#                 "message": f"Look straight! Warning {proctoring_warnings}/{max_warnings}",
#                 "head_pose_alert": head_pose_alert,
#                 "object_confidences": object_confidences
#             })
#     else:
#         return jsonify({
#             "status": "ok",
#             "head_pose_alert": head_pose_alert,
#             "object_confidences": object_confidences
#         })

# @app.route('/logout')
# def logout():
#     session.clear()
#     return redirect(url_for('login'))

# if __name__ == '__main__':
#     app.run(debug=True)
# Title : Next-Level AI Exam Monitoring: A Reinforcement Learning CNN Approach
# FY25-Q1-S1-01/23-Ver:1.0
# Author : Geethika(Developer)
# Releases : FY25-Q1-S1-01/23-Ver:1.0-CHG- Converting login/register data from googlesheets to CSV Files. 
# Releases : FY25-Q1-S2-01/23-Ver:2.0-CHG-Enhancement - User Authentication System (login(), signup())
# Releases : FY25-Q1-S3-01/23-Ver:2.0-CHG-Enhancement - Dashboard Interface & Session Management (dashboard(), logout())
# Releases : FY25-Q1-S4-01/23-Ver:2.0-CHG-Enhancement - Upcoming exams and account details fields (loadContent())
# Releases : FY25-Q1-S5-01/23-Ver:2.0-CHG-Enhancement - System compatibility check (requestMediaAccess(),startExam())
# Releases : FY25-Q1-S6-01/23-Ver:2.0-CHG-Enhancement - protoring window and tab switch detection (startProctoringChecks(),handleMouseLeave())
# Releases : FY25-Q1-S7-01/23-Ver:2.0-CHG-Enhancement - Model development for headpose detection (pose())
# Releases : FY25-Q1-S8-01/23-Ver:2.0-CHG-Enhancement - Noise detection (audio_detection())
# Releases : FY25-Q1-S9-01/23-Ver:2.0-CHG-Enhancement - Integrated noise detection with proctoring (startProctoringChecks())
from flask import Flask, render_template, request, redirect, url_for, session, flash, jsonify
import csv
import os
import subprocess
from werkzeug.utils import secure_filename
import threading
import time
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import mediapipe as mp
from collections import deque
from noise import NoiseDetector  # Import the NoiseDetector class
import logging
from ultralytics import YOLO
from mss import mss
import logging
from flask_socketio import SocketIO

app = Flask(__name__)
app.secret_key = 'your_secret_key'
socketio = SocketIO(app)  # Initialize SocketIO

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Ensure the data directory exists
if not os.path.exists('data'):
    os.makedirs('data')

# Ensure CSV files exist
if not os.path.exists('data/signup.csv'):
    with open('data/signup.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Name', 'Username', 'Email', 'Password', 'Image'])

if not os.path.exists('data/login.csv'):
    with open('data/login.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Username', 'Password'])

# Global variables for proctoring
proctoring_warnings = 0
max_warnings = 3
cheat_threshold = 0.70
exam_active = False

# # Load CNN model for object detection
# cnn_model = load_model(r'C:\Users\GEETHIKA\OneDrive\Desktop\final prjct\object_detection.h5', compile=False)

# Initialize NoiseDetector
noise_detector = NoiseDetector(threshold=0.7)
# Configure logging
logging.basicConfig(filename="proctoring.log", level=logging.INFO, format="%(asctime)s - %(message)s")
# Load YOLOv8 model
yolo_model = YOLO("yolov8n.pt")  # Pre-trained on COCO dataset

# Define objects to monitor (e.g., mobile phone, earbuds, person)
SUSPICIOUS_OBJECTS = {"cell phone", "book", "laptop", "earbuds"}

# Define the bounding box of the proctoring window
proctoring_window = {"top": 100, "left": 100, "width": 800, "height": 600}
# Global variables for proctoring
proctoring_warnings = 0
max_warnings = 3
exam_active = False
# RL Agent for dynamic threshold adjustment
class RLAgent:
    def __init__(self):
        self.thresholds = {
            'sound': 20,
            'head_pose': 15,
            'object_confidence': 0.7
        }

    def update_thresholds(self, state, reward):
        # Simple RL logic (replace with a proper RL algorithm)
        if reward > 0:
            self.thresholds['sound'] -= 1  # Increase sensitivity
            self.thresholds['head_pose'] -= 1
        else:
            self.thresholds['sound'] += 1  # Decrease sensitivity
            self.thresholds['head_pose'] += 1

# Initialize RL Agent
rl_agent = RLAgent()

# Head Pose Detection (MediaPipe)
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# CNN-based Object Detection
def detect_objects(frame):
    resized_frame = cv2.resize(frame, (64, 64))  # Resize for CNN input
    normalized_frame = resized_frame / 255.0
    prediction = yolo_model.predict(np.expand_dims(normalized_frame, axis=0))
    return prediction[0]  # Return confidence scores for objects

# YOLOv8 Object Detection
def yolo_object_detection(frame):
    results = yolo_model(frame)
    face_count = 0
    suspicious_objects_detected = []

    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            label = result.names[int(box.cls[0])]

            # Check for suspicious objects
            if label in SUSPICIOUS_OBJECTS and confidence > 0.5:
                suspicious_objects_detected.append(label)

            # Count faces
            if label == "person":
                face_count += 1

    return face_count, suspicious_objects_detected

import cv2
import numpy as np
import mediapipe as mp
from collections import deque

# Add MediaPipe initialization
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Add Head Pose and Object Detection parameters
EYE_AR_THRESHOLD = 0.25
EYE_AR_CONSEC_FRAMES = 15
GAZE_THRESHOLD = 0.4
PROHIBITED_OBJECTS = {"cell phone", "book", "laptop", "earbuds"}
SUSPICIOUS_OBJECTS = PROHIBITED_OBJECTS  # Alias for existing code

# Add head pose detection functions
def eye_aspect_ratio(eye_landmarks):
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])
    return (A + B) / (2.0 * C)

def get_head_pose(image, landmarks):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0), (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0), (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ])
    size = image.shape
    focal_length = size[1]
    center = (size[1]/2, size[0]/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], 
                             [0, focal_length, center[1]], 
                             [0, 0, 1]], dtype="double")
    image_points = np.array([
        (landmarks[1].x*size[1], landmarks[1].y*size[0]),
        (landmarks[152].x*size[1], landmarks[152].y*size[0]),
        (landmarks[226].x*size[1], landmarks[226].y*size[0]),
        (landmarks[446].x*size[1], landmarks[446].y*size[0]),
        (landmarks[57].x*size[1], landmarks[57].y*size[0]),
        (landmarks[287].x*size[1], landmarks[287].y*size[0])
    ], dtype="double")
    dist_coeffs = np.zeros((4, 1))
    success, rotation_vector, _ = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return rotation_vector

def is_looking_sideways(rotation_vector):
    return abs(rotation_vector[1][0]) > GAZE_THRESHOLD


# Proctoring Logic
def proctoring_checks():
    global proctoring_warnings, exam_active
    face_detection_frames = deque(maxlen=EYE_AR_CONSEC_FRAMES)
    noise_detector = NoiseDetector(threshold=0.7)

    # Start noise detection in a separate thread
    noise_thread = threading.Thread(target=noise_detector.detect)
    noise_thread.start()

    cap = cv2.VideoCapture(0)

    while exam_active:
        try:
            ret, frame = cap.read()
            if not ret:
                continue

            # Object detection
            results = yolo_model(frame, conf=0.6)
            face_count = 0
            prohibited_objects = []
            
            # Head pose and eye detection
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_results = face_mesh.process(frame_rgb)
            head_pose_alert = False
            eye_closure_alert = False

            # Process YOLO results
            for result in results:
                for box in result.boxes:
                    label = yolo_model.names[int(box.cls[0])]
                    if label == "person" and float(box.conf[0]) > 0.7:
                        face_count += 1
                    if label in PROHIBITED_OBJECTS and float(box.conf[0]) > 0.7:
                        prohibited_objects.append(label)

            # Process head pose and eye closure
            if face_results.multi_face_landmarks:
                for face_landmarks in face_results.multi_face_landmarks:
                    # Head pose detection
                    rotation_vector = get_head_pose(frame, face_landmarks.landmark)
                    if is_looking_sideways(rotation_vector):
                        head_pose_alert = True

                    # Eye closure detection
                    left_eye = [face_landmarks.landmark[i] for i in [362, 385, 387, 263, 373, 380]]
                    right_eye = [face_landmarks.landmark[i] for i in [33, 160, 158, 133, 153, 144]]
                    left_ear = eye_aspect_ratio([(p.x, p.y) for p in left_eye])
                    right_ear = eye_aspect_ratio([(p.x, p.y) for p in right_eye])
                    face_detection_frames.append((left_ear + right_ear) / 2 < EYE_AR_THRESHOLD)
                    
                    if len(face_detection_frames) == EYE_AR_CONSEC_FRAMES:
                        eye_closure_alert = all(face_detection_frames)

            # Check for violations
            violations = []
            if face_count > 1:
                violations.append("multiple faces detected")
            if prohibited_objects:
                violations.append(f"prohibited objects: {', '.join(prohibited_objects)}")
            if head_pose_alert:
                violations.append("improper head position")
            if eye_closure_alert:
                violations.append("eyes closed")

            # Handle violations
            if violations:
                proctoring_warnings += 1
                warning_msg = f"Warning {proctoring_warnings}/{max_warnings}: {', '.join(violations)}"
                socketio.emit('proctoring_alert', {
                    'message': warning_msg,
                    'violations': violations
                })
                
                if proctoring_warnings >= max_warnings:
                    socketio.emit('exam_terminated', {
                        'message': "Exam terminated due to multiple violations!"
                    })
                    exam_active = False
                    break
        except Exception as e:
            logging.error(f"Proctoring error: {e}")
            break

    cap.release()
    face_mesh.close()


    # Stop noise detection
    noise_detector.stop()
    noise_thread.join()
    cap.release()

# Flask routes (unchanged)
@app.route('/')
def frontend():
    return render_template('frontend.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']

        # Check if username and password match
        with open('data/login.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username and row[1] == password:
                    with open('data/signup.csv', 'r') as signup_file:
                        signup_reader = csv.reader(signup_file)
                        for user_row in signup_reader:
                            if user_row[1] == username:
                                session['loggedInUser'] = user_row[0] 
                                session['loggedInUsername'] = user_row[1]  
                                session['loggedInEmail'] = user_row[2]  
                                session['loggedInImage'] = user_row[4]  
                                return redirect(url_for('dashboard'))

        flash("Invalid username or password", "error")
        return redirect(url_for('login'))

    return render_template('login.html')

@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        name = request.form['name']
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        image = request.files['image']

        if password != confirm_password:
            flash("Password mismatched", "error")
            return redirect(url_for('signup'))

        with open('data/signup.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == username:
                    flash("Username already exists", "error")
                    return redirect(url_for('signup'))
                if row[2] == email:
                    flash("Email already exists", "error")
                    return redirect(url_for('signup'))

        # Save image
        image_filename = secure_filename(image.filename)
        image_path = os.path.join('static/images', image_filename)
        image.save(image_path)

        with open('data/signup.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([name, username, email, password, image_filename])

        with open('data/login.csv', 'a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([username, password])

        flash("Successfully registered! Please login.", "success")
        return redirect(url_for('login'))

    return render_template('signup.html')

@app.route('/dashboard')
def dashboard():
    if 'loggedInUser' not in session:
        return redirect(url_for('login'))
    return render_template('dash.html')

@app.route('/update_password', methods=['POST'])
def update_password():
    if 'loggedInUsername' not in session:
        return redirect(url_for('login'))

    username = session['loggedInUsername']
    new_password = request.json.get('password')

    try:
        updated_signup_data = []
        with open('data/signup.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[1] == username:
                    row[3] = new_password
                updated_signup_data.append(row)

        with open('data/signup.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_signup_data)

        updated_login_data = []
        with open('data/login.csv', 'r') as file:
            reader = csv.reader(file)
            for row in reader:
                if row[0] == username:
                    row[1] = new_password
                updated_login_data.append(row)

        with open('data/login.csv', 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerows(updated_login_data)

        flash("Password updated successfully!", "success")
        return {'success': True, 'redirect': url_for('login')}

    except Exception as e:
        flash("Failed to update password.", "error")
        return {'success': False}, 500

@app.route('/exam')
def exam():
    if 'loggedInUser' not in session:
        return redirect(url_for('login'))
    return render_template('exam.html')

@app.route('/start_model', methods=['GET'])
def start_model():
    global exam_active
    exam_active = True
    try:
        threading.Thread(target=proctoring_checks).start()  # Start proctoring in a separate thread
        return jsonify({"message": "Proctoring started successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/proctoring_status', methods=['GET'])
def check_proctoring():
    global proctoring_warnings, exam_active

    if not exam_active:
        return jsonify({"status": "inactive"})

    # Simulate proctoring data (replace with actual model outputs)
    head_pose_alert = False  # Default to False
    object_confidences = [0.1, 0.2, 0.3]  # Default to low confidence values

    # Check for violations
    if head_pose_alert or any(conf > rl_agent.thresholds['object_confidence'] for conf in object_confidences):
        proctoring_warnings += 1
        if proctoring_warnings >= max_warnings:
            exam_active = False
            return jsonify({
                "status": "terminated",
                "message": "Exam terminated due to multiple violations.",
                "head_pose_alert": head_pose_alert,
                "object_confidences": object_confidences
            })
        else:
            return jsonify({
                "status": "warning",
                "message": f"Look straight! Warning {proctoring_warnings}/{max_warnings}",
                "head_pose_alert": head_pose_alert,
                "object_confidences": object_confidences
            })
    else:
        return jsonify({
            "status": "ok",
            "head_pose_alert": head_pose_alert,
            "object_confidences": object_confidences
        })

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

if __name__ == '__main__':
    app.run(debug=True,port=5879)