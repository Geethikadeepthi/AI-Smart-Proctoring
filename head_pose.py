import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Constants for eye aspect ratio (EAR)
EYE_AR_THRESHOLD = 0.25  # Threshold for detecting closed eyes
EYE_AR_CONSEC_FRAMES = 3  # Number of consecutive frames to confirm eye closure

# Constants for gaze detection
GAZE_THRESHOLD = 0.3  # Threshold for detecting sideways gaze

# Function to calculate eye aspect ratio (EAR)
def eye_aspect_ratio(eye_landmarks):
    """
    Calculate the Eye Aspect Ratio (EAR) for eye closure detection.
    EAR = (|p2 - p6| + |p3 - p5|) / (2 * |p1 - p4|)
    """
    # Vertical distances
    A = np.linalg.norm(eye_landmarks[1] - eye_landmarks[5])
    B = np.linalg.norm(eye_landmarks[2] - eye_landmarks[4])

    # Horizontal distance
    C = np.linalg.norm(eye_landmarks[0] - eye_landmarks[3])

    # EAR formula
    ear = (A + B) / (2.0 * C)
    return ear

# Function to calculate head pose
def get_head_pose(image, landmarks):
    """
    Calculate the head pose using facial landmarks.
    Returns rotation and translation vectors.
    """
    # 3D model points (generic face model)
    model_points = np.array([
        (0.0, 0.0, 0.0),          # Nose tip
        (0.0, -330.0, -65.0),     # Chin
        (-225.0, 170.0, -135.0),  # Left eye left corner
        (225.0, 170.0, -135.0),   # Right eye right corner
        (-150.0, -150.0, -125.0), # Left mouth corner
        (150.0, -150.0, -125.0)   # Right mouth corner
    ])

    # Camera internals
    size = image.shape
    focal_length = size[1]
    center = (size[1] / 2, size[0] / 2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]], dtype="double")

    # 2D image points from landmarks
    image_points = np.array([
        (landmarks[1].x * size[1], landmarks[1].y * size[0]),  # Nose tip
        (landmarks[152].x * size[1], landmarks[152].y * size[0]),  # Chin
        (landmarks[226].x * size[1], landmarks[226].y * size[0]),  # Left eye left corner
        (landmarks[446].x * size[1], landmarks[446].y * size[0]),  # Right eye right corner
        (landmarks[57].x * size[1], landmarks[57].y * size[0]),   # Left mouth corner
        (landmarks[287].x * size[1], landmarks[287].y * size[0])   # Right mouth corner
    ], dtype="double")

    # Solve for pose
    dist_coeffs = np.zeros((4, 1))  # Assuming no lens distortion
    (success, rotation_vector, translation_vector) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    return rotation_vector, translation_vector

# Function to check if the user is looking sideways
def is_looking_sideways(rotation_vector):
    """
    Check if the user is looking sideways based on the yaw angle.
    """
    yaw = rotation_vector[1]  # Yaw is the second element in the rotation vector
    threshold = 0.5  # Increased threshold to ignore slight movements
    return abs(yaw) > threshold

# Function to process a single frame for head pose and eye detection
def process_frame(frame):
    """
    Process a single frame for head pose and eye detection.
    Returns a list of violations (e.g., ["head_pose", "eye_closure"]).
    """
    violations = []
    
    # Convert the frame to RGB
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Get head pose
            rotation_vector, _ = get_head_pose(frame, face_landmarks.landmark)

            # Check if the user is looking sideways
            if is_looking_sideways(rotation_vector):
                violations.append("head_pose")
                print("Warning: Improper head position detected!")

            # Extract eye landmarks
            left_eye_landmarks = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in [362, 385, 387, 263, 373, 380]])
            right_eye_landmarks = np.array([(face_landmarks.landmark[i].x * frame.shape[1], face_landmarks.landmark[i].y * frame.shape[0]) for i in [33, 160, 158, 133, 153, 144]])

            # Calculate eye aspect ratio (EAR)
            left_ear = eye_aspect_ratio(left_eye_landmarks)
            right_ear = eye_aspect_ratio(right_eye_landmarks)

            # Check for eye closure
            if left_ear < EYE_AR_THRESHOLD or right_ear < EYE_AR_THRESHOLD:
                violations.append("eye_closure")
                print("Warning: Eyes closed detected!")

    return violations

# Main function for standalone testing (optional)
if __name__ == "__main__":
    # Open the video feed
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Process the frame
        violations = process_frame(frame)

        # Display violations on the frame
        if violations:
            cv2.putText(frame, f"Violations: {', '.join(violations)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # Display the frame
        cv2.imshow("Head Pose and Eye Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Cleanup
    cap.release()
    cv2.destroyAllWindows()