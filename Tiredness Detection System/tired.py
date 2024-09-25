import cv2
import dlib
import numpy as np
from scipy.spatial import distance as dist

# Define the Eye Aspect Ratio (EAR) calculation
def eye_aspect_ratio(eye):
    # Compute the vertical distances
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    
    # Compute the horizontal distance
    C = dist.euclidean(eye[0], eye[3])
    
    # Calculate EAR
    ear = (A + B) / (2.0 * C)
    return ear

# Thresholds for eye state and tiredness detection
EAR_THRESHOLD = 0.25  # Threshold for detecting closed eyes
EAR_CONSEC_FRAMES = 48  # Number of consecutive frames eyes must be below threshold to consider as "tired"

# Initialize dlib's face detector and facial landmarks predictor
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# Indexes for the eye landmarks
(left_eye_start, left_eye_end) = (42, 48)
(right_eye_start, right_eye_end) = (36, 42)

# Initialize counters
frame_counter = 0

# Start video capture (0 for the primary camera)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = detector(gray)
    
    # Status message to be updated
    status = "NOT TIRED"
    
    for face in faces:
        # Predict facial landmarks
        shape = predictor(gray, face)
        landmarks = np.array([[p.x, p.y] for p in shape.parts()])
        
        # Get the left and right eye coordinates
        left_eye = landmarks[left_eye_start:left_eye_end]
        right_eye = landmarks[right_eye_start:right_eye_end]
        
        # Calculate EAR for both eyes
        left_ear = eye_aspect_ratio(left_eye)
        right_ear = eye_aspect_ratio(right_eye)
        
        # Average EAR for both eyes
        ear = (left_ear + right_ear) / 2.0
        
        # Visualize the eyes
        cv2.polylines(frame, [left_eye], True, (0, 255, 0), 1)
        cv2.polylines(frame, [right_eye], True, (0, 255, 0), 1)
        
        # Check if EAR is below the blink threshold
        if ear < EAR_THRESHOLD:
            frame_counter += 1
            if frame_counter >= EAR_CONSEC_FRAMES:
                status = "TIRED"
        else:
            frame_counter = 0
            status = "NOT TIRED"
        
        # Display EAR value for debugging
        cv2.putText(frame, f"EAR: {ear:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Display status
    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255) if status == "TIRED" else (0, 255, 0), 2)
    
    # Display the frame
    cv2.imshow("Tiredness Detection", frame)
    
    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
