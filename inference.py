import cv2
import mediapipe as mp
import numpy as np
from keras.models import load_model

def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    return 360 - angle if angle > 180.0 else angle

try:
    model = load_model("model.h5")
    labels = np.load("labels.npy", allow_pickle=True)
except Exception as e:
    print(f"Error loading model or labels: {e}")
    exit()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

pose_rules = {
    'bicep_curl': {'landmarks_to_track': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST], 'ideal_angle_at_peak': 30.0, 'rep_angle_thresholds': (160.0, 45.0), 'stage_logic': ('up', 'down')},
    'squat': {'landmarks_to_track': [mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE, mp_pose.PoseLandmark.LEFT_ANKLE], 'ideal_angle_at_peak': 90.0, 'rep_angle_thresholds': (170.0, 100.0), 'stage_logic': ('up', 'down')},
    'push_up': {'angle_for_reps': {'landmarks': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_ELBOW, mp_pose.PoseLandmark.LEFT_WRIST], 'thresholds': (160.0, 90.0), 'stage_logic': ('up', 'down')}, 'angle_for_form': {'landmarks': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE], 'ideal_angle': 170.0}},
    'hip_thrust': { 'landmarks_to_track': [mp_pose.PoseLandmark.LEFT_SHOULDER, mp_pose.PoseLandmark.LEFT_HIP, mp_pose.PoseLandmark.LEFT_KNEE], 'ideal_angle_at_peak': 170.0, 'rep_angle_thresholds': (160.0, 120.0), 'stage_logic': ('up', 'down') }
}

cap = cv2.VideoCapture(0)

# --- State and Counter Variables ---
app_state = "DETECTING"  # DETECTING, LOCKED
current_workout = "None"
rep_counter, pose_stage = 0, None
feedback_message = "Show a pose to start"
last_angle, activity_timer = 0, 0
ACTIVITY_TIMEOUT = 120  # Frames of inactivity before reset (approx. 4 seconds)

# --- New Confirmation Lock Variables ---
detection_candidate = None
detection_frames = 0
DETECTION_CONFIRMATION_FRAMES = 10 # Must see pose for 10 frames to lock

## Main Loop
while cap.isOpened():
    ret, frame = cap.read()
    if not ret: break

    frame = cv2.flip(frame, 1)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    try:
        landmarks = results.pose_landmarks.landmark
        
        # --- STATE MACHINE ---
        if app_state == "DETECTING":
            # Classification logic
            origin = landmarks[mp_pose.PoseLandmark.NOSE.value]
            norm_features = []
            for lm in landmarks: norm_features.extend([lm.x - origin.x, lm.y - origin.y])
            prediction = model.predict(np.array([norm_features]))
            
            if np.max(prediction) > 0.90:
                predicted_workout = labels[np.argmax(prediction)]
                
                if predicted_workout == detection_candidate:
                    detection_frames += 1
                else:
                    detection_candidate = predicted_workout
                    detection_frames = 1
                
                if detection_frames >= DETECTION_CONFIRMATION_FRAMES:
                    current_workout = detection_candidate
                    app_state = "LOCKED"
                    rep_counter, pose_stage, activity_timer = 0, None, 0
            else:
                detection_frames = 0
                detection_candidate = None

        elif app_state == "LOCKED":
            # Rep counting and form scoring logic
            rules = pose_rules[current_workout]
            angle_to_check = 0
            
            if current_workout == 'push_up':
                # (Same push-up logic as before)
                rep_landmarks, form_landmarks = rules['angle_for_reps']['landmarks'], rules['angle_for_form']['landmarks']
                p1_rep, p2_rep, p3_rep = [[landmarks[l.value].x, landmarks[l.value].y] for l in rep_landmarks]
                rep_angle = calculate_angle(p1_rep, p2_rep, p3_rep)
                angle_to_check = rep_angle
                p1_form, p2_form, p3_form = [[landmarks[l.value].x, landmarks[l.value].y] for l in form_landmarks]
                form_angle = calculate_angle(p1_form, p2_form, p3_form)
                up_thresh, down_thresh = rules['angle_for_reps']['thresholds']
                up_stage, down_stage = rules['angle_for_reps']['stage_logic']
                if rep_angle < down_thresh: pose_stage = down_stage
                if rep_angle > up_thresh and pose_stage == down_stage:
                    pose_stage = up_stage; rep_counter += 1
                    error = abs(form_angle - rules['angle_for_form']['ideal_angle'])
                    feedback_message = "Great Form!" if error < 15 else "Keep Back Straight"
            else:
                # (Same single-angle logic as before)
                track_landmarks = rules['landmarks_to_track']
                p1, p2, p3 = [[landmarks[l.value].x, landmarks[l.value].y] for l in track_landmarks]
                angle = calculate_angle(p1, p2, p3)
                angle_to_check = angle
                up_thresh, down_thresh = rules['rep_angle_thresholds']
                up_stage, down_stage = rules['stage_logic']
                if angle < down_thresh: pose_stage = down_stage
                if angle > up_thresh and pose_stage == down_stage:
                    pose_stage = up_stage; rep_counter += 1
                    error = abs(angle - rules['ideal_angle_at_peak'])
                    feedback_message = "Perfect!" if error < 15 else "Check Angle"

            # Inactivity timer
            if abs(angle_to_check - last_angle) < 1.5:
                activity_timer += 1
            else:
                activity_timer = 0
            last_angle = angle_to_check

            if activity_timer > ACTIVITY_TIMEOUT:
                feedback_message = f"Set Complete! Reps: {rep_counter}"
                current_workout = "None"
                app_state = "DETECTING"

    except Exception as e:
        app_state = "DETECTING"
        pass

    # --- Render Visuals ---
    cv2.rectangle(image, (0,0), (350,110), (245,117,16), -1)
    workout_display_name = current_workout.replace("_", " ").title() if current_workout != "None" else "Detecting..."
    cv2.putText(image, 'WORKOUT', (15,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, workout_display_name, (10,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'REPS', (200,20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, str(rep_counter), (195,70), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 2, cv2.LINE_AA)
    cv2.putText(image, 'FEEDBACK', (15,80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1, cv2.LINE_AA)
    cv2.putText(image, feedback_message, (10,100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 1, cv2.LINE_AA)
    
    mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                              mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2), 
                              mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2))               
    
    cv2.imshow('Workout AI Trainer', image)

    if cv2.waitKey(10) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()