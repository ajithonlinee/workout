import cv2
import mediapipe as mp
import numpy as np
import os

# --- Setup MediaPipe ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# --- Path to your dataset ---
# Make sure your dataset is organized in subfolders, where each subfolder
# is named after the workout (e.g., "workouts/squat", "workouts/push_up")
DATASET_PATH = "workouts" 

print("Starting image processing...")

# --- Data and Label Lists ---
features_data = []
labels_data = []

# --- Main Processing Loop ---
# Loop through each subfolder (each workout) in the dataset path
for workout_name in os.listdir(DATASET_PATH):
    workout_folder_path = os.path.join(DATASET_PATH, workout_name)
    
    # Skip any files that aren't directories
    if not os.path.isdir(workout_folder_path):
        continue
        
    print(f"--- Processing folder: {workout_name} ---")

    # Loop through each image in the subfolder
    for image_name in os.listdir(workout_folder_path):
        image_path = os.path.join(workout_folder_path, image_name)

        # Read the image
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not read image {image_name}. Skipping.")
                continue
        except Exception as e:
            print(f"Error reading image {image_name}: {e}. Skipping.")
            continue
            
        # Recolor image to RGB for MediaPipe
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Process the image and find pose
        results = pose.process(image_rgb)
        
        # Ensure pose landmarks were detected
        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            
            # --- Normalization of landmarks ---
            # Get coordinates of the nose (landmark 0) to use as the origin
            origin = landmarks[mp_pose.PoseLandmark.NOSE.value]
            origin_x, origin_y = origin.x, origin.y
            
            # Create a temporary list to hold the normalized coordinates for this image
            current_features = []
            for landmark in landmarks:
                # Calculate coordinates relative to the nose
                normalized_x = landmark.x - origin_x
                normalized_y = landmark.y - origin_y
                current_features.extend([normalized_x, normalized_y])
            
            # Add the normalized features and the corresponding label to our lists
            features_data.append(current_features)
            labels_data.append(workout_name)
            
            print(f"Successfully processed {image_name}")
        else:
            print(f"Warning: No pose detected in {image_name}. Skipping.")

print("\n--- Image processing complete ---")

# --- Save the processed data ---
# Convert lists to NumPy arrays
X = np.array(features_data)
y = np.array(labels_data)

# Save the arrays to files, which will be used by the training script
np.save('X_data.npy', X)
np.save('y_data.npy', y)

print(f"\nSuccessfully saved data.")
print(f"Shape of features (X): {X.shape}")
print(f"Shape of labels (y): {y.shape}")