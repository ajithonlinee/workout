import numpy as np
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model
import os

print("Starting model training...")

# --- 1. Load Processed Data ---
# Load the feature and label arrays created by process_images.py
if not os.path.exists('X_data.npy') or not os.path.exists('y_data.npy'):
    print("Error: X_data.npy or y_data.npy not found.")
    print("Please run the process_images.py script first to generate the data.")
    exit()

X = np.load('X_data.npy')
y_labels = np.load('y_data.npy')

# --- 2. Process Labels ---
# Get the unique class names (e.g., ['bicep_curl', 'push_up', 'squat'])
unique_labels = np.unique(y_labels)

# Create a dictionary to map class names to integers
label_to_int = {label: i for i, label in enumerate(unique_labels)}

# Convert the string labels in y_labels to integers
y_integers = np.array([label_to_int[label] for label in y_labels])

# One-hot encode the integer labels
# The shape of y will be (number_of_samples, number_of_classes)
y = to_categorical(y_integers)

print(f"Data loaded. Found {len(unique_labels)} classes: {unique_labels}")

# --- 3. Shuffle the Data ---
# This is crucial for effective training to prevent the model from learning the order of data.
indices = np.arange(X.shape[0])
np.random.shuffle(indices)

X_shuffled = X[indices]
y_shuffled = y[indices]

# --- 4. Define the Neural Network Architecture ---
# The architecture is the same as the original project.
input_layer = Input(shape=(X.shape[1],))  # Input shape is 66 (33 landmarks * 2 coords)

# Hidden layers
hidden_layer_1 = Dense(128, activation="tanh")(input_layer)
hidden_layer_2 = Dense(64, activation="tanh")(hidden_layer_1)

# Output layer
# The number of neurons is the number of unique classes (e.g., 3 for squat, pushup, curl)
output_layer = Dense(y.shape[1], activation="softmax")(hidden_layer_2)

# Create the model
model = Model(inputs=input_layer, outputs=output_layer)

# --- 5. Compile and Train the Model ---
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

print("\nStarting training...")
# Train the model on the shuffled data
model.fit(X_shuffled, y_shuffled, epochs=80)
print("Training complete.")

# --- 6. Save the Trained Model and Labels ---
model.save("model.h5")
np.save("labels.npy", unique_labels)

print("\nModel saved as model.h5")
print("Labels saved as labels.npy")