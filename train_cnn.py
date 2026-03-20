import os
import numpy as np
import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.layers import Dropout
from sklearn.model_selection import train_test_split
import joblib

# -------------------------------
# Parameters
# -------------------------------
IMG_SIZE = 64   # reduced size for faster training
DATA_LIMIT = 1050  # per class

# -------------------------------
# Load Dataset
# -------------------------------
def load_data(base_path):
    data = []
    labels = []

    categories = ['clean', 'stego']

    for label, category in enumerate(categories):
        folder = os.path.join(base_path, category)
        count = 0

        for file in os.listdir(folder):
            if count >= DATA_LIMIT:
                break

            path = os.path.join(folder, file)

            try:
                img = cv2.imread(path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = img / 255.0  # normalize

                data.append(img)
                labels.append(label)
                count += 1

            except:
                pass

    return np.array(data), np.array(labels)

# -------------------------------
# Load Data
# -------------------------------
print("Loading dataset...")
X, y = load_data("../dataset/train")

# reshape for CNN
X = X.reshape(-1, IMG_SIZE, IMG_SIZE, 1)

from sklearn.utils import shuffle
X, y = shuffle(X, y, random_state=42)

# split data
print("Splitting data...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# -------------------------------
# Build CNN Model
# -------------------------------
print("Building model...")

model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D(2,2),
  
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D(2,2),

# 👉 ADD THIS BLOCK
    Conv2D(128, (3,3), activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5), 
    Dense(1, activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# -------------------------------
# Train Model
# -------------------------------
print("Training model...")
history = model.fit(
    X_train, y_train,
    epochs=8,            # keep low for speed
    batch_size=32,
    validation_data=(X_test, y_test)
)

# -------------------------------
# Evaluate Model
# -------------------------------
print("Evaluating model...")
loss, accuracy = model.evaluate(X_test, y_test)
print("Test Accuracy:", accuracy)

# -------------------------------
# Save Model
# -------------------------------
print("Saving model...")
model.save("cnn_model.h5")

# save history (optional)
joblib.dump(history.history, "history.pkl")

print("Done ✅ CNN model saved")