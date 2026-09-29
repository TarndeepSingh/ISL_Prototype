import os
import json
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Masking, Bidirectional, LSTM, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping

DATA_PATH = "custom_data"
ACTIONS = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
label_map = {action: idx for idx, action in enumerate(ACTIONS)}

with open("custom_labels.json", "w") as f:
    json.dump({str(v): k for k, v in label_map.items()}, f, indent=4)

sequences, labels = [], []
for action in ACTIONS:
    action_dir = os.path.join(DATA_PATH, action)
    for npy_file in os.listdir(action_dir):
        if npy_file.endswith(".npy"):
            res = np.load(os.path.join(action_dir, npy_file))
            if res.shape == (30, 258):
                sequences.append(res)
                labels.append(label_map[action])

X = np.array(sequences, dtype=np.float32)
y = tf.keras.utils.to_categorical(labels, num_classes=len(ACTIONS))

X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.15, random_state=42, stratify=np.argmax(y, axis=1))

print(f"Loaded {len(X)} total sequences across {len(ACTIONS)} classes.")

# Build Bi-LSTM Architecture
model = Sequential([
    Masking(mask_value=0.0, input_shape=(30, 258)),
    Bidirectional(LSTM(64, return_sequences=True)),
    Dropout(0.3),
    BatchNormalization(),
    Bidirectional(LSTM(64)),
    Dropout(0.3),
    BatchNormalization(),
    Dense(64, activation='relu'),
    Dropout(0.3),
    Dense(len(ACTIONS), activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

print("\nStarting local model training...")
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=60, batch_size=16, callbacks=callbacks)

model.save("custom_model.h5")
print("\nModel saved successfully as 'custom_model.h5' and labels exported to 'custom_labels.json'!")