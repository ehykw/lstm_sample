# pose_gesture_gru/train_gru.py
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import os

# データ読み込み
labels = []
X_list = []
label_list = []

print("[INFO] Loading data from 'data/'...")
data_dir = "data"
for file in os.listdir(data_dir):
    if file.endswith(".npy"):
        label = file.replace(".npy", "")
        data = np.load(os.path.join(data_dir, file))  # shape = (N, 30, 99)
        X_list.append(data)
        label_list.extend([label] * len(data))

X = np.concatenate(X_list, axis=0)
y = np.array(label_list)
print(f"[INFO] Loaded {X.shape[0]} samples across {len(np.unique(y))} labels.")

# ラベルエンコード
y_encoded = LabelEncoder().fit_transform(y)
label_classes = np.unique(y)
np.save("labels.npy", label_classes)
print("[INFO] Encoded labels and saved to labels.npy")

# 学習・テスト分割
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# GRUモデル構築
model = tf.keras.Sequential([
    tf.keras.layers.GRU(64, return_sequences=True, input_shape=(30, 99)),
    tf.keras.layers.GRU(32),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(len(label_classes), activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("[INFO] Training GRU model...")
model.fit(X_train, y_train, epochs=30, validation_data=(X_test, y_test))
model.save("gesture_gru.h5")
print("[INFO] Model trained and saved as gesture_gru.h5")
