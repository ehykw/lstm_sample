# recognize_video_lstm.py
import cv2
import mediapipe as mp
import tensorflow as tf
import numpy as np
import sys

if (len(sys.argv) == 1):
    print("コマンド <動画ファイル名>")
    sys.exit(0)
    
# モデルとラベル読み込み
model = tf.keras.models.load_model("gesture_lstm.h5")  #こちらがモデル
labels = np.load("labels.npy")   # こちらがラベル

# MediaPipe Pose 初期化
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

# 動画ファイルの読み込み
cap = cv2.VideoCapture(sys.argv[1])  # ← 任意の動画パスに変更

sequence = []
seq_length = 30

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints += [lm.x, lm.y, lm.z]
        sequence.append(keypoints)
        if len(sequence) > seq_length:
            sequence.pop(0)

        if len(sequence) == seq_length:
            input_data = np.expand_dims(sequence, axis=0)
            prediction = model.predict(input_data, verbose=0)[0]
            gesture = labels[np.argmax(prediction)]
            prob = np.max(prediction)

            if prob > 0.8:
                cv2.putText(frame, f"{gesture} ({prob:.2f})", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

    cv2.imshow("LSTM Gesture Recognition (Video)", frame)
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()