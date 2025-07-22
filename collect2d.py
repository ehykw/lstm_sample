import cv2
import mediapipe as mp
import numpy as np
import os
import sys

if (len(sys.argv) < 3):
    print("python collect2d.py <動画ファイル> <ラベル名>")
    sys.exit(1)
label = sys.argv[2]
seq_length = 30
save_path = f"data/{label}.npy"

os.makedirs("data", exist_ok=True)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()

cap = cv2.VideoCapture(sys.argv[1])
sequence = []

def normalize_keypoints_2d(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)  # (33, 3)

    # 2Dだけ使う
    keypoints_2d = keypoints[:, :2]  # (33, 2)

    # 腰の中心を原点に（left_hip: 23, right_hip: 24）
    left_hip = keypoints_2d[23]
    right_hip = keypoints_2d[24]
    hip_center = (left_hip + right_hip) / 2
    keypoints_2d -= hip_center

    # 身長でスケール正規化（noseと両足首の中点の距離）
    nose = keypoints_2d[0]
    left_ankle = keypoints_2d[27]
    right_ankle = keypoints_2d[28]
    ankle_center = (left_ankle + right_ankle) / 2
    height = np.linalg.norm(nose - ankle_center)
    if height > 0:
        keypoints_2d /= height

    return keypoints_2d.flatten().tolist()  # 1次元リストに戻す (66次元)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)

    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])  # zは後で無視

        keypoints = normalize_keypoints_2d(keypoints)
        sequence.append(keypoints)

        if len(sequence) == seq_length:
            data = np.expand_dims(sequence, axis=0)  # (1, 30, 66)
            if os.path.exists(save_path):
                existing = np.load(save_path)
                data = np.concatenate((existing, data), axis=0)
            np.save(save_path, data)
            print(f"Saved one '{label}' sample.")
            sequence = []

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
