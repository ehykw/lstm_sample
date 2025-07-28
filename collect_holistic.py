import mediapipe as mp
import numpy as np
import os
import cv2
import sys

if len(sys.argv) < 3:
    print("python collect.py <動画ファイル> <ラベル名>")
    sys.exit(1)

label = sys.argv[2]
seq_length = 30
save_path = f"data/{label}.npy"

mp_holistic = mp.solutions.holistic
holistic = mp_holistic.Holistic(static_image_mode=False)

cap = cv2.VideoCapture(sys.argv[1])
sequence = []

def normalize_keypoints(keypoints):
    keypoints = np.array(keypoints).reshape(-1, 3)
    if len(keypoints) == 0:
        return []
    
    # 基準点を肩中心とする（Poseがある場合のみ）
    try:
        left_shoulder = keypoints[11]
        right_shoulder = keypoints[12]
        center = (left_shoulder + right_shoulder) / 2
        scale = np.linalg.norm(left_shoulder - right_shoulder)
    except:
        center = np.mean(keypoints, axis=0)
        scale = np.linalg.norm(np.std(keypoints, axis=0))

    keypoints -= center
    if scale > 0:
        keypoints /= scale

    return keypoints.flatten().tolist()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = holistic.process(image)

    keypoints = []

    # 姿勢（Pose）
    if results.pose_landmarks:
        for lm in results.pose_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([[0, 0, 0]] * 33)

    # 左手
    if results.left_hand_landmarks:
        for lm in results.left_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([[0, 0, 0]] * 21)

    # 右手
    if results.right_hand_landmarks:
        for lm in results.right_hand_landmarks.landmark:
            keypoints.append([lm.x, lm.y, lm.z])
    else:
        keypoints.extend([[0, 0, 0]] * 21)

    keypoints = normalize_keypoints(keypoints)
    sequence.append(keypoints)

    if len(sequence) == seq_length:
        data = np.expand_dims(sequence, axis=0)
        if os.path.exists(save_path):
            existing = np.load(save_path)
            data = np.concatenate((existing, data), axis=0)
        np.save(save_path, data)
        print(f"Saved one {label} sample.")
        sequence = []

    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
