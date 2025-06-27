 
import mediapipe as mp
import numpy as np
import os
import cv2

if (len(sys.argv) == 2):
    print("コマンド <動画ファイル> <ラベル名>")
label = sys.argv[2]  # <-- ここを変更して別のジェスチャーを収集　ここを15個*5セット程度は用意しないと。
seq_length = 30
save_path = f"data/{label}.npy"
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
cap = cv2.VideoCapture(0)   # 0をファイル名に変えればファイルから動画を読み込む
sequence = []
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image)
    
    if results.pose_landmarks:
        keypoints = []
        for lm in results.pose_landmarks.landmark:
            keypoints.extend([lm.x, lm.y, lm.z])
        sequence.append(keypoints)
    
        if len(sequence) == seq_length:
            data = np.expand_dims(sequence, axis=0)  # (1, 30, 99)
            if os.path.exists(save_path):
                existing = np.load(save_path) # 元々のファイルに入っていたもの
                data = np.concatenate((existing, data), axis=0) # それに追加する。
            np.save(save_path, data)
            print(f"Saved one {label} sample.")
            sequence = []
    
    cv2.imshow("Collecting", frame)
    if cv2.waitKey(1) == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()