from typing import List
import numpy as np
import cv2
import mediapipe as mp


def preprocessing_train_data(data):
    print(data)
    output = []
    for datum in data:
        norm_data = np.array(data)
        label = np.array(datum["Label"])
        norm_data = norm_data - norm_data[0]  # Keypoints 0からの相対ベクトルを求める

        norm_data = data[1:].reshape(-1)  # Keypoints 0を除外する
        norm_data = norm_data / np.max(np.abs(norm_data))  # 絶対値最大値で正規化
        label = label.reshape(-1)

        processed_data = np.concatenate(
            [left_data, right_data, distance_each_data, label], axis=0
        )
        output.append(processed_data)
    output = np.array(output)  # dim: (data_size, 122)
    processed_data, label = output[:, :-1], output[:, [-1]]
    return processed_data, label


def preprocessing_gesture_data(data: List[dict]):
    # gestureのみの処理
    output = []
    for datum in data:
        left_data = np.array(datum["Left"])
        right_data = np.array(datum["Right"])
        distance_each_data = (left_data - right_data).reshape(
            -1
        )  # 右手左手のkeypointsの差分を取る
        distance_each_data = distance_each_data / np.max(
            np.abs(distance_each_data)
        )  # 絶対値最大値で正規化

        left_data = left_data - left_data[0]  # Keypoints 0からの相対ベクトルを求める
        right_data = right_data - right_data[0]  # Keypoints 0からの相対ベクトルを求める

        left_data = left_data[1:].reshape(-1)  # Keypoints 0を除外する
        left_data = left_data / np.max(np.abs(left_data))  # 絶対値最大値で正規化
        right_data = right_data[1:].reshape(-1)
        right_data = right_data / np.max(np.abs(right_data))  # 最大値で正規化

        processed_data = np.concatenate(
            [left_data, right_data, distance_each_data], axis=0
        )
        output.append(processed_data)
    output = np.array(output)  # dim: (data_size, 122)
    return output

# ここは直さないといけない。Pose用にする。
landmark_line_ids = [
    (0, 4),
    (4, 5),
    (5, 6),
    (6, 8),
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 7),
    (9,10), # mouth
    (11, 12),
    (12,14), # right arm
    (14, 16),
    (16,22),
    (16, 18),
    (18, 20),
    (11, 13), # left arm
    (13, 15),
    (15, 21),
    (15, 17),
    (17, 19),
    (15, 19),
    (12, 24),
    (23, 24),
    (11, 23),
    (24, 26), # right foot
    (26, 28),
    (28, 30),
    (30, 32),
    (28, 32),
    (23, 25), # left foot
    (25, 27),
    (27, 29),
    (27, 31),
    (29, 31)   
]


def draw_keypoints_line(
    results,
    img,
):
    
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            img,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=4),
            mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2)
        )

    """
    # 検出した手の数分繰り返し
    img_h, img_w, _ = img.shape  # サイズ取得
    for h_id, pose_landmarks in enumerate(results.pose_landmarks):
        # landmarkの繋がりをlineで表示
        for line_id in landmark_line_ids:
            print(line_id)
            # 1点目座標取得
            lm = pose_landmarks.landmark[line_id[0]]
            lm_pos1 = (int(lm.x * img_w), int(lm.y * img_h))
            # 2点目座標取得
            lm = pose_landmarks.landmark[line_id[1]]
            lm_pos2 = (int(lm.x * img_w), int(lm.y * img_h))
            # line描画
            cv2.line(img, lm_pos1, lm_pos2, (128, 0, 0), 1)

            # landmarkをcircleで表示
            z_list = [lm.z for lm in pose_landmarks.landmark]
            z_min = min(z_list)
            z_max = max(z_list)
            for lm in pose_landmarks.landmark:
                lm_pos = (int(lm.x * img_w), int(lm.y * img_h))
                lm_z = int((lm.z - z_min) / (z_max - z_min) * 255)
                cv2.circle(img, lm_pos, 3, (255, lm_z, lm_z), -1)

                # 検出情報をテキスト出力
                # - テキスト情報を作成
                hand_texts = []
                for c_id, hand_class in enumerate(
                    results.multi_handedness[h_id].classification
                ):
                    hand_texts.append("#%d-%d" % (h_id, c_id))
                    hand_texts.append("- Index:%d" % (hand_class.index))
                    hand_texts.append("- Label:%s" % (hand_class.label))
                    hand_texts.append(
                        "- Score:%3.2f" % (hand_class.score * 100)
                    )
                    # - テキスト表示に必要な座標など準備
                    lm = hand_landmarks.landmark[0]
                    lm_x = int(lm.x * img_w) - 50
                    lm_y = int(lm.y * img_h) - 10
                    lm_c = (64, 0, 0)
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    # - テキスト出力
                    for cnt, text in enumerate(hand_texts):
                        cv2.putText(
                            img,
                            text,
                            (lm_x, lm_y + 10 * cnt),
                            font,
                            0.3,
                            lm_c,
                            1,
                        )
                """
