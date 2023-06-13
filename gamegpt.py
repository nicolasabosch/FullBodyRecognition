#!/usr/bin/env python
# -*- coding: utf-8 -*-
import csv
import copy
import argparse
import itertools
from collections import Counter
from collections import deque


import cv2 as cv
import numpy as np
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

from utils import CvFpsCalc
from model import KeyPointClassifier
from model import PointHistoryClassifier

def read_csv(csv_path):
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        labels = next(reader)
        data = []
        for row in reader:
            data.append([float(x) for x in row])

    return labels, np.array(data, dtype=np.float32)

def select_mode(key, mode):
    number = -1
    if 48 <= key <= 57:  # 0 ~ 9
        number = key - 48
    if key == 110:  # n
        mode = 0
    if key == 107:  # k
        mode = 1
    if key == 104:  # h
        mode = 2
    return number, mode

def main():
    # 引数解析 #################################################################
    args = get_args()

    cap_device = args.device
    cap_width = args.width
    cap_height = args.height

    use_static_image_mode = args.use_static_image_mode
    min_detection_confidence = args.min_detection_confidence
    min_tracking_confidence = args.min_tracking_confidence

    use_brect = True

    # カメラ準備 ###############################################################
    cap = cv.VideoCapture(cap_device)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, cap_width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, cap_height)

    # モデルロード #############################################################
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(
        static_image_mode=use_static_image_mode,
        max_num_hands=1,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    keypoint_classifier = KeyPointClassifier()

    point_history_classifier = PointHistoryClassifier()

    # ラベル読み込み ###########################################################
    with open('model/keypoint_classifier/keypoint_classifier_label.csv',
              encoding='utf-8-sig') as f:
        keypoint_classifier_labels = csv.reader(f)
        keypoint_classifier_labels = [
            row[0] for row in keypoint_classifier_labels
        ]
    with open(
            'model/point_history_classifier/point_history_classifier_label.csv',
            encoding='utf-8-sig') as f:
        point_history_classifier_labels = csv.reader(f)
        point_history_classifier_labels = [
            row[0] for row in point_history_classifier_labels
        ]

    # FPS計測モジュール ########################################################
    cvFpsCalc = CvFpsCalc(buffer_len=10)

    # 座標履歴 #################################################################
    history_length = 16
    point_history = deque(maxlen=history_length)

    # フィンガージェスチャー履歴 ################################################
    finger_gesture_history = deque(maxlen=history_length)

    #  ########################################################################
    mode = 0

    while True:
        fps = cvFpsCalc.get()

        # キー処理(ESC：終了) #################################################
        key = cv.waitKey(10)
        if key == 27:  # ESC
            break
        number, mode = select_mode(key, mode)

        # カメラキャプチャ #####################################################
        ret, image = cap.read()
        if not ret:
            break
        image = cv.flip(image, 1)  # ミラー表示

        # 画像処理 #############################################################
        image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = hands.process(image)

        image.flags.writeable = True
        image = cv.cvtColor(image, cv.COLOR_RGB2BGR)

        # 描画 ################################################################
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks:
                # 検出結果をリストに格納
                landmark_list = []
                for i, landmark in enumerate(hand_landmarks.landmark):
                    landmark_list.append({
                        'x':
                        landmark.x,
                        'y':
                        landmark.y,
                        'z':
                        landmark.z,
                        'visibility':
                        landmark.visibility,
                    })

                # 指の座標算出
                finger_coord_list = []
                for finger_i, finger_meta in enumerate(
                        mp_hands.HandLandmark):

                    if finger_i == 0:  # 手首座標
                        continue

                    landmark = hand_landmarks.landmark[finger_meta]
                    landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                        landmark.x, landmark.y, cap_width, cap_height)

                    finger_coord_list.append({
                        'x':
                        landmark_px[0],
                        'y':
                        landmark_px[1],
                    })

                # 親指CM算出
                parent_thumb = mp_hands.HandLandmark.THUMB_CMC.value
                landmark = hand_landmarks.landmark[parent_thumb]
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                    landmark.x, landmark.y, cap_width, cap_height)
                thumb_cmc_px = {
                    'x':
                    landmark_px[0],
                    'y':
                    landmark_px[1],
                }

                # 手の平CM算出
                palm_list = [
                    mp_hands.HandLandmark.WRIST,
                    mp_hands.HandLandmark.THUMB_CMC,
                    mp_hands.HandLandmark.PINKY_MCP,
                ]
                palm_x = 0
                palm_y = 0
                palm_z = 0
                for palm_landmark in palm_list:
                    landmark = hand_landmarks.landmark[palm_landmark]
                    palm_x += landmark.x
                    palm_y += landmark.y
                    palm_z += landmark.z
                palm_x /= len(palm_list)
                palm_y /= len(palm_list)
                palm_z /= len(palm_list)
                landmark_px = mp_drawing._normalized_to_pixel_coordinates(
                    palm_x, palm_y, cap_width, cap_height)
                palm_cm_px = {
                    'x':
                    landmark_px[0],
                    'y':
                    landmark_px[1],
                }

                # モードによる処理分岐 ##########################################################
                if mode == 0:
                    point_history.append(copy.deepcopy(finger_coord_list))

                    # 最も出現回数の多いジェスチャーを取得
                    flatten_history = list(itertools.chain.from_iterable(point_history))
                    finger_gesture, _ = Counter(flatten_history).most_common(1)[0]
                    finger_gesture_history.append(finger_gesture)

                    # ジェスチャーを推定
                    if len(finger_gesture_history) == finger_gesture_history.maxlen:
                        gesture_count = Counter(finger_gesture_history)
                        gesture = gesture_count.most_common(1)[0][0]

                        gesture_name = keypoint_classifier_labels[gesture]
                        if gesture_name == 'none':
                            pass
                        else:
                            cv.putText(image, gesture_name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                                       1.0, (0, 255, 0), thickness=2)

                elif mode == 1:
                    # キーポイントの推定
                    keypoint_index = keypoint_classifier.inference(
                        [finger_coord_list])[0]

                    keypoint_name = keypoint_classifier_labels[keypoint_index]
                    cv.putText(image, keypoint_name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 255, 0), thickness=2)

                elif mode == 2:
                    point_history_classifier_index = point_history_classifier.inference(
                        [list(itertools.chain.from_iterable(point_history))])[0]

                    gesture_name = point_history_classifier_labels[point_history_classifier_index]
                    cv.putText(image, gesture_name, (10, 30), cv.FONT_HERSHEY_SIMPLEX,
                               1.0, (0, 255, 0), thickness=2)

                # 検出手の描画 ####################################################
                if use_brect:
                    brect = calc_bounding_rect(cv, hand_landmarks.landmark)
                    cv.rectangle(image, brect[0], brect[1], (0, 255, 0), 2, cv.LINE_8)

                # 手の平描画 ######################################################
                cv.circle(image, (int(palm_cm_px['x']), int(palm_cm_px['y'])) , 5, (0, 0, 255), 2, cv.LINE_8)

                # 親指CM描画 ######################################################
                cv.circle(image, (int(thumb_cmc_px['x']), int(thumb_cmc_px['y'])), 5, (0, 0, 255), 2, cv.LINE_8)

                # 指の座標描画 ####################################################
                for finger_i, finger_coord in enumerate(finger_coord_list):
                    cv.putText(image, str(finger_i), (int(finger_coord['x']), int(finger_coord['y'])),
                               cv.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2, cv.LINE_AA)
                    cv.circle(image, (int(finger_coord['x']), int(finger_coord['y'])), 5, (0, 0, 255), 2, cv.LINE_8)

                # キーポイント描画 #################################################
                for i, landmark in enumerate(hand_landmarks.landmark):
                    x = min(int(landmark.x * cap_width), cap_width - 1)
                    y = min(int(landmark.y * cap_height), cap_height - 1)
                    if landmark.visibility > 0:
                        cv.circle(image, (x, y), 5, (0, 255, 0), 2, cv.LINE_8)
                    else:
                        cv.circle(image, (x, y), 5, (0, 0, 255), 2, cv.LINE_8)

        # 情報描画 #################################################################
        # FPS
        cv.putText(image, "FPS:" + str(round(fps, 2)), (10, 10),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

        # モード
        mode_str = ['Normal', 'Keypoint', 'History']
        cv.putText(image, "Mode:" + mode_str[mode], (10, 50),
                   cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow('MediaPipe Hand Demo', image)

    cap.release()
    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
