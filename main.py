import numpy as np
import cv2
import threading as thr
import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2
from mediapipe.python.solutions.drawing_utils import draw_landmarks
from mediapipe.python.solutions.drawing_styles import get_default_pose_landmarks_style
from matplotlib import pyplot as plt
from classes.Player import Player
import math
import pyautogui

PoseLandmarkerResult = mp.tasks.vision.PoseLandmarkerResult

def draw_landmarks_on_image(rgb_image, detection_result):
    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for idx in range(len(pose_landmarks_list)):
        pose_landmarks = pose_landmarks_list[idx]

        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(
                x=landmark.x,
                y=landmark.y,
                z=landmark.z) for landmark in pose_landmarks
        ])
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

to_window = None
last_timestamp_ms = 0

def print_result(detection_result: vision.PoseLandmarkerResult, output_image: mp.Image, timestamp_ms: int):
    global to_window
    global last_timestamp_ms
    if timestamp_ms < last_timestamp_ms:
        return
    last_timestamp_ms = timestamp_ms
    to_window = draw_landmarks_on_image(output_image.numpy_view(), detection_result)

base_options = BaseOptions(model_asset_path='./tasks/pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    num_poses=2,
    min_pose_detection_confidence=0.5,
    min_pose_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    base_options=base_options,
    running_mode=vision.RunningMode.LIVE_STREAM,
    output_segmentation_masks=False,
    result_callback=print_result)

## COMANDOS DE AÇÃO

def detect_left_punch(pose_landmarks):
    if pose_landmarks:
        left_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
        left_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]

        if left_wrist.y < left_elbow.y:
            return True
    return False

def detect_right_punch(pose_landmarks):
    if pose_landmarks:
        right_wrist = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
        right_elbow = pose_landmarks.landmark[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]

        if right_wrist.y < right_elbow.y:
            return True
    return False

screen_size = [1600, 1200]

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440) 

try:
    with vision.PoseLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            available_frame, frame = cap.read()

            if not available_frame:
                break

            frame = cv2.resize(frame, (screen_size[0] - 40, screen_size[1] - 120))

            cv2.line(frame, (screen_size[0] // 2, 0), (screen_size[0] // 2, screen_size[1]), (0, 0, 255), 5)
            cv2.putText(frame, "Jogador 1", (screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Jogador 2", (3 * screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame)

            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            landmarker.detect_async(mp_image, timestamp_ms)

            # if detect_punch(results.right_hand_landmarks, results.pose_landmarks):
            #     print("Soco 1")
            #     print("1")
            #pyautogui.press('ctrl')

            # if detect_open_hand(results.right_hand_landmarks) and detect_open_hand(results.left_hand_landmarks):
            #     print("Soco 2")
            #     print("2")
            #pyautogui.press('alt')

            # if detect_kick(results.pose_landmarks):
            #     print("Chute baixo")
            #     print("3")
            #pyautogui.press('shift')

            # if detect_jump(results.pose_landmarks):
            #     print("Pulo detectado")
            #     print("4")
            #pyautogui.press('up')

            if to_window is not None:
                cv2.imshow('Projeto Street Fighter Ex2', to_window)

            if cv2.waitKey(5) & 0xFF == ord('-'):
                break
except Exception as e:
    print(f"Deu erro: {e}")

cap.release()
cv2.destroyAllWindows()
