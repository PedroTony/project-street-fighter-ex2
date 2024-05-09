import numpy as np
import cv2
import mediapipe as mp
import threading
import queue
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.by import By


# driver = webdriver.Edge();

# url = "https://archive.org/details/arcade_sfex2#"
# driver.get(url)
# driver.maximize_window()


# botaoLigar = driver.find_element(By.CLASS_NAME,"ghost")
# botaoLigar.click()

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def draw_landmarks_on_image(rgb_image, landmarks):
    annotated_image = np.copy(rgb_image)
    mp_drawing.draw_landmarks(
        annotated_image,
        landmarks,
        mp_pose.POSE_CONNECTIONS,
        mp_drawing_styles.get_default_pose_landmarks_style())
    return annotated_image

def detect_left_punch(landmarks):
    wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
    elbow_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
    return wrist_y < elbow_y

def detect_right_punch(landmarks):
    wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
    elbow_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
    return wrist_y < elbow_y

def detect_left_kick(landmarks):
    knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
    hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    return knee_y < hip_y

def detect_right_kick(landmarks):
    knee_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
    hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
    return knee_y < hip_y

def detect_jump(landmarks):
    left_ankle_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
    right_ankle_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
    hip_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y + landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
    return left_ankle_y < hip_y and right_ankle_y < hip_y

def process_frame(frame, pose, results_queue, player_label):
    
    results = pose.process(frame)
    if results.pose_landmarks:
        annotated_image = draw_landmarks_on_image(frame, results.pose_landmarks)
        detections = []
        if detect_left_punch(results.pose_landmarks):
            detections.append("Left punch detected")
            pyautogui.press('space' if player_label == "Jogador 1" else 'q')
        if detect_right_punch(results.pose_landmarks):
            detections.append("Right punch detected")
            pyautogui.press('ctrl' if player_label == "Jogador 1" else 'a')
        if detect_left_kick(results.pose_landmarks):
            detections.append("Left kick detected")
            pyautogui.press('z' if player_label == "Jogador 1" else 'w')
        if detect_right_kick(results.pose_landmarks):
            detections.append("Right Kick detected")
            pyautogui.press('x' if player_label == "Jogador 1" else 'e')
        if detect_jump(results.pose_landmarks):
            detections.append("Jump detected")
            pyautogui.press('up' if player_label == "Jogador 1" else 'r')
        results_queue.put((annotated_image, detections, player_label))

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)
pose1 = mp_pose.Pose(min_detection_confidence=0.5)
pose2 = mp_pose.Pose(min_detection_confidence=0.5)
screen_size = [1600, 1000]
results_queue = queue.Queue()

try:
    while cap.isOpened():
        ret, frame = cap.read()

        frame = cv2.resize(frame, (screen_size[0] - 40, screen_size[1] - 120))
        left_frame = frame[:, :screen_size[0]//2]
        right_frame = frame[:, screen_size[0]//2:]

        threading.Thread(target=process_frame, args=(left_frame, pose1, results_queue, "Jogador 1")).start()
        threading.Thread(target=process_frame, args=(right_frame, pose2, results_queue, "Jogador 2")).start()

        while not results_queue.empty():
            annotated_image, detections, player_label = results_queue.get()
            for detection in detections:
                print(f"{player_label}: {detection}")
            cv2.imshow(f'Projeto Street Fighter EX2 - {player_label}', annotated_image)

        if cv2.waitKey(5) & 0xFF == 27: 
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
