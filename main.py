import numpy as np
import cv2
import mediapipe as mp
import threading
import queue
import pyautogui
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import time


driver = webdriver.Edge();

url = "https://archive.org/details/arcade_sfex2#"
driver.get(url)
driver.maximize_window()

botao_ligar = driver.find_element(By.CLASS_NAME,"ghost")
botao_ligar.click()

def countdown(seconds):
    for i in range(seconds, -1, -1):
        print(i)
        time.sleep(1)

countdown(60)

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
    shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER].y
    return wrist_y < shoulder_y

def detect_right_punch(landmarks):
    wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
    shoulder_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER].y
    return wrist_y < shoulder_y

def detect_left_kick(landmarks):
    knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
    hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
    return knee_y < hip_y

def detect_right_kick(landmarks):
    knee_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_KNEE].y
    hip_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y
    return knee_y < hip_y

def detect_flip(landmarks):
    left_wrist_x = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].x
    right_wrist_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].x
    left_elbow_x = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].x
    right_elbow_x = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].x
    
    return left_wrist_x < right_elbow_x and right_wrist_x > left_elbow_x

def detect_special(landmarks):

    return 

screen_flip_p1 = False
screen_flip_p2 = False

def process_frame(frame, pose, results_queue, player_label):
    results = pose.process(frame)
    detections = []
    movement_detected = False

    global screen_flip_p1
    global screen_flip_p2

    if results.pose_landmarks:
        annotated_image = draw_landmarks_on_image(frame, results.pose_landmarks)
        if detect_left_punch(results.pose_landmarks):
            detections.append("Soco esquerdo detectado")
            pyautogui.press('space' if player_label == "Jogador 1" else 'q')
            movement_detected = True
            
        if detect_right_punch(results.pose_landmarks):
            detections.append("Soco direito detectado")
            pyautogui.press('ctrl' if player_label == "Jogador 1" else 'a')
            movement_detected = True
            
        if detect_left_kick(results.pose_landmarks):
            detections.append("Chute esquerdo detectado")
            pyautogui.press('z' if player_label == "Jogador 1" else 'w')
            movement_detected = True
            
        if detect_right_kick(results.pose_landmarks):
            detections.append("Chute direito detectado")
            pyautogui.press('x' if player_label == "Jogador 1" else 'e')
            movement_detected = True
            
        if detect_flip(results.pose_landmarks):
            detections.append("Flip detectada")
            if player_label == 'Jogador 1' and not screen_flip_p1:
                screen_flip_p1 = True
            elif player_label == 'Jogador 1' and screen_flip_p1:
                screen_flip_p1 = False
            if player_label == 'Jogador 2' and not screen_flip_p2:
                screen_flip_p2 = True
            elif player_label == 'Jogador 2' and screen_flip_p2:
                screen_flip_p2 = False
            movement_detected = True
            
        if detect_special(results.pose_landmarks):
            detections.append("Habilidade especial detectada")
            movement_detected = True
            
        if not movement_detected:
            detections.append("Nenhuma ação realizada")
            if player_label == "Jogador 1" and not screen_flip_p1:
                pyautogui.press('left')
            elif player_label == "Jogador 1" and screen_flip_p1:
                pyautogui.press('right')
            if player_label == "Jogador 2" and not screen_flip_p2:
                pyautogui.press('d')
            elif player_label == "Jogador 2" and screen_flip_p2:
                pyautogui.press('g')
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
