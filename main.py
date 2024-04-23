import numpy as np
import cv2
import threading as thr
import mediapipe as mp
from matplotlib import pyplot as plt
from classes.Player import Player
import ctypes
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import math
import pyautogui

u32 = ctypes.windll.user32
screen_size = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)

print(screen_size)

player1 = Player(True, False)
player2 = Player(False, True)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

#Detectando chute
def detect_kick(body_landmarks):
    if body_landmarks:
        left_knee = body_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]
        right_knee = body_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]
        left_ankle = body_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE]
        right_ankle = body_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]

        knee_height = abs(left_knee.y - right_knee.y)
        ankle_height = abs(left_ankle.y - right_ankle.y)


        knee_threshold = 0.15
        ankle_threshold = 0.15

        if knee_height > knee_threshold and ankle_height > ankle_threshold:
            return True

    return False


#Detectando soco2
def detect_open_hand(hand_landmarks):
    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]


        thumb_distance = calculate_distance(thumb_tip, index_finger_tip)
        index_distance = calculate_distance(index_finger_tip, middle_finger_tip)
        middle_distance = calculate_distance(middle_finger_tip, ring_finger_tip)
        ring_distance = calculate_distance(ring_finger_tip, pinky_tip)

        alignment_threshold = 0.5


        if (0.05 < thumb_distance < alignment_threshold and
            0.05 < index_distance < alignment_threshold and
            0.05 < middle_distance < alignment_threshold and
            0.05 < ring_distance < alignment_threshold):
            return True

    return False

def calculate_distance(point1, point2):
    return abs(point1.x - point2.x) + abs(point1.y - point2.y)

# Detectando pulo
def detect_jump(body_landmarks):
    if body_landmarks:
        left_knee = body_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_KNEE]
        right_knee = body_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_KNEE]
        left_ankle = body_landmarks.landmark[mp_holistic.PoseLandmark.LEFT_ANKLE]
        right_ankle = body_landmarks.landmark[mp_holistic.PoseLandmark.RIGHT_ANKLE]

        knee_height = abs(left_knee.y - right_knee.y)
        ankle_height = abs(left_ankle.y - right_ankle.y)

        knee_threshold = 0.02
        ankle_threshold = 0.02


        if knee_height > knee_threshold and right_knee.y < ankle_threshold and left_knee.y < ankle_threshold:
            return True

    return False


#Detectando soco 
def detect_punch(hand_landmarks, body_landmarks):
    if hand_landmarks and body_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        middle_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.MIDDLE_FINGER_TIP]
        ring_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.RING_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.PINKY_TIP]

        thumb_distance = abs(thumb_tip.x - index_finger_tip.x) + abs(thumb_tip.y - index_finger_tip.y)
        index_distance = abs(index_finger_tip.x - middle_finger_tip.x) + abs(index_finger_tip.y - middle_finger_tip.y)
        middle_distance = abs(middle_finger_tip.x - ring_finger_tip.x) + abs(middle_finger_tip.y - ring_finger_tip.y)
        ring_distance = abs(ring_finger_tip.x - pinky_tip.x) + abs(ring_finger_tip.y - pinky_tip.y)

        thumb_index_limit = 0.09

        vertical_alignment_threshold = 0.15  
        vertical_alignment = abs(thumb_tip.y - index_finger_tip.y) < vertical_alignment_threshold
        
        # Verifica se a mão está apontando para a frente e fechada
        hand_pointing_forward = thumb_tip.x < index_finger_tip.x and \
                         pinky_tip.x < ring_finger_tip.x


        if (thumb_distance < thumb_index_limit and index_distance < thumb_index_limit and
           middle_distance < thumb_index_limit and ring_distance < thumb_index_limit and
           vertical_alignment and hand_pointing_forward):
            return True

    return False







cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_size[1])

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (screen_size[0] - 40, screen_size[1] - 120))

        cv2.line(frame, (screen_size[0] // 2, 0), (screen_size[0] // 2, screen_size[1]), (0, 0, 255), 5)
        cv2.putText(frame, "Jogador 1", (screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, "Jogador 2", (3 * screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
        
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(image)
        # print(results.face_landmarks)

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # mp_drawing.draw_landmarks(image, results.face_landmarks, mp_holistic.FACEMESH_TESSELATION,
        #                          mp_drawing.DrawingSpec(color=(80,110,10), thickness=1, circle_radius=1),
        #                          mp_drawing.DrawingSpec(color=(80,256,121), thickness=1, circle_radius=1)
        #                          )
        
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(80,22,10), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(80,44,121), thickness=2, circle_radius=2)
                                 )


        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(121,22,76), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(121,44,250), thickness=2, circle_radius=2)
                                 )


        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS, 
                                 mp_drawing.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=4),
                                 mp_drawing.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                                 )
        
        if detect_punch(results.right_hand_landmarks, results.pose_landmarks):
            print("Soco 1")
            #pyautogui.press('ctrl')

        if detect_open_hand(results.right_hand_landmarks) and detect_open_hand(results.left_hand_landmarks):
            print("Soco 2")
            #pyautogui.press('alt')

        if detect_kick(results.pose_landmarks):
            print("Chute baixo")
            #pyautogui.press('shift')

        if detect_jump(results.pose_landmarks):
            print("Pulo detectado")
            #pyautogui.press('up')

        cv2.imshow('Projeto Street Fighter Ex2', image)

        if cv2.waitKey(10) & 0xFF == ord('-'):
            break

cap.release()
cv2.destroyAllWindows()
