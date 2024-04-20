import numpy as np
import cv2
import threading as thr
import mediapipe as mp
from matplotlib import pyplot as plt
from classes.Player import Player
import ctypes

u32 = ctypes.windll.user32
screen_size = u32.GetSystemMetrics(0), u32.GetSystemMetrics(1)

print(screen_size)

player1 = Player(True, False)
player2 = Player(False, True)

mp_drawing = mp.solutions.drawing_utils
mp_holistic = mp.solutions.holistic

def detect_punch(hand_landmarks):

    if hand_landmarks:
        thumb_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.THUMB_TIP]
        index_finger_tip = hand_landmarks.landmark[mp_holistic.HandLandmark.INDEX_FINGER_TIP]
        distance = abs(thumb_tip.x - index_finger_tip.x) + abs(thumb_tip.y - index_finger_tip.y)

        if distance > 0.1:
            return True
    return False

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, screen_size[0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, screen_size[1])

with mp_holistic.Holistic(min_detection_confidence=0.5, min_tracking_confidence=0.5) as holistic:
    
    while cap.isOpened():
        ret, frame = cap.read()
        frame = cv2.resize(frame, (screen_size[0] - 40, screen_size[1] - 120))
        
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
        
        if detect_punch(results.right_hand_landmarks):
            print("Soco detectado")

        cv2.imshow('Projeto Street Fighter Ex2', image)

        if cv2.waitKey(10) & 0xFF == ord('-'):
            break

cap.release()
cv2.destroyAllWindows()
