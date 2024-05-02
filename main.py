import numpy as np
import cv2
import mediapipe as mp

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
    try:
        wrist_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_WRIST].y
        elbow_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ELBOW].y
        return wrist_y < elbow_y
    except Exception as e:
        print(f"Erro detectado em: Left punch: {e}")
    return False

def detect_right_punch(landmarks):
    try:
        wrist_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_WRIST].y
        elbow_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ELBOW].y
        return wrist_y < elbow_y
    except Exception as e:
        print(f"Erro detectado em: Right punch: {e}")
    return False

def detect_left_kick(landmarks):
    try:
        knee_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_KNEE].y
        hip_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y
        return knee_y < hip_y
    except Exception as e:
        print(f"Erro detectado em: Left kick: {e}")
    return False

def detect_jump(landmarks):
    try:
        left_ankle_y = landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE].y
        right_ankle_y = landmarks.landmark[mp_pose.PoseLandmark.RIGHT_ANKLE].y
        hip_y = (landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP].y +
                 landmarks.landmark[mp_pose.PoseLandmark.RIGHT_HIP].y) / 2
        return left_ankle_y < hip_y and right_ankle_y < hip_y
    except Exception as e:
        print(f"Erro detectado em: Jump: {e}")
    return False

def handle_detection_results(frame, results):
    try:
        if results.pose_landmarks:
            annotated_image = draw_landmarks_on_image(frame, results.pose_landmarks)
            if detect_left_punch(results.pose_landmarks):
                print("Soco esquerdo detectado!")
            if detect_right_punch(results.pose_landmarks):
                print("Soco direito detectado!")
            if detect_left_kick(results.pose_landmarks):
                print("Chute esquerdo detectado!")
            if detect_jump(results.pose_landmarks):
                print("Pulo detectado!")
            cv2.imshow('Projeto Street Fighter EX2', annotated_image)
            cv2.waitKey(2)
    except Exception as e:
        print(f"Erro: {e}")

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 2560)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1440)

screen_size = [1600, 1000]

try:
    with mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=False, min_detection_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()

            frame = cv2.resize(frame, (screen_size[0] - 40, screen_size[1] - 120))

            cv2.line(frame, (screen_size[0] // 2, 0), (screen_size[0] // 2, screen_size[1]), (0, 0, 255), 5)
            cv2.putText(frame, "Jogador 1", (screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(frame, "Jogador 2", (3 * screen_size[0] // 4, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2, cv2.LINE_AA)

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            handle_detection_results(frame, results)

            if cv2.waitKey(5) & 0xFF == ord('-'):
                break
            
except Exception as e:
    print(f"Erro: {e}")
finally:
    cap.release()
    cv2.destroyAllWindows()
