import mediapipe as mp

class Player:
    pose :  mp.solutions.pose

    #construtor
    def __init__(self, pose):
        self.pose = pose
        