import numpy as np
import cv2
import threading as thr
import mediapipe as mp
from matplotlib import pyplot as plt
from classes.Player import Player

player1 = Player(True, False)
player2 = Player(False, True)

print('{}, {}'.format(player1.left, player1.right))
print('{}, {}'.format(player2.left, player2.right))

