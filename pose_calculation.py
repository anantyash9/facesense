import cv2
import math
import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose

pose= mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)

def calculate_YZ(image):
    results = pose.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        y=abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].y - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y)
        z=abs(abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].z) - abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z))
        return y,z
    else:
        raise Exception("No pose landmarks detected")
