import cv2
import math
import numpy as np

import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

pose= mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2)
cap= cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        print("y difference :", abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].y - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y))#0.04
        print("z difference :", abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EAR].z - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EAR].z)) #0.2
        y=abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].y - results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].y)
        z=abs(abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_EYE_OUTER].z) - abs(results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_EYE_OUTER].z))
        if y>0.03:
            #draw a green circle on frame using cv2
            frame=cv2.circle(frame, (120, 50), 40,(255, 0, 0),10)
        if z>0.015:
            #draw a green circle on frame using cv2
            frame=cv2.circle(frame, (120, 120), 40,(0, 0, 255),10)
        mp_drawing.draw_landmarks(
                frame,
                results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())
        cv2.imshow("image", frame)
        cv2.waitKey(1)