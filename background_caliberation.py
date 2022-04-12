import cv2
import math
import numpy as np
import skimage.measure   
import mediapipe as mp
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

pose= mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2,enable_segmentation=True)
cap= cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    results = pose.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        red_img = np.zeros_like(frame, dtype=np.uint8)
        red_img[:, :] = (255,255,255)
        segm_2class = 0.2 + 0.8 * results.segmentation_mask
        segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
        mask=((1 - segm_2class)[:,:,0]> 0.2)
        mask=mask*255
        mask=mask.astype('uint8')
        mask=cv2.bitwise_not(mask)
        kernel=np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations =4)
        mask=cv2.bitwise_not(mask)
        edges = cv2.Canny(frame,100,200)
        maskedimage=cv2.bitwise_and(edges,edges,mask=mask)
        count = np.count_nonzero(maskedimage)
        zeros=maskedimage.size
        ratio=(count/zeros)*100 #<0.1
        #print enthropy on farme using cv2
        maskedimage=cv2.putText(maskedimage, str(ratio), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("edges", edges)
        cv2.imshow("frame", frame)
        cv2.imshow("maskedimage", maskedimage)
        cv2.imshow("mask", mask)
        cv2.waitKey(1)