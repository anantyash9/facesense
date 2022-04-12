import cv2
import math
import numpy as np  
import mediapipe as mp
mp_pose = mp.solutions.pose

pose= mp_pose.Pose(
    static_image_mode=True, min_detection_confidence=0.5, model_complexity=2,enable_segmentation=True)

def calculate_buzz(img):
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        red_img = np.zeros_like(img, dtype=np.uint8)
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
        edges = cv2.Canny(img,100,200)
        maskedimage=cv2.bitwise_and(edges,edges,mask=mask)
        count = np.count_nonzero(maskedimage)
        zeros=maskedimage.size
        ratio=(count/zeros)*100 #<0.1
        return ratio
    else:
        raise Exception("No pose landmarks detected")

def calculate_color_ratio(img,hsv_lower,hsv_upper):
    results = pose.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    if results.pose_landmarks:
        segm_2class = 0.2 + 0.8 * results.segmentation_mask
        segm_2class = np.repeat(segm_2class[..., np.newaxis], 3, axis=2)
        mask=((1 - segm_2class)[:,:,0]> 0.2)
        mask=mask*255
        mask=mask.astype('uint8')
        mask=cv2.bitwise_not(mask)
        kernel=np.ones((5,5),np.uint8)
        mask = cv2.dilate(mask,kernel,iterations =4)
        mask=cv2.bitwise_not(mask)

            # Set minimum and max HSV values to display
        lower = np.array(hsv_lower, dtype=np.uint8)
        upper = np.array(hsv_upper, dtype=np.uint8)

        # Create HSV Image and threshold into a range.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        color_mask = cv2.inRange(hsv, lower, upper)
        color_mask = cv2.bitwise_not(color_mask)
        maskedimage = cv2.bitwise_and(color_mask,color_mask, mask= mask)


        # maskedimage=cv2.bitwise_and(img,img,mask=mask)
        count = np.count_nonzero(maskedimage)
        zeros=maskedimage.size
        ratio=(count/zeros)*100 #<0.1
        cv2.imwrite('maskedimage.jpg',maskedimage)
        return ratio
    else:
        raise Exception("No pose landmarks detected")

# print(calculate_color_ratio(cv2.imread('img.jpg'),(0,0,98),(172 ,113,255)))