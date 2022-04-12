from fastapi import Body, Depends, FastAPI, File, UploadFile,Form
import uvicorn
import cv2
import numpy as np
import pose_calculation
import background_calculation
from models import PhotoResponse,PhotoRequest,PoseThreshold,HSVThreshold
from typing import Optional,List
# Declaring our FastAPI instance
app = FastAPI()


def verify_orientation(image):
    #print(image.shape)
    verified=False
    if image.shape[0] >= image.shape[1]:
        verified= True
    return verified 

def verify_face_pose(img):
    try:
        y,z=pose_calculation.calculate_YZ(img)
        print("y difference :", y)
        print("z difference :", z)
    except:
        return False
    if y>0.03 or z>0.024:

        return False
    return True

def verify_background(img):
    try:
        buzz=background_calculation.calculate_buzz(img)
        print("buzz:", buzz)
    except:
        return False
    if buzz>0.1:
        return False
    return True

 
# Defining path operation for /name endpoint
@app.post("/uploadfile/", response_model=PhotoResponse)
async def create_upload_file(file: UploadFile = File(...),photoRequestdata: Optional[PhotoRequest] = Body(...)):
    messages=[]
    print(photoRequestdata)
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    keep_processing=verify_orientation(img)
    if keep_processing:
        messages.append("Image is in correct orientation")
    else:
        messages.append("Image is not in correct orientation")
    if keep_processing:
        keep_processing=verify_face_pose(img)
        if keep_processing:
            messages.append("Face is in correct pose")
        else:
            messages.append("Face is not in correct pose")
        if keep_processing:
            keep_processing=verify_background(img)
            if keep_processing:
                messages.append("Background is nice and plain")
            else:
                messages.append("Background is not nice and plain")
    resp=PhotoResponse(messages=messages,valid=keep_processing)
    return resp
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)