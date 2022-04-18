from fastapi import Body, Depends, FastAPI, File, UploadFile,Form
import uvicorn
import cv2
import numpy as np
import pose_calculation
import background_calculation
from models import PhotoResponse,PhotoRequest,PoseThreshold,HSVThreshold
from typing import Optional,List
# Declaring our FastAPI instance

tags_metadata = [
    {
        "name": "users",
        "description": "Operations with users. These can be called without any special authentication.",
    },]


description = """
Face Sense API helps you figur out if a photo is suitable to be used in ID cards and Profile Pictures. 🚀
"""

app = FastAPI(
    title="Face Pose and Background Verification API",
    description=description,
    version="0.0.1",
    contact={
        "name": "Anant Yash Pande",
        "email": "anant.pande@infosys.com",
    },
    openapi_tags=tags_metadata,
)


def verify_orientation(image):
    #print(image.shape)
    verified=False
    if image.shape[0] >= image.shape[1]:
        verified= True
    return verified 

def verify_face_pose(img,pose):
    try:
        y,z=pose_calculation.calculate_YZ(img)
        print("y difference :", y)
        print("z difference :", z)
    except:
        return False
    if y>pose.y or z>pose.z:

        return False
    return True

def verify_background(img,hsvThreshold):
    try:
        buzz=background_calculation.calculate_color_ratio(img,hsvThreshold.hsvLower,hsvThreshold.hsvUpper)
        print("buzz:", buzz)
    except:
        return False
    if buzz>hsvThreshold.ratio:
        return False
    return True

 
# Defining path operation for /name endpoint
@app.post("/verifyPhoto/", response_model=PhotoResponse,tags=["users"])
async def verify_profile_photo(file: UploadFile = File(...),photoRequestdata: Optional[PhotoRequest] = Body(...)):
    messages=[]
    print("input : ",photoRequestdata)
    if photoRequestdata.hsvThreshold is None:
        photoRequestdata.hsvThreshold=HSVThreshold(hsvLower=[0,0,98],hsvUpper=[172,113,255],ratio=0.5)
    if photoRequestdata.poseThreshold is None:
        photoRequestdata.poseThreshold=PoseThreshold(y=0.03,z=0.024)
    
    print("used : ",photoRequestdata)
    contents = await file.read()
    nparr = np.fromstring(contents, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    keep_processing=verify_orientation(img)
    if keep_processing:
        messages.append("Image is in correct orientation")
    else:
        messages.append("Image is not in correct orientation")
    if keep_processing:
        keep_processing=verify_face_pose(img,photoRequestdata.poseThreshold)
        if keep_processing:
            messages.append("Face is in correct pose")
        else:
            messages.append("Face is not in correct pose")
        if keep_processing:
            keep_processing=verify_background(img,photoRequestdata.hsvThreshold)
            if keep_processing:
                messages.append("Background is in the desired color range")
            else:
                messages.append("Background is not in the desired color range")
    resp=PhotoResponse(messages=messages,valid=keep_processing)
    return resp
    

if __name__ == '__main__':
    uvicorn.run(app, host='0.0.0.0', port=8000)