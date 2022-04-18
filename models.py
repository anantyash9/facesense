import json
from typing import Optional,List
from fastapi import File, Form, UploadFile
from pydantic import BaseModel

class OurBaseModel(BaseModel):
    class Config:
        orm_mode = True


class PoseThreshold(OurBaseModel):
    y: float
    z: float

class HSVThreshold(OurBaseModel):
    hsvLower: List[int]
    hsvUpper: List[int]
    ratio: float

class PhotoResponse(BaseModel):
    valid: bool
    messages: List[str] = []


class PhotoRequest(OurBaseModel):
    poseThreshold: Optional[PoseThreshold]
    hsvThreshold: Optional[HSVThreshold]
    @classmethod
    def __get_validators__(cls):
        yield cls.validate_to_json

    @classmethod
    def validate_to_json(cls, value):
        if isinstance(value, str):
            return cls(**json.loads(value))
        return value
    class Config:
        schema_extra = {
            "example": {
                "poseThreshold": PoseThreshold(y=0.03,z=0.024),
                "hsvThreshold" : HSVThreshold(hsvLower=[0,0,98],hsvUpper=[172,113,255],ratio=0.5)
            }
        }