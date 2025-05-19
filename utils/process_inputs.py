from fastapi import UploadFile, HTTPException
from pydantic import BaseModel, validator
from PIL import Image
import os
import tempfile
import shutil
import re

# Dict of image extensions that will be allowed 
ALLOWED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3"}

# function to process images
def process_images(image: UploadFile):
    suffix = os.path.splitext(image.filename)[1] or ".jpg"
    if suffix not in ALLOWED_IMAGE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_img:
        shutil.copyfileobj(image.file, tmp_img)
        tmp_img_path = tmp_img.name

    return tmp_img_path 
# function to process audio uploads
def process_audio(audio_file: UploadFile):
    suffix = os.path.splitext(audio_file.filename)[1].lower()
    if suffix not in ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp_file:
        shutil.copyfileobj(audio_file.file, tmp_file)
        tmp_aud_path = tmp_file.name
    
    return tmp_aud_path

# process texts, no links accepted
class TextInput(BaseModel):
    text: str

    @validator('text')
    def no_links(cls, v):
        if re.search(r"(https?://|www\.)", v):
            raise ValueError("Links are not allowed")
        if not v.strip():
            raise ValueError("Text cannot be empty")
        return v