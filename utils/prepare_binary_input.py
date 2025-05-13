from fastapi import UploadFile, File, HTTPException
from fastapi import UploadFile, File, HTTPException
from typing import Dict
from io import BytesIO
from PIL import Image
import pydub
import speech_recognition as sr


# Process image file for model inference
def process_image(file: UploadFile) -> Dict:
    image_data = BytesIO(file.file.read())
    image = Image.open(image_data)
    
    # Example: Resize image for the model
    image = image.resize((224, 224))  # Assuming model expects 224x224 images
    
    # Convert image to a format suitable for the model
    return {"image": image}

# Process audio file for model inference
def process_audio(file: UploadFile) -> Dict:
    file_extension = file.filename.split('.')[-1].lower()
    
    audio_data = BytesIO(file.file.read())
    if file_extension == 'wav':
        audio = sr.AudioFile(audio_data)
    elif file_extension == 'mp3':
        audio = pydub.AudioSegment.from_mp3(audio_data)
        # convert mp3 to wav format
        audio = audio.export(format="wav")
    
    # Example: Convert to a format suitable for the model
    return {"audio": audio}

# Process text file for model inference
def process_text(file: UploadFile) -> Dict:
    text_data = file.file.read().decode("utf-8")
    
    # Example: Return text as is for further processing
    return {"text": text_data}

# Utility function to process different file types
def process_file(file: UploadFile, task_type: str) -> Dict:
    file_extension = file.filename.split('.')[-1].lower()

    # Depending on task type, process different file types
    if task_type == "image_classification":
        # For image classification, allow only image formats
        if file_extension not in ['jpg', 'jpeg', 'png']:
            raise HTTPException(status_code=400, detail="Invalid image file format")
        return process_image(file)
    
    elif task_type == "audio_transcription":
        # For audio transcription, allow only audio formats
        if file_extension not in ['wav', 'mp3']:
            raise HTTPException(status_code=400, detail="Invalid audio file format")
        return process_audio(file)
    
    elif task_type == "text_processing":
        # For text processing, allow only text formats
        if file_extension not in ['txt', 'doc', 'docx', '.md']:
            raise HTTPException(status_code=400, detail="Invalid text file format")
        return process_text(file)
    
    else:
        raise HTTPException(status_code=400, detail="Unsupported task type")
