""" A config file for models we want to test """

#external 
import numpy as np 
import torch 
import tensorflow as tf


models_dict = {
    # Task 1: image-classification (2 HF, 2 PT, 1 TF)
    "model_1":  {"model_provider": "huggingface", "model_category": "vision", "model_name": "google/vit-base-patch16-224",             "task": "image-classification", "uploaded_status": False},
    "model_2":  {"model_provider": "huggingface", "model_category": "vision", "model_name": "facebook/deit-base-distilled-patch16-224", "task": "image-classification", "uploaded_status": False},
    "model_3":  {"model_provider": "pytorch",      "model_category": "vision", "model_name": "resnet50",                              "task": "image-classification", "uploaded_status": False},
    "model_4":  {"model_provider": "pytorch",      "model_category": "vision", "model_name": "densenet121",                           "task": "image-classification", "uploaded_status": False},
    "model_5":  {"model_provider": "tensorflow",   "model_category": "vision", "model_name": "MobileNetV2",                           "task": "image-classification", "uploaded_status": False},

    # Task 2: object-detection (2 HF, 2 PT, 1 TF)
    "model_6":  {"model_provider": "huggingface", "model_category": "vision", "model_name": "facebook/detr-resnet-50",             "task": "object-detection",      "uploaded_status": False},
    "model_7":  {"model_provider": "huggingface", "model_category": "vision", "model_name": "ultralytics/yolov5",                   "task": "object-detection",      "uploaded_status": False},
    "model_8":  {"model_provider": "pytorch",      "model_category": "vision", "model_name": "fasterrcnn_resnet50_fpn",             "task": "object-detection",      "uploaded_status": False},
    "model_9":  {"model_provider": "pytorch",      "model_category": "vision", "model_name": "maskrcnn_resnet50_fpn",               "task": "object-detection",      "uploaded_status": False},
    "model_10": {"model_provider": "tensorflow",   "model_category": "vision", "model_name": "ssd_mobilenet_v2_fpnlite_640x640",   "task": "object-detection",      "uploaded_status": False},

    # Task 3: machine-translation (2 HF, 2 PT, 1 TF)
    "model_11": {"model_provider": "huggingface", "model_category": "text",   "model_name": "Helsinki-NLP/opus-mt-en-de",        "task": "machine-translation",   "uploaded_status": False},
    "model_12": {"model_provider": "huggingface", "model_category": "text",   "model_name": "facebook/m2m100_418M",                "task": "machine-translation",   "uploaded_status": False},
    "model_13": {"model_provider": "pytorch",      "model_category": "text",   "model_name": "transformer_wmt_en_de",              "task": "machine-translation",   "uploaded_status": False},
    "model_14": {"model_provider": "pytorch",      "model_category": "text",   "model_name": "fairseq/wmt19.en-de",                "task": "machine-translation",   "uploaded_status": False},
    "model_15": {"model_provider": "tensorflow",   "model_category": "text",   "model_name": "t5-small",                             "task": "machine-translation",   "uploaded_status": False},

    # Task 4: text-classification (2 HF, 2 PT, 1 TF)
    "model_16": {"model_provider": "huggingface", "model_category": "text",   "model_name": "bert-base-uncased",                  "task": "text-classification",   "uploaded_status": False},
    "model_17": {"model_provider": "huggingface", "model_category": "text",   "model_name": "roberta-base",                        "task": "text-classification",   "uploaded_status": False},
    "model_18": {"model_provider": "pytorch",      "model_category": "text",   "model_name": "distilbert-base-uncased",            "task": "text-classification",   "uploaded_status": False},
    "model_19": {"model_provider": "pytorch",      "model_category": "text",   "model_name": "albert-base-v2",                      "task": "text-classification",   "uploaded_status": False},
    "model_20": {"model_provider": "tensorflow",   "model_category": "text",   "model_name": "bert-base-uncased",                  "task": "text-classification",   "uploaded_status": False},

    # Task 5: automatic-speech-recognition (1 HF, 1 PT, 3 TF)
    "model_21": {"model_provider": "huggingface", "model_category": "audio",  "model_name": "facebook/wav2vec2-base-960h",        "task": "automatic-speech-recognition", "uploaded_status": False},
    "model_22": {"model_provider": "pytorch",      "model_category": "audio",  "model_name": "wav2vec2_base",                       "task": "automatic-speech-recognition", "uploaded_status": False},
    "model_23": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "DeepSpeech",                         "task": "automatic-speech-recognition", "uploaded_status": False},
    "model_24": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "SpeechTransformer",                  "task": "automatic-speech-recognition", "uploaded_status": False},
    "model_25": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "Jasper10x5",                         "task": "automatic-speech-recognition", "uploaded_status": False},

    # Task 6: audio-classification (1 HF, 1 PT, 3 TF)
    "model_26": {"model_provider": "huggingface", "model_category": "audio",  "model_name": "moby/panns_cnn14",                   "task": "audio-classification",           "uploaded_status": False},
    "model_27": {"model_provider": "pytorch",      "model_category": "audio",  "model_name": "cnn14",                               "task": "audio-classification",           "uploaded_status": False},
    "model_28": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "YAMNet",                              "task": "audio-classification",           "uploaded_status": False},
    "model_29": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "VGGish",                              "task": "audio-classification",           "uploaded_status": False},
    "model_30": {"model_provider": "tensorflow",   "model_category": "audio",  "model_name": "MobileNetV2-Audio",                  "task": "audio-classification",           "uploaded_status": False},
}
