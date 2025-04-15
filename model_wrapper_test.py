"""A testing Class for Model Wrapper """

#external 
from PIL import Image

#internal 
from configs.model_test_config import models_dict
from model.main import ModelWrapper
from utils.env_manager import *
from pipeline.model_loading_pipeline import load_model
from pipeline.model_inference import model_inference


def test_model_loading():
    for id, model in models_dict.items():
        load_model(model)


def test_inference():
    for id, model in models_dict.items():
        model_inference(model)

if __name__ == '__main__':
    test_model_loading()

    test_inference()