"""A testing Class for Model Wrapper """

#external 
from PIL import Image

#internal 
from configs.model_test_config import models_dict
from model.main import ModelWrapper
from utils.env_manager import *
from pipeline.model_loading_pipeline import load_model
from pipeline.model_inference import model_inference
from utils.logger import get_logger

LOG = get_logger('test')


def test_model_loading():
    model = models_dict['model_3']
    load_model(model)


def test_inference():
    model = models_dict['model_3']
    model_inference(model)

if __name__ == '__main__':
    LOG.info('starting model loading')
    test_model_loading()

    LOG.info('Starting inference test')
    test_inference()