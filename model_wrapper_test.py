"""A testing Class for Model Wrapper """

#external 
from PIL import Image

#internal 
from configs.model_test_config import models_dict
from model.main import ModelWrapper
from utils.env_manager import *


def main():
    test_model = models_dict['model_3']
    model_provider = test_model['model_provider']
    model_category = test_model['model_category']
    model_name = test_model['model_name']

    model_wrapper = ModelWrapper(model_provider, model_name, model_category)

    model_wrapper.load_model() 

    #perform inference 
    image_path = '../assets/images/animal/animal_pictures/cat1.jpg'
    image = Image.open(image_path)

    preds = model_wrapper.run_inference(image)
    print(preds)


if __name__ == '__main__':
    main()

