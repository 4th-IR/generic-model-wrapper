"""A script to load models from Azure and Perform Inference """

#internal
from model.main import ModelWrapper
from utils.logger import get_logger

LOG = get_logger('inference')

def model_inference(model_details):
    # fetch model details from MIS

    model_name = model_details['model_name']
    model_provider = model_details['model_provider']
    model_task = model_details['task']
    model_task = model_details['task']

    LOG.info(f'Task for model: {model_task}')

    input_data = "assets/images/animal_pictures/cat2.jpg"

    model_wrapper = ModelWrapper(model_provider, model_name, model_task)

    model_wrapper.load_model() #ensure it loads from Azure 

    model_wrapper.run_inference(input_data, model_task)



