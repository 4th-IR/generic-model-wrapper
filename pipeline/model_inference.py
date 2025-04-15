"""A script to load models from Azure and Perform Inference """

#internal
from model.main import ModelWrapper

def model_inference(model_details):
    # fetch model details from MIS

    model_name = model_details['model_name']
    model_provider = model_details['model_provider']
    model_category = model_details['model_category']

    input_data = model_details['input_data']

    model_wrapper = ModelWrapper(model_provider, model_name, model_category)

    model_wrapper.load_model() #ensure it loads from Azure 

    model_wrapper.run_inference(input_data)



