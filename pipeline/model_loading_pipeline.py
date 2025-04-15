""" A Script to Load a Model and Save to Azure """

#internal 
from model.main import ModelWrapper

def load_model(model_details):
    model_name = model_details['model_name']
    model_provider = model_details['model_provider']
    model_category = model_details['model_category']


    model_wrapper = ModelWrapper(model_provider, model_name, model_category)

    try:
        model_wrapper.load_model() 

        # if model saves to Azure - update MIS 

    except Exception as e:
        pass 

