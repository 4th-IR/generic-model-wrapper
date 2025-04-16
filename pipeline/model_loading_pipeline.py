""" A Script to Load a Model and Save to Azure """

#internal 
from model.main import ModelWrapper
from utils.api_interceptor import create_model

def load_model(model_details):
    model_name = model_details['model_name']
    model_provider = model_details['model_provider']
    model_category = model_details['model_category']


    model_wrapper = ModelWrapper(model_provider, model_name, model_category)

    try:
        model_wrapper.load_model() 

        # if model saves to Azure - update MIS 
        model_details = {
            "model_identifier": model_name,
            "version": 0,
            "created_at": None, 
            "last_modified": None,
            "framework": model_provider
        }

        return create_model(model_details)


    except Exception as e:
        raise RuntimeError(f'Model loading Failed: ... {e}') 

