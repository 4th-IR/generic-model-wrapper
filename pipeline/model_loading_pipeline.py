""" A Script to Load a Model and Save to Azure """

#internal 
from model.main import ModelWrapper

def main(model_details):
    model_name = model_details['model_name']
    model_provider = model_details['model_provider']
    model_category = model_details['model_category']

    
    model_wrapper = ModelWrapper(model_provider, model_name, model_category)

    model_wrapper.load_model() 


if __name__ == '__main__':
    main()