""" A script to interface with the MIS API """

#external 
import requests as re

base_url = 'https://mis-app-service.mangobeach-c18b898d.switzerlandnorth.azurecontainerapps.io/api/v1/models'


def create_model(model_details: dict) -> bool:
    """ A function to create a model in the MIS repository 
    Args:
        model_details: dict 
    """
    try:
        data = re.post(
            url=base_url,
            data=model_details
        )

        if data.status_code == 200:
            return True 
        
        return False
    except Exception as e:
        print(e)  # raise status error 


def fetch_model(model_id: str)  -> dict:
    """ A function to fetch a model from the MIS system 
    Args:
        model_id (str): the model identifier (usually the name)
    """
    pass