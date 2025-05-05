from configs.model_test_config import models_dict
import json

with open('./configs/model_config.json', '+a') as data:
    json.dump(models_dict, data)
    