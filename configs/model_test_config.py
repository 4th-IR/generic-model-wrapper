""" A config file for models we want to test """

#external 
import numpy as np 
import torch 
import tensorflow as tf 


models_dict = {
    "model_1": {
        "model_provider": "huggingface",
        "model_category": "multimodal",
        "model_name": "Qwen/Qwen-VL",
        "task": "text-generation",
        "sample_input": [
                        {'image': 'https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen-VL/assets/demo.jpeg'},
                        {'text': 'Generate the caption in English with grounding:'},
                    ],  

    },
    "model_2": {
        "model_provider": "huggingface",
        "model_category": "multimodal",
        "model_name": "Salesforce/blip-image-captioning-base",
        "task": "image-recognition",
        "sample_input": [
                        {'image': 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg'},
                        {'text': 'a photography of'},
                    ],
    },
     "model_3": {
        "model_provider": "huggingface",
        "model_category": "multimodal",
        "model_name": "Salesforce/blip-image-captioning-base",
        "task": "image-recognition",
        "sample_input": [
                        {'image': '/home/model-wrapper/tests/assets/images/animal_pictures/dog2.jpg'},
                        {'text': 'a photography of'},
                    ],
    },
    "model_4": {
        "model_provider": "huggingface",
        "model_category": "multimodal",
        "model_name": "microsoft/git-base",
        "task": "image-recognition",
        "sample_input": [
                        {'image': '/home/model-wrapper/tests/assets/images/animal_pictures/cat1.jpg'},
                        {'text': 'describe the picture'},
                    ],
    },
    "model_5": {
        "model_provider": "huggingface",
        "model_category": "multimodal",
        "model_name": "Salesforce/blip2-opt-2.7b",
        "task": "image-recognition",
        "sample_input": [
                        {'image': '/home/model-wrapper/tests/assets/images/animal_pictures/cat1.jpg'},
                        {'text': 'How many cats are in the picture?'}
        ],
    },
    "model_6": {
        "model_provider": "pytorch",
        "model_category": "audio",
        "model_name": "wav2vec2_base",
        "task": "audio-classification",
        "sample_input": torch.randn(1, 16000),  # Processed via MelSpectrogram
        "framework_specific": {
            "sample_rate": 16000
        }
    },
    "model_7": {
        "model_provider": "pytorch",
        "model_category": "text",
        "model_name": "bert-base-uncased",
        "task": "text-classification",
        "sample_input": torch.randint(0, 10000, (1, 128)),  # Pre-tokenized input
        "framework_specific": {
            "attention_mask": torch.ones(1, 128)
        }
    },
    "model_8": {
        "model_provider": "tensorflow",
        "model_category": "vision",
        "model_name": "MobileNetV2",
        "task": "image-segmentation",
        "sample_input": np.random.rand(256, 256, 3),  # Handled by resize/normalize
        "framework_specific": {
            "input_shape": (256, 256, 3)
        }
    },
    "model_9": {
        "model_provider": "pytorch",
        "model_category": "audio",
        "model_name": "CONVTASNET_BASE_LIBRI2MIX",
        "task": "audio-event-classification",
        "sample_input": "./audios/audio2.wav",  # STFT processed
        "framework_specific": {
            "frame_length": 255,
            "frame_step": 128
        }
    },
     "model_10": {
        "model_provider": "pytorch",
        "model_category": "vision",
        "model_name": "vgg19",
        "task": "image-classification",
        "sample_input": "./assets/images/cat1.jpg",  # STFT processed
        "framework_specific": {
            "frame_length": 255,
            "frame_step": 128
        }
    },
    "model_11": {
        "model_provider": "tensorflow",
        "model_category": "text",
        "model_name": "gemma_2b_en",
        "task": "text-generation",
        "sample_input": tf.constant(["The quick brown fox"]),  # Direct model.predict
        "framework_specific": {
            "max_length": 50
        }
    }
    }