""" A config file for models we want to test """

#external 
import numpy as np 
import torch 
import tensorflow as tf


models_dict = {
    "model_1": {
        "model_provider": "huggingface",
        "model_category": "vision",
        "model_name": "google/vit-base-patch16-224",
        "task": "image-classification",
        "sample_input": "assets/images/animal_pictures/cat2.jpg",  # Supported via image_processor
        "framework_specific": {
            "image_processor": "AutoImageProcessor",
            "requires_feature_extractor": False
        }},
    "model_2": {
        "model_provider": "huggingface",
        "model_category": "vision",
        "model_name": "facebook/detr-resnet-50",
        "task": "object-detection",
        "sample_input": "assets/images/animal_pictures/dog1.jpg",  # Handled by image_processor
        "framework_specific": {
            "image_processor": "DetrImageProcessor"
        }
    },
    "model_3": {
        "model_provider": "huggingface",
        "model_category": "text",
        "model_name": "gpt2",
        "task": "text-generation",
        "sample_input": "The future of AI is",  # Handled by tokenizer+generate()
        "framework_specific": {
            "tokenizer": "AutoTokenizer"
        }
    },
    "model_4": {
        "model_provider": "huggingface",
        "model_category": "audio",
        "model_name": "facebook/wav2vec2-base-960h",
        "task": "automatic-speech-recognition",
        "sample_input": "assets/audios/audio1.wav",  # Handled by _load_audio()
        "framework_specific": {
            "feature_extractor": "Wav2Vec2FeatureExtractor"
        }
    },
    "model_5": {
        "model_provider": "pytorch",
        "model_category": "vision",
        "model_name": "resnet50",
        "task": "image-classification",
        "sample_input": torch.rand(1, 3, 224, 224),  # Handled by tensor reshaping
        "framework_specific": {
            "normalization": {
                "mean": [0.485, 0.456, 0.406],
                "std": [0.229, 0.224, 0.225]
            }
        }
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
        "model_provider": "tensorflow",
        "model_category": "audio",
        "model_name": "YAMNet",
        "task": "audio-event-classification",
        "sample_input": np.random.uniform(-1.0, 1.0, 15600),  # STFT processed
        "framework_specific": {
            "frame_length": 255,
            "frame_step": 128
        }
    },
    "model_10": {
        "model_provider": "tensorflow",
        "model_category": "text",
        "model_name": "GPT2",
        "task": "text-generation",
        "sample_input": tf.constant(["The quick brown fox"]),  # Direct model.predict
        "framework_specific": {
            "max_length": 50
        }
    }
    }