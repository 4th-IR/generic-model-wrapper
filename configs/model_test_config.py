""" A config file for models we want to test """

models_dict = {
    "model_1": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "WAV2VEC2_BASE"},
    "model_2": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
    "model_3": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "resnet18"},
    "model_4": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "Wav2Vec2Bundle"},
    "model_5": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "inception_v3"},
    "model_6": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "EMFORMER_RNNT_BASE_LIBRISPEECH"},
    "model_7": {"model_provider": "pytorch", "model_category": "torch_vision", "model_name": "shufflenet_v2_x1_5"},
    "model_8": {"model_provider": "pytorch", "model_category": "torch_audio", "model_name": "TACOTRON2_WAVERNN_CHAR_LJSPEECH"},
        }