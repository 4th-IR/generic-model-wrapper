# Model Support Progress & Required Additions

This document outlines the current support status for models listed in `configs/model_config.json` and the necessary additions to achieve full compatibility with the loading and inference pipeline.

## Current Status Summary

- **Infrastructure:** Basic loading wrappers (`ModelWrapper`, `torch_route.py`, `tensorflow_route.py`) and preprocessing utilities (`preprocessing.py`) are in place.
- **Supported Tasks (Partial):** Image Classification, Text Classification (TF/HF), Machine Translation (TF/HF), Text Generation (TF), Image Generation (TF) have some level of support demonstrated in test scripts or likely covered by HF pipelines.
- **Major Gaps:** Lack of specific inference logic and post-processing for several tasks (Object Detection, ASR, Audio Classification), limitations in non-interactive model loading (especially TensorFlow), and missing support for models requiring specific external libraries (e.g., fairseq).

## Model-Specific Status & Gaps

| Model ID  | Provider    | Task                           | Predicted Status | Notes & Missing Features                                                                                                                               |
| :-------- | :---------- | :----------------------------- | :--------------- | :----------------------------------------------------------------------------------------------------------------------------------------------------- |
| `model_1` | huggingface | image-classification           | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_2` | huggingface | image-classification           | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_3` | pytorch     | image-classification           | **Likely Pass**  | Supported by `torchvision.models` & `torch_route.py`. Inference example exists.                                                                        |
| `model_4` | pytorch     | image-classification           | **Likely Pass**  | Supported by `torchvision.models` & `torch_route.py`. Inference logic similar to `model_3`.                                                              |
| `model_5` | tensorflow  | image-classification           | **Likely Pass**  | Supported by `tf.keras.applications` & `tensorflow_route.py`. Inference example exists.                                                                |
| `model_6` | huggingface | object-detection               | **Fail**         | Loading OK. **Missing Inference Logic & Post-processing** (bounding boxes).                                                                            |
| `model_7` | huggingface | object-detection               | **Fail**         | Loading might need `ultralytics` integration. **Missing Inference Logic & Post-processing**.                                                           |
| `model_8` | pytorch     | object-detection               | **Fail**         | Loading OK (`torchvision.models`). **Missing Inference Logic & Post-processing**.                                                                      |
| `model_9` | pytorch     | object-detection               | **Fail**         | Loading OK (`torchvision.models`). **Missing Inference Logic & Post-processing** (boxes + masks).                                                      |
| `model_10`| tensorflow  | object-detection               | **Fail**         | Loading needs non-interactive TF Hub/KerasHub/KaggleHub support. **Missing Inference Logic & Post-processing**.                                        |
| `model_11`| huggingface | machine-translation          | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_12`| huggingface | machine-translation          | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_13`| pytorch     | machine-translation          | **Fail**         | Loading needs `hub_repo` specification or library integration. **Missing Inference Logic** (tokenization, generation loop).                            |
| `model_14`| pytorch     | machine-translation          | **Fail**         | Loading needs `fairseq` integration. **Missing Inference Logic**.                                                                                      |
| `model_15`| tensorflow  | machine-translation          | **Likely Pass**  | Likely supported by KerasHub Seq2SeqLM preset & `tensorflow_route.py`. Inference example exists.                                                       |
| `model_16`| huggingface | text-classification            | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_17`| huggingface | text-classification            | **Likely Pass**  | Standard HF task.                                                                                                                                      |
| `model_18`| pytorch     | text-classification            | **Fail**         | Loading needs HF `transformers` integration via PT or Torch Hub. **Missing Inference Logic** (if not using HF pipeline).                               |
| `model_19`| pytorch     | text-classification            | **Fail**         | Similar to `model_18`.                                                                                                                                 |
| `model_20`| tensorflow  | text-classification            | **Likely Pass**  | Likely supported by KerasHub TextClassifier preset & `tensorflow_route.py`. Inference example exists.                                                  |
| `model_21`| huggingface | automatic-speech-recognition | **Fail**         | Loading OK. **Missing Inference Logic & Decoding**.                                                                                                    |
| `model_22`| pytorch     | automatic-speech-recognition | **Fail**         | Loading OK (`torchaudio.pipelines`). **Missing Inference Logic & Decoding**.                                                                           |
| `model_23`| tensorflow  | automatic-speech-recognition | **Fail**         | Loading needs non-interactive TF Hub/library support (e.g., DeepSpeech). **Missing Inference Logic & Decoding**.                                       |
| `model_24`| tensorflow  | automatic-speech-recognition | **Fail**         | Loading needs non-interactive TF Hub/library support. **Missing Inference Logic & Decoding**.                                                          |
| `model_25`| tensorflow  | automatic-speech-recognition | **Fail**         | Loading needs non-interactive TF Hub/library support (e.g., NeMo). **Missing Inference Logic & Decoding**.                                             |
| `model_26`| huggingface | audio-classification           | **Fail**         | Loading OK. **Missing Inference Logic**.                                                                                                               |
| `model_27`| pytorch     | audio-classification           | **Fail**         | Loading needs library integration (e.g., `panns_inference`). **Missing Inference Logic**.                                                              |
| `model_28`| tensorflow  | audio-classification           | **Fail**         | Loading needs non-interactive TF Hub support (YAMNet). **Missing Inference Logic**.                                                                    |
| `model_29`| tensorflow  | audio-classification           | **Fail**         | Loading needs non-interactive TF Hub support (VGGish). **Missing Inference Logic**.                                                                    |
| `model_30`| tensorflow  | audio-classification           | **Fail**         | Loading needs non-interactive TF Hub/library support. **Missing Inference Logic**.                                                                     |

## Required Additions / Modifications

1.  **`utils/tensorflow_route.py`:**
    *   Remove interactive `input()` calls. Pass necessary info (e.g., `model_type`, `kaggle_handle`) via function arguments or read from an enhanced config.
    *   Implement robust, non-interactive loading from TensorFlow Hub.
    *   Add specific library integrations if required (e.g., DeepSpeech, NeMo).

2.  **`utils/torch_route.py`:**
    *   Allow passing `hub_repo` argument, potentially read from config.
    *   Integrate `fairseq` library for `model_14`.
    *   Consider integrating `transformers` for PyTorch backend to load models like `model_18`, `model_19`.
    *   Integrate `panns_inference` or similar for `model_27`.

3.  **`model/main.py` (or Inference Logic Location):**
    *   Implement `run_inference` logic for `object-detection`:
        *   Handle different output formats (DETR, YOLO, FasterRCNN, SSD).
        *   Include post-processing for bounding boxes and potentially masks (`model_9`).
    *   Implement `run_inference` logic for `machine-translation` (PyTorch):
        *   Include appropriate tokenization and generation loops.
    *   Implement `run_inference` logic for `automatic-speech-recognition`:
        *   Include decoding steps to generate transcripts (e.g., CTC decoding).
    *   Implement `run_inference` logic for `audio-classification`.

4.  **`utils/preprocessing.py`:**
    *   Review and potentially enhance `prepare_multimodal_input` to ensure correct data formatting for the newly supported tasks, especially for non-Hugging Face models.
    *   Add any task-specific preprocessing not covered (e.g., specific audio feature extraction if not handled by the model/feature_extractor itself).

5.  **`configs/model_config.json`:**
    *   Consider adding optional fields like `hub_repo` (PyTorch), `keras_hub_preset_type` (TF), `kaggle_handle` (TF), `required_library` to facilitate loading specific models without modifying the loading code extensively.