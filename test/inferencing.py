import torch, torchvision
from torchvision.transforms import transforms
from PIL import Image
import time
import torchaudio
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import decode_predictions
import numpy as np
import os
import keras
import keras_hub
import matplotlib.pyplot as plt
import psutil
import pandas as pd
import soundfile



def inference_model(model_provider:str, model_name:str, model_category:str, model_path="./models_saved"):
    
    if model_provider == "pytorch" and model_category=="torch_vision":
        try:
            print("== Image model inferencing begins ...==")

            model = torch.load(f"{model_path}/{model_name}.pt", weights_only=False)
            model.eval()

            preprocess = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])

            # Load the image and apply the preprocessing pipeline.
            image_path = "./animal_pctures/dog1.jpg"  # Replace with the actual image path
            input_image = Image.open(image_path).convert("RGB")
            input_tensor = preprocess(input_image)
            input_batch = input_tensor.unsqueeze(0)

            # Move to GPU if available
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model.to(device)
            input_batch = input_batch.to(device)

            # Perform inference.
            with torch.no_grad():
                output = model(input_batch)

            # Apply softmax to convert logits to probabilities.
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            predicted_prob, predicted_class = torch.max(probabilities, dim=0)

            print(f"Predicted class: {predicted_class.item()}, Probability: {predicted_prob.item():.4f}")

            print("== Vision model inferencing successful ==")
            # print(F"{TOTAL_TIME_TAKEN} minutes in total")

        except Exception as e:
            print("Error during torch vision model inferencing", e)

    if model_provider == "pytorch" and model_category=="torch_audio":
        try:
            import torchaudio
            torchaudio.set_audio_backend("soundfile")
            print("== audio model inferencing beginning ==")
            # Load the pre-trained ConvTasNet model
            # bundle = f"{torchaudio.pipelines}.{model_name}"
            model = torch.load(f"{model_path}/{model_name}.pt", weights_only=False)
            model.eval()

            # Load the audio file
            audio_path = "./audios/audio2.wav"
            waveform, sample_rate = torchaudio.load(audio_path)

            # Resample if the audio sample rate doesn't match the model's expected sample rate
            expected_sampling_rate = 16000
            if sample_rate != expected_sampling_rate:
                resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=expected_sampling_rate)
                waveform = resampler(waveform)
                sample_rate = expected_sampling_rate

            # Ensure the waveform is mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)

            waveform = waveform.unsqueeze(0) 

            # Perform source separation
            with torch.no_grad():
                separated_sources = model(waveform)
            print("Separated sources shape:", separated_sources.shape)
        except Exception as e:
            print("Error inferencing torch audio model", e)


    """
            INFERENCING LOGIC FOR TENSORFLOW/KAGGLE/KERAS
                                    
    """

    save_dir = "./models_saved"
    os.makedirs(save_dir, exist_ok=True)
    preset_name = model_name

    # IMAGE CLASSIFICATION
    if model_provider == "tensorflow" and model_category == "image_classifier":

        try:
            print(f"==={preset_name} is downloading... ===")

            image_classifier = keras_hub.models.ImageClassifier.from_preset(preset_name, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            image_classifier.save(model_save_path)
            print(f"Image classifier model '{preset_name}' saved at {model_save_path}")

            print(f"==={preset_name} inference starting... ===")
            
            # load the model
            model = keras.models.load_model(f"{model_path}/{model_name}.keras")

            image_path = "./animal_pctures/dog1.jpg"

        #  Make sure the image is loaded and preprocessed correctly
            image = keras.utils.load_img(image_path, target_size=(224, 224))
            image_array = keras.utils.img_to_array(image)
            image_array = np.expand_dims(image_array, axis=0)
            image_array = keras.applications.imagenet_utils.preprocess_input(image_array)

            preds = model.predict(image_array)
            print(keras_hub.utils.decode_imagenet_predictions(preds))
            print(f"==={preset_name} inference was successful===")
        except Exception as e:
            print("Error", e)

    # SEQ2SEQ
    if model_provider == "tensorflow" and model_category == "seq2seq_lm":
        try:
            print(f"==={preset_name} is downloading... ===")
            seq2seq_lm = keras_hub.models.Seq2SeqLM.from_preset(preset_name, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            seq2seq_lm.save(model_save_path)
            print(f"Seq2seq_lm '{preset_name}' saved at {model_save_path}")
                
        
            print(f"==={preset_name} inference starting... ===")

            model = keras.models.load_model(f"{model_path}/{model_name}.keras")
            model.compile(sampler="top_k")

            input_texts = "Translate to french: The quick brown fox jumped over the lazy dog."
            generated_texts = model.generate([input_texts], max_length=30)  

            print("Generated Summary:", generated_texts[0])

            print(f"==={preset_name} inferencing was successful===")
        except Exception as e:
            print("Error", e)

    # TEXT CLASSIFICATION
    if model_provider == "tensorflow" and model_category == "text_classification":
        try:
            print(f"==={preset_name} is downloading... ===")
            text_classifier = keras_hub.models.TextClassifier.from_preset(preset_name, num_classes=2, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            text_classifier.save(model_save_path)
            print(f"Text classifier model '{preset_name}' saved at {model_save_path}")


            print(f"==={preset_name} inference starting... ===")
            model = keras.models.load_model(f"{model_path}/{model_name}.keras")

            sample_inputs = [
            "Keras is an easy-to-use deep learning library.",
            "This product was awful and broke within two days.",
            "I am so happy about the food",
        ]

            # Run predictions
            predictions = model.predict(sample_inputs)

            # Convert predictions to class labels
            predicted_classes = np.argmax(predictions, axis=-1)
            print(predicted_classes)

            # Print predictions
            for text, label in zip(sample_inputs, predicted_classes):
                if label == 1:
                    label = "negative"
                if label == 0:
                    label = "positive"

                print(f"Input: {text}\nPredicted class: {label}\n")

            print(f"==={preset_name} inference successful... ===")
        except Exception as e:
            print("Error", e)

    # TEXT TO IMAGE
    if model_provider == "tensorflow" and model_category == "text_to_image":
        try:
            print(f"==={preset_name} is downloading... ===")
            text_to_image = keras_hub.models.TextToImage.from_preset(preset_name, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            text_to_image.save(model_save_path)
            print(f"Text to image model '{preset_name}' saved at {model_save_path}")

            print(f"==={preset_name} inference starting... ===")
            model = keras.models.load_model(f"{model_path}/{model_name}.keras")
            prompt = "A red apple on a white table"
            image = model.generate(prompt)

            # Display the first generated image
            plt.imshow(image[0])
            plt.axis("off")
            plt.title(prompt)
            plt.show()

            print(f"==={preset_name} inference successful... ===")

        except Exception as e:
            print("Error", e)

    # IMAGE TO IMAGE
    if model_provider == "tensorflow" and model_category == "image_to_image":
        try:
            print(f"==={preset_name} is downloading... ===")

            image_to_image = keras_hub.models.ImageToImage.from_preset(preset_name, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            image_to_image.save(model_save_path)
            print(f"Image to image model '{preset_name}' saved at {model_save_path}")

            print(f"==={preset_name} inferencing starting... ===")
            model = keras.models.load_model(f"{model_path}/{model_name}.keras")

            print("creating the reference image for inference")

            reference_image = np.ones((1024, 1024, 3), dtype="float32")


            print("Image is generating")
            prompt = "A red apple on a white table"
            image = model.generate(reference_image, prompt)

            # Display the first generated image
            plt.imshow(image[0])
            plt.axis("off")
            plt.title(prompt)
            plt.show()

            print(f"==={preset_name} inferencing was successful===")
        except Exception as e:
            print("Error", e)

    # CAUSAL LM
    if model_provider == "tensorflow" and model_category == "causal_lm":
        try:
            print(f"==={preset_name} is downloading... ===")

            causal_lm = keras_hub.models.CausalLM.from_preset(preset_name, load_weights=True)
            model_save_path = os.path.join(model_path, f"{preset_name}.keras")
            causal_lm.save(model_save_path)

            print(f"causal_lm '{preset_name}' saved at {model_save_path}")

            print(f"==={preset_name} inferencing starting... ===")
            model.compile(sampler="top_k")

            input_texts = "The cat sat."
            generated_texts = model.generate([input_texts], max_length=30)

            print("Generated prediction:", generated_texts[0])

            print(f"==={preset_name} inferencing was successful===")
        
        except Exception as e:
            print("Error", e)

    else:
        print("invalid input")
       


# if __name__ == "__main__":


    # models_dict = {
    # "model_1": {"model_provider": "tensorflow", "model_category": "causal_lm", "model_name": "bart_large_en_cnn"},
    # "model_2": {"model_provider": "tensorflow", "model_category": "seq2seq_lm", "model_name": "bart_base_en"},
    # "model_3": {"model_provider": "tensorflow", "model_category": "text_classification", "model_name": "deberta_v3_large_en"},
    # "model_4": {"model_provider": "tensorflow", "model_category": "causal_lm", "model_name": "bloom_560m_multi"},
    # "model_5": {"model_provider": "tensorflow", "model_category": "image_classifier", "model_name": "vit_large_patch16_384_imagenet"},
    # "model_6": {"model_provider": "tensorflow", "model_category": "image_classifier", "model_name": "resnet_vd_200_imagenet"},
    # "model_7": {"model_provider": "tensorflow", "model_category": "seq2seq_lm", "model_name": "bart_large_en_cnn"},
    # "model_8": {"model_provider": "tensorflow", "model_category": "text_classification", "model_name": "distil_bert_base_en"},
    # }

    # excel_file = 'torch_model_metrics.xlsx'

    # model_inference_metrics = []


    # for model in models_dict.values():
    #     model_name = model["model_name"]
    #     model_provider = model["model_provider"]
    #     model_category = model["model_category"]


    #     start_inference_time = time.time()
    #     process = psutil.Process(os.getpid())

    #     mem_before = process.memory_info().rss / (1024 ** 2)

    #     inference_model(model_provider, model_name, model_category)

    #     end_inference_time = time.time()

    #     mem_after = process.memory_info().rss / (1024 ** 2)

    #     mem_used = mem_after - mem_before


    #     TOTAL_TIME_TAKEN = round((end_inference_time - start_inference_time)/60, 2)
    #     model_data = {'model_name': model_name,
    #                     'model_provider': model_provider,
    #                     "model_category": model_category,
    #                     "total_time_taken(mins)": TOTAL_TIME_TAKEN,
    #                     "memory_used": mem_used}
                                
    #     model_inference_metrics.append(model_data)

    # df = pd.DataFrame(model_inference_metrics)

    # if os.path.exists(excel_file):
    #     existing_df = pd.read_excel(excel_file)
    #     updated_df = pd.concat([existing_df, df], ignore_index=True)
    # else:
    #     updated_df = df

    # # Save back to Excel
    # updated_df.to_excel(excel_file, index=False)

    # print(f"Saved {len(df)} model entries to {excel_file}")


