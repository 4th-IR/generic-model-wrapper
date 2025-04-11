import tensorflow as tf
import os
import kagglehub
import keras_hub
import tfimm


def from_tensorflow(model_name: str, save_path: str = "saved_models", **kwargs):
    # Create the save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    os.environ['KAGGLEHUB_CACHE'] = save_path

    # First, check if the model exists in tf.keras.applications
    if hasattr(tf.keras.applications, model_name):
        print("Checking tf.keras.applications...")
        try:
          model_class = getattr(tf.keras.applications, model_name)
          # Load the model with default weights (ImageNet)
          model = model_class(weights='imagenet')
          model_save_path = os.path.join(save_path, f"{model_name}.keras")
          # Save the model using tf.keras save (this produces a single file in the .keras format)
          model.save(model_save_path)
          print(f"Model '{model_name}' saved at {model_save_path}")

        except Exception as e:
          print(f"Error downloading model {model_name}: {str(e)}. Checking TFIMM")
        #   # Load pretrained TF model
          model = tfimm.create_model(model_name, pretrained=True)

          # Save it in the Keras format
          model_save_path = os.path.join(save_path, f"{model_name}.tf")
          model.save(model_save_path)

          print(f"Successfully downloaded and saved {model_name} to {model_save_path}")

        return model_save_path

    # If the model is not found in tf.keras.applications, check for a KerasHub preset
    else:
        print(f"Model '{model_name}' not found in tf.keras.applications.")
        print("Checking KerasHub...")
        model_type = input("Enter a model type (e.g., text_classifier, image_classifier): ").strip()
        preset_name = model_name  # Use the same name as a preset identifier

        # For text classifiers, load the entire classifier from the preset
        if model_type == "text_classifier":
            num_classes = kwargs.get("num_classes", 2)
            text_classifier = keras_hub.models.TextClassifier.from_preset(preset_name, num_classes=num_classes)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            text_classifier.save(model_save_path)
            print(f"Text classifier model '{preset_name}' saved at {model_save_path}")
            return model_save_path

        # For image classifiers, load and save similarly
        elif model_type == "image_classifier":
            image_classifier = keras_hub.models.ImageClassifier.from_preset(preset_name)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            image_classifier.save(model_save_path)
            print(f"Image classifier model '{preset_name}' saved at {model_save_path}")
            return model_save_path

        elif model_type == "image_to_image":
            image_to_image = keras_hub.models.ImageToImage.from_preset(preset_name)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            image_to_image.save(model_save_path)
            print(f"Image to image model '{preset_name}' saved at {model_save_path}")
            return model_save_path

        elif model_type == "text_to_image":
            text_to_image = keras_hub.models.TextToImage.from_preset(preset_name)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            text_to_image.save(model_save_path)
            print(f"Text to image model '{preset_name}' saved at {model_save_path}")
            return model_save_path

        elif model_type == "seq2seq_lm":
            seq2seq_lm = keras_hub.models.Seq2SeqLM.from_preset(preset_name)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            seq2seq_lm.save(model_save_path)
            print(f"Seq2seq_lm '{preset_name}' saved at {model_save_path}")
            return model_save_path

        elif model_type == "causal_lm":
            causal_lm = keras_hub.models.CausalLM.from_preset(preset_name)
            model_save_path = os.path.join(save_path, f"{preset_name}.keras")
            causal_lm.save(model_save_path)
            print(f"Seq2seq_lm '{preset_name}' saved at {model_save_path}")
            return model_save_path

        else:
            print("Unknown model type for KerasHub. Falling back to KaggleHub.")
            print("""Please enter the KaggleHub model handle in this format:
                    <owner>/<model_name>/<framework>/<variation>/<version>
                    For example: google/efficientnet-v2/tensorFlow2/imagenet21k-b1-classification/1
                  """)
            model_handle = input("-> ").strip()
            model_path = kagglehub.model_download(model_handle)
            print(f"KaggleHub model downloaded to: {model_path}")
            return model_path

# # Example usage
# if __name__ == "__main__":
#     model_name = "gemma_2b_en"  # Change this to any available model in tf.keras.applications or preset name
#     save_directory = "models"  # Define the directory where the model will be saved
#     from_tensorflow(model_name, save_directory)