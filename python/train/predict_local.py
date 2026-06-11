import tensorflow as tf
import numpy as np
from PIL import Image
import os

# --- 1. Model Configuration ---
# The central configuration for all your specialized models
MODELS = {
    "rice": {
        "path": "rice_model_v2.h5",
        "classes": ['Bacterial leaf blight', 'Brown spot', 'Leaf smut']
    },
    "tomato": {
        "path": "tomato_model_v1.h5",
        "classes": [
            'Tomato_Bacterial_spot', 'Tomato_Early_blight', 'Tomato_Late_blight',
            'Tomato__Tomato_YellowLeaf__Curl_Virus','Tomato___healthy'
        ] # Update if you used a different subset
    },
    "potato": {
        "path": "potato_model_v1.h5",
        "classes": ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']
    },
    "wheat": {
        "path": "wheat_model_final_version.h5",
        "classes": ['black_rust', 'brown_rust', 'healthy', 'septoria']
    }

}

# Get the absolute path of the directory where this script is located
script_dir = os.path.dirname(os.path.abspath(__file__))

# --- 2. User Input ---
# Ask the user to choose a crop from the available options
crop_choice = input(f"Enter the crop type {list(MODELS.keys())}: ").lower()

# Check if the chosen crop is valid
if crop_choice not in MODELS:
    print(f"❌ Error: Invalid crop type. Please choose from {list(MODELS.keys())}")
    exit()

# Ask the user for the image path
image_path_input = input(f"Enter the path to your {crop_choice} test image: ")
image_path = os.path.abspath(image_path_input)

# --- 3. Load the Correct Model and Classes ---
model_info = MODELS[crop_choice]
model_filename = model_info["path"]
class_names = model_info["classes"]
model_path = os.path.join(script_dir, "..", "models", model_filename)

print(f"\nLoading the specialized '{crop_choice}' model...")
try:
    loaded_model = tf.keras.models.load_model(model_path)
    print("✅ Model loaded successfully.")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit()

# --- 4. Load and Preprocess the Image ---
try:
    img = Image.open(image_path).convert("RGB").resize((224, 224))
    img_array = np.expand_dims(np.array(img), axis=0)
    # Apply the same preprocessing used in training
    img_array = tf.keras.applications.mobilenet_v2.preprocess_input(img_array)
    print("✅ Image preprocessed successfully.")
except FileNotFoundError:
    print(f"❌ ERROR: The image file was not found at {image_path}")
    exit()

# --- 5. Make and Display Prediction ---
predictions = loaded_model.predict(img_array)
predicted_class_index = np.argmax(predictions[0])
confidence = np.max(predictions[0])
predicted_label = class_names[predicted_class_index]

print("\n--- Prediction Result ---")
print(f"Crop Type: {crop_choice.capitalize()}")
print(f"Predicted Condition: {predicted_label.replace('___', ' ')}")
print(f"Confidence: {confidence:.2%}")
print("-----------------------")

