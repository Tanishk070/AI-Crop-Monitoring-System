'''import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import os

# --- 1. Dataset Setup ---
# Adjust this path if your data is located elsewhere
dataset_path = "../../data/rice_leaf_disease/" 
img_size = (224, 224)
batch_size = 32

# Load datasets directly from directories
train_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=img_size,
    batch_size=batch_size
)

class_names = train_ds.class_names
print("✅ Classes found:", class_names)

# --- 2. Performance Optimization ---
# Configure dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# --- 3. Data Augmentation ---
# Create data augmentation layers
data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.1),
        layers.RandomZoom(0.1),
    ]
)

# --- 4. Build Model with Transfer Learning ---
# Load pre-trained MobileNetV2 without the top classification layer
base_model = keras.applications.MobileNetV2(
    input_shape=(224, 224, 3),
    include_top=False,
    weights='imagenet'
)
# Freeze the base model layers
base_model.trainable = False

# Create the new model by adding our layers on top
inputs = keras.Input(shape=(224, 224, 3))
x = data_augmentation(inputs)  # Apply augmentation
x = keras.applications.mobilenet_v2.preprocess_input(x) # Preprocess for MobileNetV2
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x) # Add dropout for regularization
outputs = layers.Dense(len(class_names), activation='softmax')(x)
model = keras.Model(inputs, outputs)

# Compile the model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# --- 5. Train the Model ---
print("\n🔥 Starting model training...")
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=5 # Keep low for a quick prototype
)
print("✅ Training complete.")

# --- 6. Save the Model ---
# Ensure the models directory exists
models_dir = "../models"
os.makedirs(models_dir, exist_ok=True)
model.save(os.path.join(models_dir, "rice_leaf_model.h5"))
print(f"✅ Model saved to {os.path.join(models_dir, 'rice_leaf_model.h5')}")

# --- 7. Plot Accuracy ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['accuracy'], label="train_acc")
plt.plot(history.history['val_accuracy'], label="val_acc")
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
'''

import tensorflow as tf

print("TensorFlow Version:", tf.__version__)
print("Keras Version:", tf.keras.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

print("\n✅ Setup is successful!")