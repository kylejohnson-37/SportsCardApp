# 1️⃣ Install required libraries before running this script:
# pip install huggingface_hub datasets tensorflow pillow scikit-learn requests

try:
    from datasets import load_dataset
except ModuleNotFoundError:
    raise ModuleNotFoundError("The 'datasets' library is missing. Install it using: pip install datasets")

import requests
import tensorflow as tf
import os
from PIL import Image
import numpy as np
from io import BytesIO
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# 1. DOWNLOAD & SAVE DATA
# -----------------------

dataset = load_dataset("GotThatData/sports-cards")
output_dir = "sports_cards_tf"
os.makedirs(output_dir, exist_ok=True)

train_dir = os.path.join(output_dir, "train")
os.makedirs(train_dir, exist_ok=True)

print("Sample keys:", dataset["train"][0].keys())

for i, ex in enumerate(dataset["train"]):
    label = ex.get("manufacturer") or "unknown"
    label = label.replace(" ", "_").lower()

    label_dir = os.path.join(train_dir, label)
    os.makedirs(label_dir, exist_ok=True)

    image = ex.get("image")  # Use the correct key for the image field

    if isinstance(image, str):  # If it's a URL
        try:
            response = requests.get(image, timeout=10)
            if response.status_code == 200:
                img_obj = Image.open(BytesIO(response.content)).convert("RGB")
                img_obj.save(os.path.join(label_dir, f"{i}.jpg"))
        except Exception as e:
            print(f"⚠️ Failed to download image for sample {i}: {e}")

    elif isinstance(image, Image.Image):  # If it's already a PIL image
        image.save(os.path.join(label_dir, f"{i}.jpg"))

# -----------------------
# 2. LOAD AS TENSORFLOW DATASETS WITH SPLIT
# -----------------------

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

if not os.path.exists(train_dir) or len(os.listdir(train_dir)) == 0:
    raise FileNotFoundError("No images saved in sports_cards_tf/train. The dataset may not contain downloadable images.")

full_ds = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=True,
    seed=123
)

if tf.data.experimental.cardinality(full_ds).numpy() == 0:
    raise RuntimeError("No images found in sports_cards_tf/train/<label>. Verify that image downloading worked correctly.")

val_size = int(0.2 * tf.data.experimental.cardinality(full_ds).numpy())
val_ds = full_ds.take(val_size)
train_ds = full_ds.skip(val_size)

class_names = full_ds.class_names

AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)

# -----------------------
# 3. DATA AUGMENTATION
# -----------------------

data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
    tf.keras.layers.RandomContrast(0.1),
])

# -----------------------
# 4. BUILD MODEL (TRANSFER LEARNING)
# -----------------------

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False

inputs = tf.keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.3)(x)
outputs = tf.keras.layers.Dense(len(class_names), activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)

# -----------------------
# 5. COMPILE & TRAIN
# -----------------------

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history = model.fit(train_ds, validation_data=val_ds, epochs=10)

model.save("sports_cards_model.h5")
print("✅ Model trained and saved as sports_cards_model.h5!")

# -----------------------
# 6. EVALUATION
# -----------------------

y_true, y_pred = [], []
for images, labels in val_ds:
    preds = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(preds, axis=1))

if len(y_true) > 0:
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=class_names))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=False, xticklabels=class_names, yticklabels=class_names, cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()
else:
    print("⚠️ No validation samples available for evaluation.")

# -----------------------
# 7. INFERENCE FUNCTION
# -----------------------

def predict_card(image_path):
    img = Image.open(image_path).resize(IMG_SIZE)
    img_array = np.expand_dims(np.array(img) / 255.0, axis=0)
    preds = model.predict(img_array)
    predicted_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)
    return predicted_class, confidence

# Example usage:
# predicted_label, conf = predict_card("path_to_card.jpg")
# print(f"Predicted: {predicted_label} (Confidence: {conf:.2f})")
