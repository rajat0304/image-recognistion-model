# run in google Colab
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os

#block of code for Settings
DATA_DIR = "/path/to/dataset"   # change: path to your dataset root
IMG_SIZE = (224, 224)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
EPOCHS = 15

# 1) for Loading the dataset
# taking the open dataset from kaggle of different cars like BMW,MERCEDIES,etc 
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "train"),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=True,
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATA_DIR, "val"),
    labels="inferred",
    label_mode="categorical",
    batch_size=BATCH_SIZE,
    image_size=IMG_SIZE,
    shuffle=False
)

#defing the class name 
class_names = train_ds.class_names
num_classes = len(class_names)
print("Classes:", class_names)

# Prefetch(to increase the latency and speed of program)
#It help to improve the imficency of the code
train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

# 2) Data augmentation (on-the-fly)
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.08),
    layers.RandomZoom(0.08),
    # add more if you want
], name="data_augmentation")

# 3) Building the model with transfer learning (MobileNetV2)

preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

base_model = tf.keras.applications.MobileNetV2(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # freeze for initial training

inputs = keras.Input(shape=IMG_SIZE + (3,))
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)

model = keras.Model(inputs, outputs)
#importing keras here to perform the task

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4) Callbacks
#calling as the name defined in it 
checkpoint_cb = keras.callbacks.ModelCheckpoint("best_model.h5", save_best_only=True, monitor="val_accuracy")
earlystop_cb = keras.callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)

# 5) Training the model (initial)
#intial traning of the model(least optimise)
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# 6) better optimision
base_model.trainable = True
# Freeze first N layers or set a cutoff
fine_tune_at = 100
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable = False

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

fine_tune_epochs = 10
total_epochs = EPOCHS + fine_tune_epochs

history_fine = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1] + 1,
    callbacks=[checkpoint_cb, earlystop_cb]
)

# 7) Evaluate on test set
if os.path.isdir(os.path.join(DATA_DIR, "test")):
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        os.path.join(DATA_DIR, "test"),
        labels="inferred",
        label_mode="categorical",
        batch_size=BATCH_SIZE,
        image_size=IMG_SIZE,
        shuffle=False
    )
    test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
    loss, acc = model.evaluate(test_ds)
    print("Test accuracy:", acc)

# 8) Saveing the final model
model.save("image_recognition_model")

# 9) Example predict on a single image
# can process only single image at a time 
import numpy as np
from tensorflow.keras.preprocessing import image

def predict_image(img_path, model, class_names):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    arr = image.img_to_array(img)
    arr = np.expand_dims(arr, axis=0)
    arr = preprocess_input(arr)
    preds = model.predict(arr)
    idx = np.argmax(preds[0])
    return class_names[idx], preds[0][idx]

# usage:
# class_pred, prob = predict_image("/path/to/some.jpg", model, class_names)
# print(class_pred, prob)
