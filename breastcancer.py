import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
import shap

#parameters
batch_size = 32
img_height = 224
img_width = 224
train_path = r"C:\Users\Yhlas\Downloads\archive\BreaKHis 400X\train"
test_path = r"C:\Users\Yhlas\Downloads\archive\BreaKHis 400X\test"
epochs = 20

#data loading
train_ds = tf.keras.utils.image_dataset_from_directory(
    train_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=True
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    test_path,
    image_size=(img_height, img_width),
    batch_size=batch_size,
    shuffle=False
)

normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))

#DATA AUGMENTATION
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal_and_vertical"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.1),
])

#applying augmentation to training dataset
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

#base
base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(224,224,3))
base_model.trainable = False 

#adding custom classifier head
x = GlobalAveragePooling2D()(base_model.output)
x = Dense(128, activation='relu')(x)
output = Dense(1, activation='sigmoid')(x)#binary classification

model = Model(inputs=base_model.input, outputs=output)

#compile
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.summary()

callbacks = [
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1),
    EarlyStopping(monitor='val_loss', patience=7, restore_best_weights=True)
]

for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=tf.keras.optimizers.Adam(1e-5),
              loss='binary_crossentropy',
              metrics=['accuracy'])    

#compute class weights
y_train = np.concatenate([y.numpy() for x, y in train_ds], axis=0)
classes = np.unique(y_train)
class_weights_arr = compute_class_weight(
    class_weight='balanced',
    classes=classes,
    y=y_train
)

class_weights = {int(c): w for c, w in zip(classes, class_weights_arr)}
print("Class Weights:", class_weights)


#TRAINING

history = model.fit(
    train_ds, 
    validation_data=val_ds,
    epochs=epochs,
    class_weight=class_weights
)

#evaluate the model
loss, accuracy = model.evaluate(val_ds)
print(f"\nValidation Accuracy: {accuracy:.4f}")

# Convert the entire validation dataset to NumPy arrays
y_true = np.concatenate([y.numpy() for x, y in val_ds], axis=0)
y_pred = np.concatenate([model.predict(x).flatten() for x, y in val_ds], axis=0)
y_pred_classes = np.round(y_pred).astype(int)



# Evaluate using scikit-learn metrics

print("\nClassification Report:")
print(classification_report(y_true, y_pred_classes))

print("\nConfusion Matrix:")
print(confusion_matrix(y_true, y_pred_classes))

#SHAP explanation

shuffle=False

sample_batch = next(iter(val_ds))
X_sample, _ = sample_batch
X_sample = X_sample[:16]

bg_batch = next(iter(train_ds))
X_bg, _ = bg_batch
X_bg = X_bg[:16]   # background (16 samples)

# Take a small sample from validation to explain
val_batch = next(iter(val_ds))
X_sample, _ = val_batch
X_sample = X_sample[:8]   # keep small for speed/memory

#GradientExplainer (capital E) — works with TF models
explainer = shap.GradientExplainer(model, X_bg)
# Get shap values for the sample
shap_values = explainer.shap_values(X_sample)

try:
    shap.image_plot(shap_values, X_sample.numpy())
except Exception:
    # fallback: if shap_values is a list with one element, unwrap it
    if isinstance(shap_values, list) and len(shap_values) == 1:
        shap.image_plot(shap_values[0], X_sample.numpy())
    else:
        # last-resort: convert to numpy and plot
        shap.image_plot(np.array(shap_values), X_sample.numpy())




 
