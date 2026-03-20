import tensorflow as tf
from tensorflow.keras import layers, models

# Load dataset
train_data = tf.keras.preprocessing.image_dataset_from_directory(
    "dataset",
    image_size=(150,150),
    batch_size=32
)

# Build CNN mode
model = models.Sequential([
    layers.Rescaling(1./255),

    layers.Conv2D(32, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# Compile model
model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

# Train model
model.fit(train_data, epochs=5)

# Save model
model.save("model.h5")

print("Model trained and saved!")
