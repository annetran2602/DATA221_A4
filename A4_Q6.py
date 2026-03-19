# Anne Tran (UCID: 30286177)
# Assign 4_Q6

import tensorflow as tf
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras import layers, models
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
import numpy as np

# Training data
X_train=np.array(X_train)
y_train=np.array(y_train)

# Testing data
X_test=np.array(X_test)
y_test=np.array(y_test)

# Normalize pixel values
X_train=X_train.astype('float32')/255.0
X_test=X_test.astype('float32')/255.0

# Reshape the image
X_train=X_train[..., None]
X_test=X_test[..., None]

# Build a CNN
model=models.Sequential([
    layers.Input(shape=(28,28,1)),
    layers.Conv2D(16,3,padding='same',activation='relu'),
    layers.MaxPool2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPool2D(),
])

model.add(layers.Flatten())
model.add(layers.Dense(128, activation='relu'))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(10, activation='softmax'))

model.summary()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
history=model.fit(X_train, y_train, epochs=15, batch_size=64, validation_split=0.1)

test_loss, test_accuracy = model.evaluate(X_test, y_test)

print("Test loss:", test_loss)
print("Test accuracy:", test_accuracy)


