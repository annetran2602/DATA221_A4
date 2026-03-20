# Anne Tran (UCID: 30286177)
# Assign 4_Q7

import tensorflow as tf
from keras.src.metrics.metrics_utils import confusion_matrix
from tensorflow.keras.datasets import fashion_mnist
(X_train, y_train), (X_test, y_test) = fashion_mnist.load_data()
import numpy as np
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

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

# Prediction
predicted_result_probs=model.predict(X_test, verbose=0)
predicted_result=np.argmax(predicted_result_probs, axis=1)

print("true", int(y_test[0]))
print("predicted image", int(predicted_result[0]))

confusionMatrix=confusion_matrix(y_test,predicted_result)
print(confusionMatrix)

misclassified_idx=np.where(predicted_result!=y_test)[0]
labels={0: "T-shirt/top", 1: "Trouser", 2: "Pullover", 3: "Dress",
    4: "Coat", 5: "Sandal", 6: "Shirt", 7: "Sneaker",
    8: "Bag", 9: "Ankle boot"}

for i in range(min(3, len(misclassified_idx))):
    idx=misclassified_idx[i]
    plt.imshow(X_test[idx].reshape(28, 28), cmap='gray')
    plt.title(f"True: {labels[y_test[idx]]} | Predicted: {labels[predicted_result[idx]]}")
    plt.axis('off')
    plt.show()

