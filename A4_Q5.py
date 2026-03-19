# Anne Tran (UCID: 30286177)
# Assign 4_Q5
from keras import Sequential
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from tensorflow.keras.layers import Dense, InputLayer
import tensorflow as tf
import numpy as np

# Load data
data=load_breast_cancer()
X=data.data
y=data.target

# Train split the training & testing data
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# Decision tree model
decision_tree_model=DecisionTreeClassifier(criterion='entropy')
decision_tree_model.fit(X_train, y_train)
y_predict_1=decision_tree_model.predict(X_test)

# Display confusion matrix for decision tree model
confusion_matrix1=confusion_matrix(y_test,y_predict_1)
print(f"Confusion matrix for decision tree model: \n{confusion_matrix1}")

# Neural network model
scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X_train)
X_test_scale=scaler.transform(X_test)

neural_network_model=Sequential()

input_layer=InputLayer(shape=(30,)) # input layer
neural_network_model.add(input_layer)

hidden_layer=Dense(16, activation='relu') # 1 hidden layer
neural_network_model.add(hidden_layer)

output_layer=Dense(1, activation='sigmoid') # 1 output layer
neural_network_model.add(output_layer)

neural_network_model.compile(loss='binary_crossentropy')

neural_network_model.fit(X_train_scale, y_train, epochs=50, batch_size=16, verbose=0)

y_predict_2=(neural_network_model.predict(X_test_scale) > 0.5).astype(int).flatten()

# Display confusion matrix for neural network
confusion_matrix2=confusion_matrix(y_test,y_predict_2)
print(f"Confusion matrix for neural network model: \n{confusion_matrix2}")




