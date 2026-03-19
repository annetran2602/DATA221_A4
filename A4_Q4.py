# Anne Tran (UCID: 30286177)
# Assign 4_Q4

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, InputLayer

# Training & testing data
data=load_breast_cancer()
X=data.data
y=data.target
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# Standardize feature
scaler=StandardScaler()
X_train_scale=scaler.fit_transform(X_train) # fit_transform- training data
X_test_scale=scaler.transform(X_test) # transform- testing data

# Define a model
neural_network_model=Sequential()

# Define input, hidden, output layer
input_layer=InputLayer(shape=(30,)) # input layer
neural_network_model.add(input_layer)

hidden_layer=Dense(16, activation='relu') # 1 hidden layer
neural_network_model.add(hidden_layer)

output_layer=Dense(1, activation='sigmoid') # 1 output layer
neural_network_model.add(output_layer)

# Configure the model
neural_network_model.compile(loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
neural_network_model.fit(X_train_scale, y_train, epochs=50, batch_size=16, verbose=0)

# Compute training accuracy & testing accuracy
train_loss, train_accuracy=neural_network_model.evaluate(X_train_scale, y_train, verbose=0)
test_loss, test_accuracy=neural_network_model.evaluate(X_test_scale, y_test, verbose=0)

print(f"Train accuracy: {train_accuracy:.3f}")
print(f"Test accuracy: {test_accuracy:.3f}")





