# Anne Tran (UCID: 30286177)
# Assign 4_Q1

import numpy as np
from sklearn.datasets import load_breast_cancer
data=load_breast_cancer()

X=data.data
y=data.target

# Report the shape of X and y
print("Shape of x: ", X.shape)
print("Shape of y: ", y.shape)

# Contribution
name, count=np.unique(y, return_counts=True)
print("\nNumber of samples to each class")
for name, count in zip(name, count):
    print(f"{data.target_names[name]}: {count}")
