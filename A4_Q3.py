# Anne Tran (UCID: 30286177)
# Assign 4_Q3

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import numpy as np

data=load_breast_cancer()

X=data.data
y=data.target

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=42)

decision_tree_classifier=DecisionTreeClassifier(criterion='entropy', max_depth=10)
decision_tree_classifier.fit(X_train, y_train)

# Training & testing prediction
y_train_predict=decision_tree_classifier.predict(X_train)
y_test_predict=decision_tree_classifier.predict(X_test)


# Training & testing accuracy
train_accuracy=accuracy_score(y_train,y_train_predict)
test_accuracy=accuracy_score(y_test, y_test_predict)

# Feature importance
feature_importance=decision_tree_classifier.feature_importances_
feature_names=data.feature_names

# Top 5 features
indices=np.argsort(feature_importance)[::-1][:5]
print("\nTop 5 important feature:")
for i in indices:
    print(f"{feature_names[i]}: {feature_importance[i]:.4f}")




