# Anne Tran (UCID: 30286177)
# Assign 4_Q2

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

# Train split training & testing data
data=load_breast_cancer()
X=data.data
y=data.target
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

# Construct the decision tree
decision_tree_classifier=DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(X_train, y_train)

# Predict result and compute accuracy of the model
predicted_result=decision_tree_classifier.predict(X_test)
accuracy=accuracy_score(y_test, predicted_result)

# Answer the question-------------------------------------------------------
# 1.
