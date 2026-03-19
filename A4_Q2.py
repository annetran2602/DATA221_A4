# Anne Tran (UCID: 30286177)
# Assign 4_Q2

from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data=load_breast_cancer()
x=load_breast_cancer.data
y=load_breast_cancer.target
features_train, features_test, labels_train, labels_test=train_test_split(x,y, test_size=0.2)

decision_tree_classifier=DecisionTreeClassifier(criterion='entropy')
decision_tree_classifier.fit(features_train, labels_train)

predicted_labels=decision_tree_classifier.predict(features_test)
accuracy=accuracy_score(labels_test, predicted_labels)

# Answer the question-------------------------------------------------------
# 1.
