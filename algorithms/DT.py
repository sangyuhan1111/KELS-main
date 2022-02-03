from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Decision tree
def Decision_Tree(X_train, X_test, y_train, y_test):
    dt = DecisionTreeClassifier()
    dt = dt.fit(X_train, y_train)
    y_pred = dt.predict(X_test)

    # accuracy
    print("Decision Tree")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))