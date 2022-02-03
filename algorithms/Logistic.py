from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def Logistic(X_train, X_test, y_train, y_test):
    # softmax regression(Multinomial Logistic Regression)
    softmax_reg = LogisticRegression(multi_class="multinomial", solver="saga", penalty='l1', C=1, max_iter=10000, random_state=42)
    softmax_reg.fit(X_train, y_train)
    y_pred = softmax_reg.predict(X_test)

    # accuracy
    print("Logistic Regression Classifier")
    print("accuracy:", accuracy_score(y_pred, y_test)*100,"%")
    print()
    print("confusion matrix")
    print(confusion_matrix(y_test, y_pred))
    print()
    print("classificaion report")
    print(classification_report(y_test, y_pred))