from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

def SVM(X_train, X_test, y_train, y_test, kernel='rbf'):
    print("Support Vector Machine Classifier")
    ## SVM
    if kernel=='poly':
        # support vector machine wiht poly kernel
        SVM_poly = SVC(kernel = 'poly')
        SVM_poly.fit(X_train, y_train)
        y_pred = SVM_poly.predict(X_test)
        # get accuracy_score
        print("poly accuracy:", accuracy_score(y_pred, y_test)*100,"%")
        print()

    elif kernel == 'rbf':
        # support vector machine wiht rbf kernel
        SVM_rbf = SVC(kernel = 'rbf')
        SVM_rbf.fit(X_train, y_train)
        y_pred = SVM_rbf.predict(X_test)
        # get accuracy_score
        print("rbf accuracy:", accuracy_score(y_pred, y_test)*100,"%")
        print()
        print("confusion matrix")
        print(confusion_matrix(y_test, y_pred))
        print()
        print("classificaion report")
        print(classification_report(y_test, y_pred))
        print()

    elif kernel == 'linear':
        # support vector machine wiht linear kernel
        SVM_linear = SVC(kernel = 'linear')
        SVM_linear.fit(X_train, y_train)
        y_pred = SVM_linear.predict(X_test)
        # get accuracy_score
        print("linear accuracy:", accuracy_score(y_pred, y_test)*100,"%")
        print("confusion matrix")
        print(confusion_matrix(y_test, y_pred))
        print()
        
    else:
        print(kernel,"kernel is not provide")