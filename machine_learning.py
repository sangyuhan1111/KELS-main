import numpy as np
import pandas as pd
import os
from utils.utils import *
from sklearn.model_selection import StratifiedShuffleSplit
from algorithms.DT import Decision_Tree
from algorithms.Logistic import Logistic
from algorithms.RandomForest import RandomForest
from algorithms.SVM import SVM
from algorithms.RNN import oneRNN




# path_merge = './preprocessed/merge/inner'
path_merge = '../KELS_data/preprocessed/merge/inner'
df_input = pd.read_csv(os.path.join(path_merge, 'input_merge.csv'))
df_input = df_input.set_index('L2SID', drop=True)
df_label = pd.read_csv(os.path.join(path_merge, 'label_merge.csv'))
df_label = df_label.set_index('L2SID', drop=True)
df_label = df_label.astype(int)

# train test set split (stratified)
sss = StratifiedShuffleSplit(n_splits=1, test_size=.2, random_state=42)
X, y = df_input, df_label["L2Y6_K_CS"]
for train_idx, test_idx in sss.split(X, y):
    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

y_test = y_test.to_numpy()
print("train size:", X_train.shape)
print("test size:", X_test.shape)


# Decision_Tree(X_train, X_test, y_train, y_test)
# Logistic(X_train, X_test, y_train, y_test)
# RandomForest(X_train, X_test, y_train, y_test)
SVM(X_train, X_test, y_train, y_test, kernel='rbf')