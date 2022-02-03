import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.decomposition import KernelPCA
from utils import get_csvs, preprocessing_stu

path = './dataset'
path_data2013 = os.path.join(path, '2013data')
raw_csvs = get_csvs()

L2Y1S = raw_csvs[0][0]
# L2Y1P = raw_csvs[0][1]

df_input, df_label = preprocessing_stu(L2Y1S)
print("input shape:", df_input.shape)
# print(df_input.head())
print()

def pca_visualization(df_input, df_label, label="L2Y1_E_CS"):
    df = pd.merge(df_input, df_label[label], left_index=True, right_index=True, how='left')
    df = df.rename(columns={0:'pc1', 1:'pc2'})
    grouped = df.groupby(label)

    plt.rcParams['figure.figsize'] = [10, 8]
    fig, ax = plt.subplots()

    for name, group in grouped:
        if name==1 or name==4:
            ax.plot(group.pc1, group.pc2, marker='o', linestyle='', label=name, alpha=.3)

    ax.legend(fontsize=12, loc='upper left') # legend position

    plt.title('Scatter Plot of PCA with 2 components', fontsize=20)
    plt.xlabel('PC1', fontsize=14)
    plt.ylabel('PC2', fontsize=14)
    plt.show()

def pca_visualization_3d(df_input, df_label, label="L2Y1_E_CS"):
    df = pd.merge(df_input, df_label[label], left_index=True, right_index=True, how='left')
    df = df.rename(columns={0:'pc1', 1:'pc2', 2:'pc3'})
    grouped = df.groupby(label)

    plt.rcParams['figure.figsize'] = [10, 8]

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')

    for name, group in grouped:
        ax.scatter(group.pc1, group.pc2, group.pc3, marker='o', label=name)

    ax.legend(fontsize=12, loc='upper left') # legend position

    plt.title('Scatter Plot of PCA with 3 components', fontsize=20)
    ax.set_xlabel('PC1', fontsize=14)
    ax.set_ylabel('PC2', fontsize=14)
    ax.set_zlabel('PC3', fontsize=14)
    plt.show()

# find valid number of components
pca = PCA(n_components=.95)
pca_95 = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
# print(pca_95.head())
print("valid number of components: ",pca_95.shape[1])
print()

# pca with 2 components
pca = PCA(n_components = 2)
df_pca2 = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
print("PCA with 2 components")
# print(df_pca2.head())
print(" explained variance ratio :", pca.explained_variance_ratio_)
print(" sum of explained variance ratio :", sum(pca.explained_variance_ratio_)*100, "%")
print()

# pca with 3 components
pca = PCA(n_components = 3)
df_pca3 = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
print("PCA with 3 components")
# print(df_pca3.head())
print(" explained variance ratio :", pca.explained_variance_ratio_)
print(" sum of explained variance ratio :", sum(pca.explained_variance_ratio_)*100, "%")
print()

# pca with 2 components
pca = KernelPCA(n_components = 2, kernel="sigmoid", gamma=0.01)
df_kpca2 = pd.DataFrame(pca.fit_transform(df_input), index = df_input.index)
print("PCA with 2 components")
# print(df_kpca2.head())
print()

pca_visualization(df_pca2, df_label)
# pca_visualization_3d(df_pca3, df_label)
# pca_visualization(df_kpca2, df_label)