import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

from utils import get_csvs, preprocessing_stu, plot_2d
from sklearn.manifold import LocallyLinearEmbedding

path = './dataset'
path_data2013 = os.path.join(path, '2013data')
raw_csvs = get_csvs()
L2Y1S = raw_csvs[0][0]
# L2Y1P = raw_csvs[0][1]
df_input, df_label = preprocessing_stu(L2Y1S)

LLE = LocallyLinearEmbedding(n_components=2, n_neighbors=3, random_state=42)
lle_3 = pd.DataFrame(LLE.fit_transform(df_input), index=df_input.index)

# print(lle_3.head())
plot_2d(lle_3, df_label)