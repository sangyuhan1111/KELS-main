import torch
import numpy as np
import os
import pandas as pd

import torch.utils.data as D

np.random.seed(42)

def get_year(series):
    nan_index = series.index[series.notna()].tolist()
    year = sorted(list(set([int(_[3]) for _ in nan_index])))
    year_col = ['L2Y'+str(y) for y in year]
    
    return year, year_col
    
class KELS(D.Dataset):
    def __init__(self, root_dir='./preprocessed/merge/outer'):
        self.root_dir = root_dir
        self.input_df = pd.read_csv(os.path.join(self.root_dir, 'input_merge.csv')).set_index('L2SID')
        self.label_df = pd.read_csv(os.path.join(self.root_dir, 'label_merge.csv')).set_index('L2SID')
        
    def __len__(self):
        if len(self.label_df) == len(self.input_df):
            return len(self.label_df)
    
        return False
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        input_series = self.input_df.iloc[idx]
        label_series = self.label_df.iloc[idx]
        year, year_col = get_year(label_series)
        
        # ## without year 6
        # if 6 in year:
        #     year.remove(6)
        
        input, label = {}, {}
        label6_fit = {1:4.0, 2:4.0, 3:3.0, 4:3.0, 5:3.0, 6:2.0, 7:2.0, 8:1.0, 9:1.0}
        year_ = []

        for y, y_col in zip(year, year_col):
            # # without year 6
            # if y != 6:
            #     label_index = [ _ for _ in label_series.index if _.startswith(y_col)]
            #     input_index = [ _ for _ in input_series.index if _.startswith(y_col)]

            #     input_y = input_series[input_index].to_dict()
            #     label_y = label_series[label_index].to_dict()



            #     input[y], label[y] = input_y, label_y
            
            # with year 6
            label_index = [ _ for _ in label_series.index if _.startswith(y_col)]
            input_index = [ _ for _ in input_series.index if _.startswith(y_col)]

            input_y = input_series[input_index].to_dict()
            label_y = label_series[label_index].to_dict()
                
            if any(np.isnan(val) for val in input_y.values())==False and any(np.isnan(val) for val in label_y.values())==False:
                if y == 6:
                    label_y = {k:label6_fit[int(label_y[k])] for k in label_y}

                input[y], label[y] = input_y, label_y
                year_.append(y)

        sample = {'year':year_, 'input':input, 'label':label}
        return sample
        

def train_val_test_split(dataset, test_size=500, val_ratio=.2):
    # KELS dataset size : 7156 (760, 6396)
    # first split test set with size (less than , # of complete data)
    # then, split train validation set with ratio
    # return : split dataset
    
    test_size = min(872, test_size) 
    complete_indices, incomplete_indices = np.array([]), np.array([])
    
    for idx, sample in enumerate(dataset):
        # if sample['year'] == [1,2,3,4,5]:
        if sample['year'] == [1,2,3,4,5,6]:
            complete_indices = np.append(complete_indices, idx)
        else:
            incomplete_indices = np.append(incomplete_indices, idx)
    
    # print(len(complete_indices))
    # print(len(incomplete_indices))
    
    test_indices = np.random.choice(complete_indices, test_size, replace=False)    
    train_val_indices = np.concatenate([np.setdiff1d(complete_indices, test_indices), incomplete_indices])
    
    val_size = int(len(train_val_indices)*val_ratio)
    val_indices = np.random.choice(train_val_indices, val_size, replace=False)
    train_indices = np.setdiff1d(train_val_indices, val_indices)
        
    test_indices = test_indices.astype(int)
    val_indices = val_indices.astype(int)
    train_indices = train_indices.astype(int)
        
        
    
    return D.SubsetRandomSampler(train_indices), D.SubsetRandomSampler(val_indices), D.SubsetRandomSampler(test_indices)