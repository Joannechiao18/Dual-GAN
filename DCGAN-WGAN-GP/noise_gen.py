import pandas as pd
import numpy as np
import torch
from sklearn import preprocessing

df = pd.read_csv("./../Dataset/train/train.csv")

tensor = np.zeros((0, 128), float)

def custom_training_set():
    global tensor, df
    for column in range(df.shape[1]):
        column = str(column)
        nan_num = df[column].isna().sum()
        for start in range(df.shape[0]-nan_num):
            if (start + 128) > (df.shape[0]-nan_num):
                break
            arr = np.expand_dims(df[column][start: start + 128], axis=1)
            mix_max_scaler = preprocessing.StandardScaler()
            arr_scaled = mix_max_scaler.fit_transform(arr)
            arr_scaled = np.transpose(arr_scaled)
            tensor = np.append(tensor, arr_scaled, axis=0)


    dataset = torch.from_numpy(tensor)
    dataset = dataset.unsqueeze(1).unsqueeze(-1)
    dataset = dataset.type(torch.FloatTensor)
    dataset = dataset.cuda()

    return dataset


