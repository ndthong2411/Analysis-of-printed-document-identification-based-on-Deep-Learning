import torch
from dataset import pre_data
import pandas as pd
import numpy as np
def read_dataset(input_size, batch_size, root, dataset_path, config, train_size= .7, shuffle= True):
# def read_dataset(input_size, batch_size, root, dataset_path, train_size= .7, shuffle= True):
  
  dataframe= pd.read_excel(dataset_path)
  
  dataframe=dataframe[dataframe['config'] == config]
  df_size= len(dataframe)
  if shuffle:
    dataframe = dataframe.sample(frac = 1)
  train_size= train_size
  val_size= 1-train_size
  split_train= int(np.floor(train_size*df_size))
  train_df= dataframe.iloc[:split_train]
  val_df= dataframe.iloc[split_train:]
  train_dataset= pre_data.Dataset(input_size, root,train_df,mode='train')
  val_dataset= pre_data.Dataset(input_size, root,val_df,mode='test')
  train_dataloader = torch.utils.data.DataLoader(train_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle= True,
                                                 num_workers= 8,
                                                 drop_last= False)
  valid_dataloader = torch.utils.data.DataLoader(val_dataset, 
                                                 batch_size=batch_size,
                                                 shuffle= True,
                                                 num_workers= 8,
                                                 drop_last= False)
  return train_dataloader, valid_dataloader