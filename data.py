import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from collections import defaultdict
from copy import deepcopy

class User_Item_Dataset():
    def __init__(self, user_item_p):
        self.ui_df = pd.read_csv(user_item_p)
        self.ui_dict = defaultdict(list)
        self.ui_train_dict = defaultdict(list)
        self.ui_valid_dict = defaultdict(list)
        self.train_idx = []
        self.valid_idx = []

        for _, r in self.ui_df.iterrows():
            self.ui_dict[r['user_id']].append(r['item_id'])

    def generate_train_valid_split(self, threshold = 1, method = 'leave_one_last'):
        if (method == 'leave_one_last'):
            for user, items in zip(self.ui_dict.keys(), self.ui_dict.values()):
                if len(items) <= threshold:
                    continue
                else:
                    self.ui_valid_dict[user] = [items[-1]]
                    self.ui_train_dict[user] = list(filter(lambda x: x != items[-1], items))
        else:
            print('Unknown method')


        for i, (_, r) in enumerate(self.ui_df.iterrows()):
            if r['user_id'] in self.ui_train_dict.keys() and r['item_id'] in self.ui_train_dict[r['user_id']]:
                self.train_idx.append(i)
            elif r['user_id'] in self.ui_valid_dict.keys() and r['item_id'] in self.ui_valid_dict[r['user_id']]:
                self.valid_idx.append(i)

    def get_train_valid_idx(self):
        return self.train_idx, self.valid_idx

class User_Item_Train_Dataset(Dataset):
    def __init__(self, user_p, item_p, ui_dataset, num_ns):
        self.user_p = user_p
        self.item_p = item_p
        self.ui_dataset = ui_dataset
        self.num_ns = num_ns

        self.user_df = pd.read_csv(user_p, index_col=0)
        self.item_df = pd.read_csv(item_p, index_col=0)

        self.num_user = len(self.user_df)
        self.num_item = len(self.item_df)
        self.samples = []

        

    def __len__(self):
        return len(self.samples)

    def generate_train_samples(self):
        train_idx, _ = self.ui_dataset.get_train_valid_idx() 

        ns_item_id = np.random.choice(self.item_df.index, size = 15*len(train_idx*self.num_ns))
        ns_item_idx = 0

        user_item_df = self.ui_dataset.ui_df
        user_item_dict = self.ui_dataset.ui_dict
        
        self.samples = []

        for _ , r in user_item_df.iloc[train_idx, :].iterrows():
            for i in range(self.num_ns):
                while True:
                    #if self.item_df['item_id'][ns[ns_idx]]  not in self.user_item_dict[r['user_id']]:
                    if ns_item_id[ns_item_idx]  not in user_item_dict[r['user_id']]:
                        break
                    else:
                        ns_item_idx+=1
                self.samples.append((r['user_id'], r['item_id'], ns_item_id[ns_item_idx]))


    def __getitem__(self, idx): 
        user_id, pos_item_id, neg_item_id = self.samples[idx][0], self.samples[idx][1], self.samples[idx][2]
        return self.user_df.loc[user_id].to_numpy(), self.item_df.loc[pos_item_id].to_numpy(), self.item_df.loc[neg_item_id].to_numpy()

class User_Item_Valid_Dataset(Dataset):
    def __init__(self, user_p, item_p, ui_dataset, num_ns):
        self.user_df = pd.read_csv(user_p, index_col=0)
        self.item_df = pd.read_csv(item_p, index_col=0)
        self.ui_dataset = ui_dataset
        self.num_ns = num_ns
        self.samples = []
        self.candidate_dict = defaultdict(list)
        
    def __len__(self):
        return len(self.samples)

    def generate_candidate(self):
        self.candidate_dict = defaultdict(list)
        ui_valid_dict = self.ui_dataset.ui_valid_dict
        ui_train_dict = self.ui_dataset.ui_train_dict
        for user_id in ui_valid_dict.keys():
            for item_id in self.item_df.index:
                if item_id not in ui_train_dict[user_id]:
                    self.candidate_dict[user_id].append(item_id)

    def generate_valid_samples(self, user_id):
        self.samples = []
        for candidate_id in self.candidate_dict[user_id]:
            self.samples.append((user_id, candidate_id))

    def __getitem__(self, idx):
        user_id, candidate_id = self.samples[idx][0], self.samples[idx][1]
        return self.user_df.loc[user_id].to_numpy(), self.item_df.loc[candidate_id].to_numpy(), candidate_id
    
    def get_valid_users(self):
        return self.ui_dataset.ui_valid_dict.keys()

    def get_positive_items(self, user_id):
        return self.ui_dataset.ui_valid_dict[user_id]
