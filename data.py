import numpy as np
import torch
from torch.utils.data import Dataset
import os
import pandas as pd
from collections import defaultdict

class User_Item_ID(Dataset):
    def __init__(self, user_item_p):
        self.user_item_df = pd.read_csv(user_item_p)
        self.num_ps = len(self.user_item_df)

    def __len__(self):
        return self.num_ps
    
    def __getitem__(self, idx):
        return self.user_item_df['user_id'][idx], self.user_item_df['item_id'][idx]

class User_Item_Dataset(Dataset):
    def __init__(self, user_p, item_p, user_item_p, num_ns):
        self.user_p = user_p
        self.item_p = item_p
        self.user_item_p = user_item_p
        self.num_ns = num_ns

        self.user_item_df = pd.read_csv(user_item_p)
        self.user_df = pd.read_csv(user_p, index_col=0)
        self.item_df = pd.read_csv(item_p, index_col=0)

        self.user_item_dict = defaultdict(list)
        for _, r in self.user_item_df.iterrows():
            self.user_item_dict[r['user_id']].append(r['item_id'])
        self.num_ps = len(self.user_item_df)
        self.num_user = len(self.user_df)
        self.num_item = len(self.item_df)
        self.samples = []

    def __len__(self):
        return len(self.samples)

    def generate_samples(self, sampler):
        ns_item_id = np.random.choice(self.item_df.index, size = 15*self.num_ps*self.num_ns)
        ns_item_idx = 0

        for _ , r in self.user_item_df.iloc[sampler, :].iterrows():
            for i in range(self.num_ns):
                while True:
                    #if self.item_df['item_id'][ns[ns_idx]]  not in self.user_item_dict[r['user_id']]:
                    if ns_item_id[ns_item_idx]  not in self.user_item_dict[r['user_id']]:
                        break
                    else:
                        ns_item_idx+=1
                self.samples.append((r['user_id'], r['item_id'], ns_item_id[ns_item_idx]))
        

    def __getitem__(self, idx): 
        user_id, pos_item_id, neg_item_id = self.samples[idx][0], self.samples[idx][1], self.samples[idx][2]
        return self.user_df.loc[user_id].to_numpy(), self.item_df.loc[pos_item_id].to_numpy(), self.item_df.loc[neg_item_id].to_numpy()
