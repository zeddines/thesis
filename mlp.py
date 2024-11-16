import torch
from torch.utils.data import DataLoader
import os
from data import User_Item_Dataset, User_Item_ID
import torch.optim as optim
from model import MLP
from tqdm import tqdm
import time
from sklearn.model_selection import KFold
from loss import BPR
import numpy as np

######parameters######
# data paths
user_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'user_train_offline.csv')
item_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'item_train_offline.csv')
user_item_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'user_item_train_offline.csv')

# training
num_ns = 3
seed = 1
in_dim = 75 # 32 item feauture 43 user feature
epochs = 1000
batch_size = 32
k_fold = 5

# evaluation
at_K = [5, 10, 20]
######################
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)

if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_reserved(0)/1024**3,1), 'GB')


# TODO add dataset for testing
dataset_train_ps = User_Item_ID(user_item_path)
dataset_train = User_Item_Dataset(user_path, item_path, user_item_path, num_ns)


kf = KFold(n_splits = k_fold, shuffle = True, random_state = seed)

model = MLP(in_dim)
model = model.to(device)
optimizer = optim.Adam(model.parameters())

timer_start = time.time()

for fold_num, (train_idx, valid_idx) in tqdm(enumerate(kf.split(dataset_train_ps))):
    dataset_train.generate_samples(train_idx)
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
     
    epoch_loss_list = []

    for epoch_num in tqdm(range(epochs)):
        model.train()
        for users, pos_items, neg_items in dataloader_train:
            pos_samples = torch.cat((users, pos_items), dim = 1)
            neg_samples = torch.cat((users, neg_items), dim = 1)

            pos_samples = pos_samples.to(device)
            neg_samples = neg_samples.to(device)

            loss = model(pos_samples, neg_samples)

            # update
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            

        # evaluation
        


timer_end = time.time()
print(f'training time taken = {timer_end-timer_start}')

# prediction


# training loop TODO
#user = x[0]
#pos_item = x[1]
#neg_item = x[2]
#
#pos_samples = torch.cat((user, pos_item), dim = 1)
#neg_samples = torch.cat((user, neg_item), dim = 1)
