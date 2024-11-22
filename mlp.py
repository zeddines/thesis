import torch
from torch.utils.data import DataLoader
import os
from data import User_Item_Dataset, User_Item_Train_Dataset, User_Item_Valid_Dataset
import torch.optim as optim
from model import MLP
from tqdm import tqdm
import time
from sklearn.model_selection import KFold
from loss import BPR
import numpy as np
from collections import defaultdict
import math

######parameters######
# data paths
user_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'user_train_offline.csv')
item_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'item_train_offline.csv')
user_item_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'data', 'user_item_train_offline.csv')

# save model
save_model_path = os.path.join(os.sep, 'media', 'zeddines', '9A3D-B0B1', 'model')

# training
num_ns = 3
seed = 1
in_dim = 75 # 32 item feauture 43 user feature
epochs = 1000
batch_size = 32
learning_rates = [0.1, 0.01, 0.001, 0.0001]
# early_stopping = True
# patience_value = 5

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



dataset = User_Item_Dataset(user_item_path)
dataset.generate_train_valid_split()

dataset_train = User_Item_Train_Dataset(user_path, item_path, dataset, num_ns)

dataset_valid = User_Item_Valid_Dataset(user_path, item_path, dataset, num_ns)
dataset_valid.generate_candidate()



model = MLP(in_dim)
model = model.to(device)
optimizer = optim.Adam(model.parameters())

timer_start = time.time()

     
epoch_loss_list = []
epoch_eval_list = [] # list of (mean ndcg_at_K, mean_hr_at_K) mean ndcg_at_K/hr_at_K is dict with keys = at_k
epoch_eval_best = {'ndcg': {k: (-1, -100) for k in at_K}, 'hr': {k: (-1, -100) for k in at_K}} # value of inner dict is (epoch, value)

# index is epoch, value is tuple of metrics - hr and ndcg
for epoch_num in tqdm(range(epochs)):
    epoch_train_start = time.time()

    # train
    dataset_train.generate_train_samples()
    dataloader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = True, num_workers = 4, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
    model.train()
    batch_loss_list = []
    for users, pos_items, neg_items in dataloader_train:
        pos_samples = torch.cat((users, pos_items), dim = 1)
        neg_samples = torch.cat((users, neg_items), dim = 1)

        pos_samples = pos_samples.to(device)
        neg_samples = neg_samples.to(device)

        pos_score, neg_score = model(pos_samples, neg_samples)
        loss = BPR(pos_score, neg_score)

        batch_loss_list.append(loss)
        # update
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()



    epoch_train_loss = torch.mean(torch.stack(batch_loss_list)).item
    epoch_loss_list.append(epoch_train_loss)
    epoch_train_time = time.time() - epoch_train_start
    print(f"epoch {epoch_num}, training time taken = {epoch_train_time}, BPR loss = {epoch_train_loss} ")

    # evaluation
    model.eval()
    #TODO run model for every user
    
    epoch_valid_start = time.time()
    with torch.no_grad():
        #hr_at_K = defaultdict(lambda: dict.fromkeys(at_K, 0.0))
        hr_at_K = defaultdict(lambda: {k: 0.0 for k in at_K})
        #ndcg_at_K = defaultdict(lambda: dict.fromkeys(at_K, 0.0))
        ndcg_at_K = defaultdict(lambda: {k: 0.0 for k in at_K})
        for user_id in dataset_valid.get_valid_users():
            dataset_valid.generate_valid_samples(user_id)
            pos_items = dataset_valid.get_positive_items(user_id) 
            #shuffle needs to be false
            dataloader_valid = DataLoader(dataset_valid, batch_size = batch_size, shuffle = False, num_workers = 4, worker_init_fn=lambda worker_id: np.random.seed(seed + worker_id))
            scores = [] 
            candidates_id = []
            for user, batch_candidates, batch_candidates_id in dataloader_valid:
                # user = 2d tensor, same along dim 1 as it is same user
                # batch_candidates = 2d tensor
                # batch_candidates_id = 1d tensor
                samples = torch.cat((user, batch_candidates), dim = 1)
                samples.to(device)
                batch_score = model.predict(samples)
                candidates_id.append(batch_candidates_id)
                scores.append(batch_score)
                # sort by score
            
            scores = torch.cat(scores, dim = 0) 
            candidates_id = torch.cat(candidates_id, dim = 0)

            sorted_idx = torch.argsort(scores)
            sorted_candidates_id = candidates_id[sorted_idx]
            candidates_ranking = sorted_candidates_id.numpy()

            # evalauate top K recommendations based on dataset_valid
            for k in at_K:
                hr = 0
                dcg = 0
                for r, cid in enumerate(candidates_ranking):
                    if r == k:
                        break
                    if cid in pos_items:
                        hr_at_K[user_id][k] = 1
                        dcg += 1/math.log2(r+1)

                idcg = sum([1/math.log2(i+1) for i in range(1, len(pos_items)+1)])
                ndcg_at_K[user_id][k] = dcg/idcg

        num_users = len(dataset_valid.get_valid_users())
        mean_ndcg_at_K = {k: 0.0 for k in at_K} 
        mean_hr_at_K = {k: 0.0 for k in at_K}

        for k in at_K:
            for user_id in dataset_valid.get_valid_users():
                mean_ndcg_at_K[k] += ndcg_at_K[user_id][k]
                mean_hr_at_K[k] += hr_at_K[user_id][k]

            mean_ndcg_at_K[k] = mean_ndcg_at_K[k]/len(dataset_valid.get_valid_users())
            mean_hr_at_K[k] = mean_hr_at_K[k]/len(dataset_valid.get_valid_users())

        epoch_eval_list.append((mean_ndcg_at_K, mean_hr_at_K))
        epoch_valid_time = time.time() - epoch_valid_start

        
        # print epoch results
        print(f"epoch {epoch_num}, validation time taken = {epoch_valid_time}, mean ndcg@{at_K}: {mean_ndcg_at_K.values()}, mean hr@{at_K}: {mean_hr_at_K.values()}")

        # epoch_eval_best = {'ndcg': dict.fromkeys(at_K), 'hr': dict.fromkeys(at_K)} # value of inner dict is (epoch, value)
        for k in at_K:
            if mean_ndcg_at_K[k] > epoch_eval_best['ndcg'][k][1]:
                epoch_eval_best['ndcg'][k] = (epoch_num, mean_ndcg_at_K[k])
                # save model
                torch.save(model.state_dict(), os.path.join(save_model_path, "e{epoch_num},ndcg_best.pt")) 
            if mean_hr_at_K[k] > epoch_eval_best['hr'][k][1]:
                epoch_eval_best['hr'][k] = (epoch_num, mean_hr_at_K[k])
                
timer_end = time.time()
print(f'total training time taken = {timer_end-timer_start}')



