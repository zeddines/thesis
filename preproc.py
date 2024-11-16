import pandas as pd
import numpy as np
import os

user_id_c = 3
item_id_c = 4
#item feature dim = 32 (col 5-36)
item_f_c = slice(5,36)
#user feature dim = 43 (col 37-79)
user_f_c = slice(37, 79)
l_c = 80

path = os.path.join(os.sep, "media", "zeddines", "9A3D-B0B1", "data")

df = pd.read_csv(os.path.join(path, "ipv_events_for_train_offline.csv"), index_col=0, sep=r'[\t,]', header=None, engine='python')

column_name = {user_id_c:"user_id", item_id_c:"item_id"}
column_name.update(dict(zip(range(5,37), [f"item_f_{i}" for i in range(0, 32)])))
column_name.update(dict(zip(range(37,80), [f"user_f_{i}" for i in range(0, 43)])))
column_name.update({l_c: "label"})

#filter label that is -1 (this dataset no -1)
#df = df.loc[df[l_c] == -1]
#seperate into user and item csv

user_item = pd.concat([df[user_id_c], df[item_id_c]], axis=1)
user_item.rename(columns = column_name, inplace = True)
user_item.to_csv(os.path.join(path, "user_item_train_offline.csv"), index = False)

'''
user = pd.concat([df[user_id_c], df.loc[:, user_f_c]], axis = 1)
user = user.loc[~user[user_id_c].duplicated(), :]
user.sort_values(user_id_c, inplace = True)
user.rename(columns = column_name, inplace = True)
user.to_csv(os.path.join(path, "user_train_offline.csv"), index = False)

item = pd.concat([df[item_id_c], df.loc[:, item_f_c]], axis = 1)
item = item.loc[~item[item_id_c].duplicated(), :]
item.sort_values(item_id_c, inplace = True)
item.rename(columns = column_name, inplace = True)
item.to_csv(os.path.join(path, "item_train_offline.csv"), index = False)
'''
