import numpy as np
import pandas as pd
import os
from collections import defaultdict
import matplotlib.pyplot as plt
"""
user_id_c = 3
item_id_c = 4
#item feature dim = 32 (col 5-36)
item_f_c = slice(5,36)
#user feature dim = 43 (col 37-79)
user_f_c = slice(37, 79)
l_c = 80
"""

data_dir = os.path.join(os.sep, "media", "zeddines", "9A3D-B0B1", "data")

df_users = pd.read_csv(os.path.join(data_dir, "user_train_offline.csv"), engine = "python")
df_items = pd.read_csv(os.path.join(data_dir, "item_train_offline.csv"), engine = "python")
df = pd.read_csv(os.path.join(data_dir, "ipv_events_for_train_offline.csv"), index_col = 0, header = None, sep = r'[\t,]', engine = "python")

user_item_pos = defaultdict(list)

# includ malicious clicks
for index, row in df.iterrows():
    user_id = row[3]
    item_id = row[4]
    
    user_item_pos[user_id].append(item_id) 

freq_count = [len(items) for items in user_item_pos.values()]
print(f"total number of items = {len(df_items.index)}")
print(f"max pos samples = {max(freq_count)}, min pos samples {min(freq_count)}")

plt.hist(freq_count, range=(0, 100), bins=101)
plt.gca().set(title='Frequency Histogram', ylabel='Frequency');
plt.show()

