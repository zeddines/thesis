import pandas as pd
import os


path = os.path.join(os.sep, "media", "zeddines", "9A3D-B0B1", "data")

df = pd.read_csv(os.path.join(path, 'user_train_offline.csv'), index_col=0)
print(df.loc[23])
print(type(df.loc[23]))

print(next(df.iterrows()))
print(type(df.index))
print(df.loc[23])
print(df.index[1])
