import pandas as pd
import numpy as np
words=pd.read_csv('D:\美赛\problem3\word_embedding.csv')
print(words.head())
w=words.iloc[:,1:].values
print(w.shape)