from problem2.preprocessing import Datatransform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.utils.data as Data
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from data import data_preprocessing
plt.style.use('seaborn')
data=pd.read_excel('D:\\美赛\\data\\frequency.xlsx',sheet_name=0)
data=data_preprocessing.Data_used(data)
scale=StandardScaler()
fr=data.frequency_raw.reshape(-1,1)
vowel=data.vowel.reshape(-1,1)
repeat=data.repetition.reshape(-1,1)
sc=data.repe_score.reshape(-1,1)
poh=data.poh.reshape(-1,1)
dif=data.difficulty.reshape(-1,1)
X=np.concatenate([dif,poh,fr,vowel,repeat,sc],axis=1)
X_train_s=scale.fit_transform(X)
word_att=pd.DataFrame(data=X_train_s,columns=['difficulty','percentage_of_hard','frequency','vowel','repeat_number','repeat_score'])
datacor=np.corrcoef(word_att.values,rowvar=0)
datacor=pd.DataFrame(data=datacor,columns=word_att.columns,index=word_att.columns)
plt.figure(figsize=(8,6))
ax=sns.heatmap(datacor,square=True,annot=True,fmt='.3f',linewidths=.5,cmap='YlGnBu',cbar_kws={'fraction':0.046,'pad':0.03})
plt.show()
