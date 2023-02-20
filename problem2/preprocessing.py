import numpy as np
import pandas as pd
def Datatransform(word_list):
    n=len(word_list)
    out=np.zeros((n,5,26))
    for i in range(n):
        for j in range(5):
            for k in range(ord('a'),ord('z')+1):
               if word_list[i][j]==chr(k):
                   out[i][j][k-ord('a')]=1
    out=out.reshape(n,-1)
    return out.reshape(n,-1).astype(np.float32)

def Difficulty_transform(feature_list):
    data=feature_list.copy()
    data[data['difficulty'] <= 250] = 0
    data[(data['difficulty'] > 250) & (data['difficulty'] <= 310)] = 1
    data[data['difficulty'] > 310] = 2
    feature=pd.get_dummies(data, columns=['difficulty'])
    out=np.array(feature.loc[:,['difficulty_0','difficulty_1','difficulty_2']]).reshape(-1,3)
    return out.astype(np.float32)

def normalization(data):
    return (data-np.min(data))/(np.max(data)-np.min(data))
def feature_transform(data):
    data=data.copy()
    score=np.array(data['score']).astype(np.float32)
    fr=np.array(data['functionoffrequency']).astype(np.float32)
    vowel=np.array(data['vowel']).astype(np.float32)
    score=score.reshape(-1,1)
    fr=(fr/10).reshape(-1,1)
    vowel=(vowel/4).reshape(-1,1)
    x=np.concatenate([score,fr,vowel],axis=1)
    return x.reshape(-1,3)



