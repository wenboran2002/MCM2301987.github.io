import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from data import data_preprocessing
import sklearn.preprocessing as pre
plt.style.use('seaborn')
data=pd.read_excel('D:\\美赛\\data\\frequency.xlsx',sheet_name=0)
data=data_preprocessing.Data_used(data)
reweight_rv=data.vowel*0.25+data.repetition*0.5+data.repe_score*0.25
reweight_rv=np.array(reweight_rv).reshape(-1,1)
reweight_rv[reweight_rv<=2]*=0.8
reweight_rv[reweight_rv>2]*=1.2
fr=data.frequency.reshape(-1,1)
fr[fr<=5]*=0.8
fr[fr>5]*=1.2
print(reweight_rv.shape,fr.shape)
X=np.concatenate([reweight_rv,fr,data.poh.reshape(-1,1)],axis=1)
print(X[:5])
plt.scatter(X[:,0],X[:,1])
plt.show()


from sklearn.cluster import DBSCAN
# dbscan = DBSCAN(eps = 1,min_samples=10)
# dbscan.fit(X)
dbscan2 = DBSCAN(eps = 1,min_samples=10)
dbscan2.fit(X[:,:2])

def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1],
                c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)
plt.figure(figsize=(6, 6))

# plt.subplot(121)
# plot_dbscan(dbscan, X, size=100)
#
# plt.subplot(122)
print(dbscan2.labels_)
plot_dbscan(dbscan2, X[:,:2], size=600, show_ylabels=False)
poh_1=[]
poh_2=[]
poh_3=[]
poh_4=[]
for i in range(X.shape[0]):
    if dbscan2.labels_[i]==0:
        poh_1.append(X[i,2])
    if dbscan2.labels_[i] == 1:
        poh_2.append(X[i,2])
    if dbscan2.labels_[i]==2:
        poh_3.append(X[i,2])
    if dbscan2.labels_[i]==3:
        poh_4.append(X[i,2])
plt.show()
print(np.mean(poh_1),np.mean(poh_2),np.mean(poh_3),np.mean(poh_4))
