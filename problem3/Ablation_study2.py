from problem2.preprocessing import Datatransform,Difficulty_transform,feature_transform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
from torchviz import make_dot
data=pd.read_excel('D:\\美赛\\data\\frequency.xlsx')
words=pd.read_csv('D:\美赛\problem3\word_embedding.csv')
# print(Difficulty_transform(data))
# train_x=Datatransform(np.array(data['Word']))[:251]
w=np.array(words.iloc[:,1:].values).astype(np.float32)

train_x1=torch.from_numpy(Datatransform(np.array(data['Word']))[:251])
test_x1=torch.from_numpy(Datatransform(np.array(data['Word']))[251:])
# train_x2=torch.from_numpy(feature_transform(data)[:251])
# test_x2=torch.from_numpy(feature_transform(data)[251:])
train_x3=torch.from_numpy(w[:251])
test_x3=torch.from_numpy(w[251:])

train_y=torch.from_numpy(Difficulty_transform(data)[:251].astype(np.float32))
test_y=torch.from_numpy(Difficulty_transform(data)[251:].astype(np.float32))


train_data1=Data.TensorDataset(train_x1,train_y)
# train_data2=Data.TensorDataset(train_x2,train_y)
train_data3=Data.TensorDataset(train_x3,train_y)
test_data1=Data.TensorDataset(test_x1,test_y)
# test_data2=Data.TensorDataset(test_x2,test_y)
test_data3=Data.TensorDataset(test_x3,test_y)
train_loader1=Data.DataLoader(dataset=train_data1,batch_size=64,shuffle=True,num_workers=0)
# train_loader2=Data.DataLoader(dataset=train_data2,batch_size=64,shuffle=True,num_workers=0)
train_loader3=Data.DataLoader(dataset=train_data3,batch_size=64,shuffle=True,num_workers=0)
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        self.hidden1_1=nn.Linear(in_features=130,out_features=100,bias=True)
        self.hidden1_2=nn.Linear(in_features=3,out_features=20,bias=True)
        self.hidden1_3=nn.Linear(in_features=300,out_features=300,bias=True)
        self.hidden2_1=nn.Linear(100,50)
        self.hidden2_2=nn.Linear(20,10)
        self.hidden2_3=nn.Linear(300,100)
        self.hidden3_1=nn.Linear(50,10)
        self.hidden3_3=nn.Linear(100,10)
        self.classification=nn.Sequential(
            nn.Linear(10,3),
            nn.Sigmoid()
        )
        self.alpha=nn.Parameter(torch.tensor([0.5]),requires_grad=True)
        self.beta=nn.Parameter(torch.tensor([0.5]),requires_grad=True)
    def forward(self,x1,x3):
        x1=F.relu(self.hidden1_1(x1))
        # x2=F.relu(self.hidden1_2(x2))
        x3=F.relu(self.hidden1_3(x3))
        x1=F.relu(self.hidden2_1(x1))
        # x2=F.relu(self.hidden2_2(x2))
        x3=F.relu(self.hidden2_3(x3))
        x1=F.relu(self.hidden3_1(x1))
        x3=F.relu(self.hidden3_3(x3))
        x=x1+self.beta*x3
        output=self.classification(x)
        return output
mlpreg=MLPregression()
# mlpc=MLPregression()
# x=torch.randn(1,131).requires_grad_(True)
# y=mlpc(x)
# Mymlpcvis=make_dot(y,params=dict(list(mlpc.named_parameters())+[('x',x)]))
# Mymlpcvis.view('classification_model.svg','D:\\美赛\\figures')
optimizer=torch.optim.AdamW(mlpreg.parameters(),lr=0.001,weight_decay=0.01)
loss_func=nn.SmoothL1Loss()
train_loss_all=[]
for epoch in range(200):
    train_loss=0
    train_num=0
    for (b_x1,b_y1),(b_x3,b_y3) in zip(train_loader1,train_loader3):
        output=mlpreg(b_x1,b_x3)
        loss=loss_func(output,b_y1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*b_x1.size(0)
        train_num+=b_x1.size(0)
    print('loss:',train_loss/train_num)
    train_loss_all.append(train_loss/train_num)

# visualize loss function
plt.style.use('seaborn')
plt.plot(train_loss_all,'r-',label='Train loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
# plt.show()
print(test_x1.shape)
pre_y=mlpreg(test_x1,test_x3)
pre_y=pre_y.data.numpy()
test_y=test_y.data.numpy()
# print(test_y,pre_y)
correct=np.sum(np.argmax(pre_y,axis=1).reshape(-1,1)==np.argmax(test_y,axis=1).reshape(-1,1))
accuracy=correct/len(test_y)
print(accuracy)
mae=mean_absolute_error(test_y,pre_y)
print('validation absolute error:',mae)

spoke=torch.from_numpy(Datatransform(['spoke']).astype(np.float32))
spoke_f = torch.from_numpy(feature_transform(data[data['Word']=='spoke']))
print(spoke.shape)
# pre_y_s=mlpreg(spoke,spoke_f)
# print(pre_y_s)

# pre_y_spoke=mlpreg(torch.from_numpy(spoke))
# print(pre_y_spoke)
# mlpreg.predict(spoke)