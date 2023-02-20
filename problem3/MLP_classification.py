from problem2.preprocessing import Datatransform,Difficulty_transform
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
print(Difficulty_transform(data))
# train_x=Datatransform(np.array(data['Word']))[:251]
t=True
train_x=Datatransform(np.array(data['Word']))[:251]
train_y=Difficulty_transform(data)[:251]
test_x=Datatransform(np.array(data['Word']))[251:]
test_y=Difficulty_transform(data)[251:]
train_xt=torch.from_numpy(train_x.astype(np.float32))
train_yt=torch.from_numpy(train_y.astype(np.float32))
test_xt=torch.from_numpy(test_x.astype(np.float32))
test_yt=torch.from_numpy(test_y.astype(np.float32))
train_data=Data.TensorDataset(train_xt,train_yt)
test_data=Data.TensorDataset(test_xt,test_yt)
train_loader=Data.DataLoader(dataset=train_data,batch_size=64,shuffle=True,num_workers=0)
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        self.hidden1=nn.Linear(in_features=130,out_features=100,bias=True)
        self.hidden1_2=nn.Linear(in_features=3,out_features=20,bias=True)
        self.hidden2=nn.Linear(100,50)
        self.hidden2_2=nn.Linear(20,10)
        self.hidden3=nn.Linear(50,10)
        self.classification=nn.Sequential(
            nn.Linear(10,3),
            nn.Sigmoid()
        )
    def forward(self,x1,x2):
        x1=F.relu(self.hidden1(x1))
        x2=F.relu(self.hidden1_2(x2))
        x1=F.relu(self.hidden2(x1))
        x2=F.relu(self.hidden2_2(x2))
        x1=F.relu(self.hidden3(x1))
        x=x1+x2
        output=self.classification(x)
        return output


mlpreg=MLPregression()
# mlpc=MLPregression()
# x=torch.randn(1,130).requires_grad_(True)
# y=mlpc(x)
# Mymlpcvis=make_dot(y,params=dict(list(mlpc.named_parameters())+[('x',x)]))
# Mymlpcvis.view('classification_model.svg','D:\\美赛\\figures')
optimizer=torch.optim.AdamW(mlpreg.parameters(),lr=0.001,weight_decay=0.01)
loss_func=nn.SmoothL1Loss()
train_loss_all=[]
for epoch in range(200):
    train_loss=0
    train_num=0
    for step,(b_x,b_y) in enumerate(train_loader):
        output=mlpreg(b_x)
        loss=loss_func(output,b_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*b_x.size(0)
        train_num+=b_x.size(0)
    print('loss:',train_loss/train_num)
    train_loss_all.append(train_loss/train_num)

# visualize loss function
plt.style.use('seaborn')
plt.plot(train_loss_all,'r-',label='Train loss')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
# plt.show()
print(test_xt.shape)
pre_y=mlpreg(test_xt)
pre_y=pre_y.data.numpy()
mae=mean_absolute_error(test_y,pre_y)
print('测试集绝对值误差:',mae)

spoke=torch.from_numpy(Datatransform(['spoke']).astype(np.float32))
print(spoke.shape)
pre_y_s=mlpreg(spoke)
print(pre_y_s)

# pre_y_spoke=mlpreg(torch.from_numpy(spoke))
# print(pre_y_spoke)
# mlpreg.predict(spoke)