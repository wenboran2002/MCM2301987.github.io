from problem2.preprocessing import Datatransform
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
import torch.utils.data as Data
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error,mean_absolute_error
data=pd.read_excel('D:\美赛\data\Problem_C_Data_Wordle.xlsx')
train_x=Datatransform(np.array(data['Word']))[:251]
train_y=np.array(data.iloc[:251,5:12]/100)
test_x=Datatransform(np.array(data['Word']))[251:]
test_y=np.array(data.iloc[251:,5:12]/100)


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
        self.hidden2=nn.Linear(100,100)
        self.hidden3=nn.Linear(100,50)
        self.predict=nn.Linear(50,7)
    def forward(self,x):
        x=F.relu(self.hidden1(x))
        x=F.relu(self.hidden2(x))
        x=F.relu(self.hidden3(x))
        output=self.predict(x)
        return output
mlpreg=MLPregression()
optimizer=torch.optim.Adam(mlpreg.parameters(),lr=0.01)
loss_func=nn.MSELoss()
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

# spoke=np.zeros((5,26))
# spoke[0][18]=1
# spoke[1][15]=1
# spoke[2][14]=1
# spoke[3][10]=1
# spoke[4][4]=1
# spoke=spoke.reshape(1,130)
spoke=torch.from_numpy(Datatransform(['eerie']).astype(np.float32))
print(spoke.shape)
pre_y_s=mlpreg(spoke)
print(pre_y_s)
# pre_y_spoke=mlpreg(torch.from_numpy(spoke))
# print(pre_y_spoke)
# mlpreg.predict(spoke)