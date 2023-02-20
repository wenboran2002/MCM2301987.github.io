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
from prettytable import PrettyTable
data=pd.read_excel('D:\\美赛\\data\\frequency.xlsx')
words=pd.read_csv('D:\美赛\problem3\word_embedding.csv')
# print(Difficulty_transform(data))
# train_x=Datatransform(np.array(data['Word']))[:251]
w=np.array(words.iloc[:,1:].values).astype(np.float32)

train_x1=torch.from_numpy(Datatransform(np.array(data['Word']))[:251])
test_x1=torch.from_numpy(Datatransform(np.array(data['Word']))[251:])
train_x2=torch.from_numpy(feature_transform(data)[:251])
test_x2=torch.from_numpy(feature_transform(data)[251:])
train_x3=torch.from_numpy(w[:251])
test_x3=torch.from_numpy(w[251:])

train_y=torch.from_numpy(Difficulty_transform(data)[:251].astype(np.float32))
test_y=torch.from_numpy(Difficulty_transform(data)[251:].astype(np.float32))


train_data1=Data.TensorDataset(train_x1,train_y)
train_data2=Data.TensorDataset(train_x2,train_y)
train_data3=Data.TensorDataset(train_x3,train_y)
test_data1=Data.TensorDataset(test_x1,test_y)
test_data2=Data.TensorDataset(test_x2,test_y)
test_data3=Data.TensorDataset(test_x3,test_y)
train_loader1=Data.DataLoader(dataset=train_data1,batch_size=64,shuffle=True,num_workers=0)
train_loader2=Data.DataLoader(dataset=train_data2,batch_size=64,shuffle=True,num_workers=0)
train_loader3=Data.DataLoader(dataset=train_data3,batch_size=64,shuffle=True,num_workers=0)
class MLPregression(nn.Module):
    def __init__(self):
        super(MLPregression,self).__init__()
        self.mlp1=nn.Sequential(
            nn.Linear(in_features=130, out_features=100, bias=True),
            nn.Linear(100, 50),
            nn.Linear(50, 50),
            nn.Linear(50, 10)
        )
        self.mlp2 = nn.Sequential(
            nn.Linear(in_features=3, out_features=20, bias=True),
            nn.Linear(20, 10),
            nn.Linear(10, 10),
            nn.Linear(10, 10)
        )
        self.mlp3 = nn.Sequential(
            nn.Linear(in_features=300, out_features=300, bias=True),
            nn.Linear(300, 100),
            nn.Linear(100, 100),
            nn.Linear(100, 10)
        )
        self.classification=nn.Sequential(
            nn.Linear(10,3),
            nn.Sigmoid()
        )
        self.alpha=nn.Parameter(torch.tensor([0.5]),requires_grad=True)
        self.beta=nn.Parameter(torch.tensor([0.5]),requires_grad=True)
        self.hidden=nn.Linear(10,10)
        self.m=nn.BatchNorm1d(10)

    def forward(self,x1,x2,x3):
        x1=self.mlp1(x1)
        x2=self.mlp2(x2)
        x3=self.mlp3(x3)
        x=x1+self.alpha*x2+self.beta*x3
        x=self.hidden(x)
        x = self.m(x)
        x=F.relu(x)
        nn.Dropout(0.5)
        output=self.classification(x)
        return output
mlpreg=MLPregression()
# mlpc=MLPregression()
# x=torch.randn(1,131).requires_grad_(True)
# y=mlpc(x)
# Mymlpcvis=make_dot(y,params=dict(list(mlpc.named_parameters())+[('x',x)]))
# Mymlpcvis.view('classification_model.svg','D:\\美赛\\figures')
optimizer=torch.optim.AdamW(mlpreg.parameters(),lr=1e-2,weight_decay=0.01)
loss_func=nn.SmoothL1Loss()
train_loss_all=[]
accuracy=[]
for epoch in range(10):
    train_loss=0
    train_num=0
    correct=0
    for (b_x1,b_y1),(b_x2,b_y2),(b_x3,b_y3) in zip(train_loader1,train_loader2,train_loader3):
        output=mlpreg(b_x1,b_x2,b_x3)
        loss=loss_func(output,b_y1)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss+=loss.item()*b_x1.size(0)
        train_num+=b_x1.size(0)
        pre=torch.argmax(output,1)
        lable=torch.argmax(b_y1,1)
        correct+=(pre==lable).sum().item()
        # print(output)
    print('loss:',train_loss/train_num,correct/train_num)
    train_loss_all.append(train_loss/train_num)
    accuracy.append(correct/train_num)

#weights
# for name in mlpreg.state_dict():
#     print(name)
#
# print(mlpreg.state_dict()['hidden1_2.weight'])
# print(mlpreg.state_dict()['alpha'])
# print(mlpreg.state_dict()['beta'])

# m=mlpreg.state_dict()['hidden1_2.weight'].data.numpy()
# d=pd.DataFrame(m)
# d.to_csv('./weight.csv')

# visualize loss function
plt.style.use('seaborn')
plt.plot(train_loss_all,'r-o',label='Train loss')
plt.plot(accuracy,'b-o',label='Accuracy')
plt.legend()
plt.xlabel('epoch')
plt.ylabel('Loss')
# plt.show()
print(test_x1.shape)
pre_y=mlpreg(test_x1,test_x2,test_x3)
pre_y=pre_y.data.numpy()
test_y=test_y.data.numpy()
# print(test_y,pre_y)
label=np.argmax(test_y,axis=1).reshape(-1,1)
correct=np.sum(np.argmax(pre_y,axis=1).reshape(-1,1)==label)
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
label_dict={'0':'hard','1':'normal','2':'easy'}
label_dict = [label for _, label in label_dict.items()]
class ConfusionMatrix(object):

    def __init__(self, num_classes: int, labels: list):
        self.matrix = np.zeros((num_classes, num_classes))  # 初始化混淆矩阵，元素都为0
        self.num_classes = num_classes  # 类别数量，本例数据集类别为5
        self.labels = labels  # 类别标签

    def update(self, preds, labels):
        for p, t in zip(preds, labels):  # pred为预测结果，labels为真实标签
            self.matrix[p, t] += 1  # 根据预测结果和真实标签的值统计数量，在混淆矩阵相应位置+1

    def summary(self):  # 计算指标函数
        # calculate accuracy
        sum_TP = 0
        n = np.sum(self.matrix)
        for i in range(self.num_classes):
            sum_TP += self.matrix[i, i]  # 混淆矩阵对角线的元素之和，也就是分类正确的数量
        acc = sum_TP / n  # 总体准确率
        print("the model accuracy is ", acc)

        # kappa
        sum_po = 0
        sum_pe = 0
        for i in range(len(self.matrix[0])):
            sum_po += self.matrix[i][i]
            row = np.sum(self.matrix[i, :])
            col = np.sum(self.matrix[:, i])
            sum_pe += row * col
        po = sum_po / n
        pe = sum_pe / (n * n)
        # print(po, pe)
        kappa = round((po - pe) / (1 - pe), 3)
        # print("the model kappa is ", kappa)

        # precision, recall, specificity
        table = PrettyTable()  # 创建一个表格
        table.field_names = ["", "Precision", "Recall", "Specificity"]
        for i in range(self.num_classes):  # 精确度、召回率、特异度的计算
            TP = self.matrix[i, i]
            FP = np.sum(self.matrix[i, :]) - TP
            FN = np.sum(self.matrix[:, i]) - TP
            TN = np.sum(self.matrix) - TP - FP - FN

            Precision = round(TP / (TP + FP), 3) if TP + FP != 0 else 0.
            Recall = round(TP / (TP + FN), 3) if TP + FN != 0 else 0.  # 每一类准确度
            Specificity = round(TN / (TN + FP), 3) if TN + FP != 0 else 0.

            table.add_row([self.labels[i], Precision, Recall, Specificity])
        print(table)
        return str(acc)

    def plot(self):  # 绘制混淆矩阵
        matrix = self.matrix
        print(matrix)
        plt.imshow(matrix, cmap=plt.cm.Blues)

        # 设置x轴坐标label
        plt.xticks(range(self.num_classes), self.labels, rotation=45)
        # 设置y轴坐标label
        plt.yticks(range(self.num_classes), self.labels)
        # 显示colorbar
        plt.colorbar()
        plt.xlabel('True Labels')
        plt.ylabel('Predicted Labels')
        plt.title('Confusion matrix (acc=' + self.summary() + ')')

        # 在图中标注数量/概率信息
        thresh = matrix.max() / 2
        for x in range(self.num_classes):
            for y in range(self.num_classes):
                # 注意这里的matrix[y, x]不是matrix[x, y]
                info = int(matrix[y, x])
                plt.text(x, y, info,
                         verticalalignment='center',
                         horizontalalignment='center',
                         color="white" if info > thresh else "black")
        plt.tight_layout()
        plt.show()

confusion=ConfusionMatrix(num_classes=3,labels=label_dict)
confusion.update(np.argmax(pre_y,axis=1).reshape(-1,1),labels=label)
# confusion.plot()
# plt.show()
eerie=pd.read_csv('D:\美赛\problem3\eerie.csv')
eerie=np.array(eerie.values[:,-1]).reshape(1,-1).astype(np.float32)
print(eerie[-1].shape)
feature_=torch.from_numpy(np.array([1.6,0.48399,0.4]).reshape(-1,3).astype(np.float32))
pre_y_spoke=mlpreg(torch.from_numpy(Datatransform(['eerie'])),feature_,torch.from_numpy(eerie))
print(pre_y_spoke)
mlpreg.predict(spoke)