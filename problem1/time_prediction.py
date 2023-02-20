from sklearn.preprocessing import PolynomialFeatures

from data import data_preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from sklearn.model_selection import train_test_split
from mapie.regression import MapieRegressor
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
import torch
import torch.nn.functional as F
import torchvision
from torch import nn
data_excel=pd.read_excel('D:\美赛\data\Problem_C_Data_Wordle.xlsx')
data=data_preprocessing.Data_used(data_excel)
time=data_excel.copy()
time=time.set_index('Date')['Number of  reported results'][50:]

# print(time['Number of  reported results'])
plt.style.use('seaborn')
fig,ax=plt.subplots()
def time_sequence_process(data_raw):
    # plot original sequence:
    time.plot()
    plt.show()
    # Autocorrelation test:
    plot_acf(time)
    plt.show()

def mapie_predict(data):
    y=np.array(data[50:].values)
    X=np.arange(len(y)).reshape(-1,1)
    X_train_and_cal, X_test, y_train_and_cal, y_test = train_test_split(X, y, test_size=1 / 3,shuffle=False)
    # X_train, X_cal, y_train, y_cal = train_test_split(X_train_and_cal, y_train_and_cal, test_size=1 / 2,shuffle=False)
    # print(X_train.shape,y_train.shape)

    # mapie = MapieRegressor(estimator=model, cv="prefit").fit(X_cal, y_cal)
    lin_reg = LinearRegression()
    poly_features = PolynomialFeatures(degree=3, include_bias=False)
    polynomial_reg = Pipeline([('poly_features', poly_features),
                               ('lin_reg', lin_reg)])
    model=polynomial_reg.fit(X_train_and_cal, y_train_and_cal)
    # Get interval predictions on test data, with alpha=5%
    mapie = MapieRegressor(estimator=model, cv="prefit").fit(X_test, y_test)
    y_test_pred_interval = mapie.predict(np.arange(360,400).reshape(-1,1), alpha=.05)[1].T.reshape(2,-1)
    print(y_test_pred_interval)
    plt.plot(np.arange(360,400).reshape(-1,1),y_test_pred_interval[0])
    plt.plot(np.arange(360,400).reshape(-1,1),y_test_pred_interval[1])
    plt.show()
# mapie_predict(time)
# time_sequence_process(time)

class RegLSTM(nn.Module):
    def __init__(self, inp_dim, out_dim, mid_dim, mid_layers):
        super(RegLSTM, self).__init__()

        self.rnn = nn.LSTM(inp_dim, mid_dim, mid_layers)  # rnn
        self.reg = nn.Sequential(
            nn.Linear(mid_dim, mid_dim),
            nn.Tanh(),
            nn.Linear(mid_dim, out_dim),
        )  # regression

    def forward(self, x):
        y = self.rnn(x)[0]  # y, (h, c) = self.rnn(x)
        seq_len, batch_size, hid_dim = y.shape
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y

    def output_y_hc(self, x, hc):

        y, hc = self.rnn(x, hc)  # y, (h, c) = self.rnn(x)

        seq_len, batch_size, hid_dim = y.size()
        y = y.view(-1, hid_dim)
        y = self.reg(y)
        y = y.view(seq_len, batch_size, -1)
        return y, hc
def minmaxscaler(x):
    minx = np.amin(x)
    maxx = np.amax(x)
    return (x - minx)/(maxx - minx), (minx, maxx)

def preminmaxscaler(x, minx, maxx):
    return (x - minx)/(maxx - minx)

def unminmaxscaler(x, minx, maxx):
    return x * (maxx - minx) + minx

max_time=[]
min_time=[]
for i in range(0,300,3):
    max_time.append(max(time[i:i+3].values))
    min_time.append(min(time[i:i+3].values))
loss_list=[]
def cal(time,label_,color):
    bchain = np.array(time)
    bchain = bchain[:, np.newaxis]
    inp_dim = 1
    out_dim = 1
    mid_dim = 10
    mid_layers = 2
    data_x = bchain[:-1, :]
    data_y = bchain[+1:, :]
    # 第二种操作，用滑动窗口的方法构造数据集
    # train_size = 300
    train_size = 90
    train_x = data_x[:train_size, :]
    train_y = data_y[:train_size, :]
    train_x, train_x_minmax = minmaxscaler(train_x)
    train_y, train_y_minmax = minmaxscaler(train_y)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_x_tensor = torch.tensor(train_x, dtype=torch.float32, device=device)
    train_y_tensor = torch.tensor(train_y, dtype=torch.float32, device=device)
    # 开始构造滑动窗口  40个为1个窗口，step为3
    batch_x = list()
    batch_y = list()

    window_len = 80
    for end in range(len(train_x_tensor), window_len, -3):
        batch_x.append(train_x_tensor[end-40:end])
        batch_y.append(train_y_tensor[end-40:end])

    # batch_x的shape是(25, 40, 1)  25个时间序列，每个时间序列是40个时间步

    from torch.nn.utils.rnn import pad_sequence
    batch_x = pad_sequence(batch_x)
    batch_y = pad_sequence(batch_y)

    # batch_x的shape是(40, 25, 1)   输入模型的时候可以25个时间序列并行处理

    # 加载模型
    model = RegLSTM(inp_dim, out_dim, mid_dim, mid_layers).to(device)
    loss = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)

    # 开始训练
    print("Training......")
    for e in range(1000):
        out = model(batch_x)

        Loss = loss(out, batch_y)
        loss_list.append(Loss.item())
        optimizer.zero_grad()
        Loss.backward()
        optimizer.step()

        if e % 10 == 0:
            print('Epoch: {:4}, Loss: {:.5f}'.format(e, Loss.item()))
    # torch.save(model.state_dict(), './net.pth')
    # print("Save in:", './net.pth')
    print(data_x.shape)
    new_data_x = np.append(data_x.copy().flatten(),[0 for i in range(20)]).reshape(-1,1)
    # print(np.array(new_data_x).reshape(-1,1))
    # new_data_x=data_x.copy()
    new_data_x[train_size:] = 0

    test_len=80
    eval_size = 1
    zero_ten = torch.zeros((mid_layers, eval_size, mid_dim), dtype=torch.float32, device=device)
    for i in range(train_size, len(new_data_x)):  # 要预测的是i
        test_x = new_data_x[i-test_len:i, np.newaxis, :]
        test_x = preminmaxscaler(test_x, train_x_minmax[0], train_x_minmax[1])
        batch_test_x = torch.tensor(test_x, dtype=torch.float32, device=device)

        if i == train_size:
            test_y, hc = model.output_y_hc(batch_test_x, (zero_ten, zero_ten))
        else:
            test_y, hc = model.output_y_hc(batch_test_x[-2:], hc)
        test_y = model(batch_test_x)
        predict_y = test_y[-1].item()
        predict_y = unminmaxscaler(predict_y, train_x_minmax[0], train_y_minmax[1])
        new_data_x[i] = predict_y
    ax.plot(new_data_x, color, label=label_)
    return new_data_x

max_new_data=cal(max_time,'max','blue')
min_new_data=cal(min_time,'min','green')
print(min_new_data[-1],max_new_data[-1])
ax.fill_between(np.arange(len(max_time)+19),max_new_data.reshape(-1),min_new_data.reshape(-1),facecolor='red',alpha=0.1)
ax.set_title('number prediction')
ax.set_ylabel('number')
ax.set_xlabel('date')
# plt.plot(np.array(time), 'r', label='real', alpha=0.3)
plt.legend(loc='best')
plt.savefig('./lstm.eps')
plt.show()

# out=pd.DataFrame(new_data_x)
# out.to_csv('./newdata.csv')
loss=pd.DataFrame(loss_list)
loss.to_csv('./loss.csv')
