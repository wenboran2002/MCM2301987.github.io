import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime   #数据索引改为时间
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_pacf,plot_acf
from data import data_preprocessing
plt.style.use('seaborn')
data_excel=pd.read_excel('D:\美赛\data\Problem_C_Data_Wordle.xlsx')
data=data_preprocessing.Data_used(data_excel)
time=data_excel.copy()
time=time.set_index('Date')['Number of  reported results']
# 单位根检验-ADF检验
print(sm.tsa.stattools.adfuller(time))
print(acorr_ljungbox(time, lags=1))
plt.ylim(-0.5,1.5)
sm.graphics.tsa.plot_acf(time,lags=20)
plt.show()


