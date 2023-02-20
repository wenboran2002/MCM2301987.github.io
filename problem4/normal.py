import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import  cm
from scipy.stats import chi2
data=pd.read_excel('D:\美赛\data\Problem_C_Data_Wordle.xlsx',sheet_name=0)
plt.style.use('seaborn')

for i in range(110):
    probability=np.array(data.iloc[i*3,5:12])/100
    plt.plot(probability,color=cm.Wistia(i*0.008),linewidth=6,alpha=i*0.008)

# normal distribution
x=np.array([0.1*i for i in range(60)])
def norm_dis(x,sigma,mu):
    y=1/(np.sqrt(2*np.pi)*sigma)*np.exp(-(x-mu)**2/(2*sigma**2))
    return y
y=norm_dis(x,1.1,3)
plt.plot(x,y,'r--',linewidth=5,label='normal distribution')
plt.xlabel('tries')
plt.ylabel('probability')
plt.title('tries distribution')
plt.legend()
plt.show()

# chi2 distribution
number=np.array(data.iloc[:,3])
number=(number-np.min(number))/(np.max(number)-np.min(number))
x = np.linspace(0,50, len(number))
plt.plot(x,number/8)
plt.plot(x, chi2.pdf(x,df=6), 'k-',label='df=2')
plt.fill_between(x,number/8,chi2.pdf(x,df=6),facecolor='y',alpha=0.1)
plt.xlabel('time')
plt.ylabel('number_standardized')
plt.title('number distribution')
# plt.plot(x, chi2.pdf(x,df=8), 'k-',label='df=2')
plt.show()

