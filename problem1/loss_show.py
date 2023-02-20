import matplotlib.pyplot as plt
import pandas as pd
plt.style.use('seaborn')
loss=pd.read_csv('loss.csv')
print(loss.head())
plt.plot(loss.iloc[:,1].values[:150:5])
plt.title('Loss')
plt.xlabel('epocs')
plt.ylabel('loss')
plt.show()