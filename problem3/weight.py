import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
weight=pd.read_csv('./weight.csv').iloc[:,1:]
weight.columns=['DL-score','F-score','V-score']
print(weight.sum(axis=0))
heatmap=sns.heatmap(weight)
plt.title('weight-matrix')
plt.show()