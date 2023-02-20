import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
data=pd.read_excel('D:\\美赛\\data\\frequency.xlsx',sheet_name=2)
diff=data['difficulty']
plt.style.use('seaborn')
plt.plot(diff,'m-',alpha=0.5,linewidth=1.2)
plt.xlabel('word')
plt.ylabel('difficulty')
plt.title('classification_law')
plt.axhline(250,color='c',ls='--')
plt.axhline(310,color='c',ls='--')
plt.text(175,175,'hard',fontsize=20)
plt.text(300,275,'normal',fontsize=20)
plt.text(200,350,'easy',fontsize=20)
plt.show()

