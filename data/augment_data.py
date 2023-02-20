import numpy as np
import pandas as pd
fr=pd.read_excel('./frequency.xlsx',sheet_name=1)
org=pd.read_excel('./Problem_C_Data_Wordle.xlsx')
n=len(org.index)
print(fr.head())
for i in range(n):
    for j in range(len(fr.index)):
        if org[i]['Word']==fr[j]['word']:
            org[i]['fr']==fr[j]['fre']
print(org.head())


