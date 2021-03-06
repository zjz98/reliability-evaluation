# -*- coding: utf-8 -*
import  pandas  as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import datetime
from sklearn import metrics
from dateutil import rrule
import math
import re
# import sys
# # reload(sys)
# sys.setdefaultencoding('utf-8')
# import warnings
# warnings.filterwarnings("ignore")

pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)

df=pd.read_excel('C:/Users/dwx780786/Desktop/reliability/WiseCloudMediaHostingService/metrics.xlsx')

df['LOC']=(df['LOC']-df['LOC'].min())/(df['LOC'].max()-df['LOC'].min())
df['CN']=(df['CN']-df['CN'].min())/(df['CN'].max()-df['CN'].min())
df['WMC']=(df['WMC']-df['WMC'].min())/(df['WMC'].max()-df['WMC'].min())
df['NOM']=(df['NOM']-df['NOM'].min())/(df['NOM'].max()-df['NOM'].min())
df['CBO']=(df['CBO']-df['CBO'].min())/(df['CBO'].max()-df['CBO'].min())
df['NAM']=(df['NAM']-df['NAM'].min())/(df['NAM'].max()-df['NAM'].min())
df['RFC']=(df['RFC']-df['RFC'].min())/(df['RFC'].max()-df['RFC'].min())
df['DIT']=(df['DIT']-df['DIT'].min())/(df['DIT'].max()-df['DIT'].min())
df['NOC']=(df['NOC']-df['NOC'].min())/(df['NOC'].max()-df['NOC'].min())
df['LCOM']=(df['LCOM']-df['LCOM'].min())/(df['LCOM'].max()-df['LCOM'].min())

plt.plot(df['count'],df['SDV_reliability'])
plt.plot(df['count'],df['ST_reliability'])
plt.plot(df['count'],df['LOC'])
plt.plot(df['count'],df['Reusability'])
plt.plot(df['count'],df['WMC'])
plt.plot(df['count'],df['NOM'])
plt.plot(df['count'],df['NAM'])
plt.plot(df['count'],df['RFC'])
plt.plot(df['count'],df['NOC'])
plt.plot(df['count'],df['DIT'])
plt.plot(df['count'],df['CBO'])
plt.plot(df['count'],df['LCOM'])


plt.legend(['Y_SDV','Y_ST',
            'LOC(sum)','Reusability(avg)','WMC(sum)','NOM(sum)','NAM(sum)','RFC(sum)',
            'NOC(sum)','DIT(sum)','CBO(sum)','LCOM(sum)'],loc='upper left')
plt.show()
# ca=math.exp(-0.000000855)
# print(ca)