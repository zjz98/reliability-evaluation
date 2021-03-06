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

#该服务中间改过名，所以需要将名称进行统一
data=pd.read_csv('C:/Users/dwx780786/Desktop/reliability/newbug.csv',low_memory=False)
data=data.dropna(subset=['sbaseline'])# 删除空值



data1=data[data["sbaseline"].str.contains('WiseCloudHOTAService')]

#该服务中有的版本号中包含括号，去除括号
for i in range(0, len(data1)):
    row = data1.iloc[i]
    if (row['sbaseline'].find('(')>=0):
        loc=row['sbaseline'].find('(')
        row['sbaseline']=row['sbaseline'][0:loc]
    data1.iloc[i]=row


data2=pd.read_csv('C:/Users/dwx780786/Desktop/reliability/devDeskChange.csv')
data2=data2.dropna(subset=['ticket_number','project'])# 删除空值
data2=data2[data2["project"].str.contains('WiseCloudHOTAService')]


data1['code_related'] = 0
for i in range(0, len(data1)):
    row1 = data1.iloc[i]
    for j in range(0,len(data2)):
        row2 = data2.iloc[j]
        if(row2['ticket_number']==row1['_id']):
            row1['code_related']=1
    data1.iloc[i] = row1

data1.to_csv('code_related.csv',encoding='utf_8_sig')
