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

df=pd.read_csv('C:/Users/dwx780786/Desktop/reliability/newbug.csv')
df=df[df['iifdefect']==True]
data=df[['sbaseline','sfindperiod','dtimefind','sbriefdesp']]#基线版本，问题发现阶段，问题发现时间，服务
data=data.dropna(subset=['sbaseline','sfindperiod','dtimefind','sbriefdesp'])# 删除空值

#选取某个服务
data=data[data["sbaseline"].str.contains('WiseCloudContentCenterService')]


#该服务中有的版本号中包含括号，去除括号
for i in range(0, len(data)):
    row = data.iloc[i]
    if (row['sbaseline'].find('(')>=0):
        loc=row['sbaseline'].find('(')
        row['sbaseline']=row['sbaseline'][0:loc]
    data.iloc[i]=row

data = data[['sbaseline','sbriefdesp']]

#根据版本排序
data['compare']=0;
for i in range(0, len(data)):
    row = data.iloc[i]
    line = row['sbaseline']

    #去掉版本号最后一位
    oflast= line.split('.')
    if(len(oflast)==4):
        oflast= oflast[0:-1]
        s=oflast[0]+'.'+oflast[1]+'.'+oflast[2]
    else:
        s=oflast[0]+'.'+oflast[1]+'.'+oflast[2]
    row['sbaseline']=s

    # 计算compare值进行排序
    s = line.split(' ')
    ss = s[len(s)-1]
    dot = ss.split('.')
    if(len(dot) == 4) :
        dot=dot[0:-1]
    if (len(dot) == 3):
        row['compare'] = 1000000000 * int(dot[0]) + 1000000 * int(dot[1]) + 1000 * int(dot[2])
    data.iloc[i] = row

data=data.sort_values(by='compare',ascending = True)

#重置索引
data=data.reset_index(drop=True)


#累计故障数量
data['count'] = 0
for i in range(1, len(data)):
    row1 = data.iloc[i-1]
    row2 = data.iloc[i]
    if(row2['sbriefdesp'].find('【ST】')>=0):
        row2['count'] = row1['count']+1
    else:
        row2['count'] = row1['count']
    data.iloc[i] = row2
data = data[['sbaseline','count']]

print(data)

#将相同的版本删除，
for i in range(1, len(data)):
    row1 = data.iloc[i-1]
    row2 = data.iloc[i]
    if row1['sbaseline'] == row2['sbaseline']:
       row1['sbaseline'] = None
       data.iloc[i - 1] = row1
    data.iloc[i - 1] = row1
data=data.dropna(subset=["sbaseline"])

#重置索引
data=data.reset_index(drop=True)
print(data)

#版本号变为0，1，2....
data['baselinenum'] = 0
firstrow = data.iloc[0]
firstrow['baselinenum'] = 1
data.iloc[0] = firstrow
for i in range(1, len(data)):
    row = data.iloc[i]
    row['baselinenum'] = i+1
    data.iloc[i]=row

data = data[['sbaseline','baselinenum','count']]
print(data)

x = data['baselinenum']
x = np.array(x)
y = data['count']
y = np.array(y)
#取完xy，取所需要版本的baselinenum
data = data[data['sbaseline'].isin(['WiseCloudContentCenterService 1.0.0','WiseCloudContentCenterService 1.0.1','WiseCloudContentCenterService 1.0.2',
                                    'WiseCloudContentCenterService 1.0.3','WiseCloudContentCenterService 1.0.4','WiseCloudContentCenterService 1.0.5',
                                    'WiseCloudContentCenterService 1.0.6','WiseCloudContentCenterService 1.0.7','WiseCloudContentCenterService 1.0.8',
                                    'WiseCloudContentCenterService 1.0.9','WiseCloudContentCenterService 1.0.11','WiseCloudContentCenterService 1.0.12',
                                    'WiseCloudContentCenterService 1.0.13','WiseCloudContentCenterService 1.1.1','WiseCloudContentCenterService 1.1.2',
                                    'WiseCloudContentCenterService 1.1.3','WiseCloudContentCenterService 1.1.4','WiseCloudContentCenterService 1.1.5',
                                    'WiseCloudContentCenterService 1.1.6','WiseCloudContentCenterService 1.2.1'])]
num = data['baselinenum']

#go模型
#定义拟合函数
def go_func(x,a,b):
    return a * (1 - np.exp(-b*x))
#拟合
popt,pcov = curve_fit(go_func,x,y,maxfev = 500000)
a = popt[0]
b = popt[1]
yvals = go_func(x,a,b)
#计算均方差
loss = metrics.mean_squared_error(y,yvals)
print(loss)


#Pham
#定义拟合函数
def Pham_func(x,a,b,r,e):
    f1 = (1 - np.exp(-b * (1 + e) * x)) * (1 - (r / (b * (1 + e)))) + r * x
    f2 = 1 + e * np.exp(-b * (1 + e) * x)
    return (a * f1) / f2
#拟合
popt,pcov = curve_fit(Pham_func,x,y,maxfev = 500000)
a = popt[0]
b = popt[1]
r = popt[2]
e = popt[3]
yvals = Pham_func(x,a,b,r,e)
#计算均方差
loss = metrics.mean_squared_error(y,yvals)
print(loss)


#Inflection S-shaped
#定义拟合函数
def Inflection_S_func(x,a,b,bb):
    return (a * (1 - np.exp(-b*x))) / (1 + bb * np.exp(-b*x))
#拟合
popt,pcov = curve_fit(Inflection_S_func,x,y,maxfev = 500000)
a = popt[0]
b = popt[1]
bb = popt[2]
yvals = Inflection_S_func(x,a,b,bb)
#计算均方差
loss = metrics.mean_squared_error(y,yvals)
print(loss)


#Delayed S-shaped
#定义拟合函数
def Delayed_S_func(x,a,b):
    return a * (1 - (1 + b * x) * np.exp(-b*x))
#拟合
popt,pcov = curve_fit(Delayed_S_func,x,y,maxfev = 500000)
a = popt[0]
b = popt[1]
yvals = Delayed_S_func(x,a,b)
#计算均方差
loss = metrics.mean_squared_error(y,yvals)
print(loss)

#绘图
plot1 = plt.plot(x, y, 's',label='original values')
plot2 = plt.plot(x, yvals, 'b',label='polyfit values')
plt.xlabel('x')
plt.ylabel('y')
plt.legend(loc=4) #指定legend的位置右下角
plt.title('polyfitting')
plt.show()


#选择合适的func
func = Delayed_S_func

np.set_printoptions(suppress=True)

for i in range(0, len(data)):
    row = data.iloc[i]
    t = row['baselinenum']
    reliability = np.exp(-(func(t-1 +0.1 , a, b) - func(t-1, a, b)))
    print(row['sbaseline'],'%.15f'%reliability)

