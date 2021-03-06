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

df=pd.read_csv('code_related.csv')
df=df[df['iifdefect']==True]
data=df[['sbaseline','sfindperiod','dtimefind','sbriefdesp','code_related']]#基线版本，问题发现阶段，问题发现时间，服务
data=data.dropna(subset=['sbaseline','sfindperiod','dtimefind','sbriefdesp'])# 删除空值

#转化为标准时间
for i in range(0, len(data)):
    row =data.iloc[i]
    a = row["dtimefind"]
    a1 = a.split('T')
    t1 = a1[0]
    t2 = ""
    for c in a1[1]:
        if c == ".":
            break
        t2 = t2 + c
    time = t1 + " " + t2
    row['dtimefind']=pd.to_datetime(time)
    data.iloc[i]=row

#根据时间排序
data = data.sort_values(by ='dtimefind')

#将时间转化为日期差
data['time']=0
firstday = data.iloc[0]['dtimefind'];

for i in range(1, len(data)):
    row = data.iloc[i]
    row['time'] = (row['dtimefind'] - firstday).days#计算日期差
    # row['time'] = rrule.rrule(rrule.WEEKLY, dtstart = firstday, until = row['dtimefind']).count()#计算周差
    data.iloc[i]=row


#累计故障数量
data['count']=1
for i in range(1, len(data)):
    row1 = data.iloc[i-1]
    row2 = data.iloc[i]
    if ((row2['sbriefdesp'].find('【SDV】') >= 0) and (row2['code_related'] == 1)):
        row2['count'] = row1['count']+1
    else:
        row2['count'] = row1['count']
    data.iloc[i]=row2
data = data[['time','count']]

#将相同天的重复删除
for i in range(1, len(data)):
    row1 = data.iloc[i-1]
    row2 = data.iloc[i]
    if row1['time'] == row2['time']:
       row1['time'] = None
    data.iloc[i - 1] = row1
data=data.dropna(subset=["time"])
#将月差加一
for i in range(0, len(data)):
    row = data.iloc[i]
    row['time']=row['time']+1
    data.iloc[i] = row

#重置索引
data=data.reset_index(drop=True)
print(data)


x = data['time']
x = np.array(x)
y = data['count']
y = np.array(y)


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

#Pham
#定义拟合函数
def Pham_func(x,a,b,r,e):
    f1 = (1 - np.exp(-b * (1 + e) * x)) * (1 - (r / (b * (1 + e)))) + r * x
    f2 = 1 + e * np.exp(-b * (1 + e) * x)
    return (a * f1) / f2
#拟合
popt,pcov = curve_fit(Pham_func,x,y,maxfev = 5000000)
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



# #绘图
# plot1 = plt.plot(x, y, 's',label='original values')
# plot2 = plt.plot(x, yvals, 'b',label='polyfit values')
# plt.xlabel('x')
# plt.ylabel('y')
# plt.legend(loc=4) #指定legend的位置右下角
# plt.title('polyfitting')
# plt.show()


#选择合适的func
func = Inflection_S_func

np.set_printoptions(suppress=True)

refun= np.exp(-(func(x + 1, a, b, bb) - func(x, a, b, bb)))

plt.plot(x,refun,label='polyfit values')
plt.show()
# x=10
# t=1
# reliability = np.exp(-(func(x + t, a, b, bb) - func(x, a, b, bb)))
# print('%.15f'%reliability)