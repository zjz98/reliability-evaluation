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

df=pd.read_excel('metrics3.xlsx')
print(df.corr())

# ca=math.exp(-0.000000855)
# print(ca)