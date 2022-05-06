import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import sklearn
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier as KNN
from sklearn.metrics import accuracy_score as acc
from sklearn.metrics import roc_auc_score
#from sklearn.impute import SimpleImputer
from sklearn.cross_decomposition import PLSRegression


import scipy
#import scipy.stats
from scipy.stats import t  # We only need the t class from scipy.stats
from scipy import stats


import numpy as np
from lime import explanation
from lime import lime_base
import math

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.neural_network import MLPClassifier

#from lime_timeseries import LimeTimeSeriesExplainer

import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
# まずオブジェクト生成
parser = argparse.ArgumentParser()
#引数設定
parser.add_argument("-i","--input", help="optional")
#parser.add_argument("-vi","--vinput", help="optional")
#parser.add_argument("-di","--dinput", help="optional")
#parser.add_argument("-gi","--ginput", help="optional")
#parser.add_argument("-ti","--time", help="optional")
args = parser.parse_args()

# % matplotlib inline
# seabornのスタイルでプロット
#sns.set(style="white", palette="muted", color_codes=True,font_scale=0.8)
#font_scale=1 default
# データ生成
#np.random.seed(0)
#dt = np.random.normal(size=100)

#notice intimacy simulation was 1000times, network simulation was 200times

df = pd.read_csv(args.input, header=0)
cs = df.columns.values.tolist()
idx = df[cs[0]].values.tolist()
#labels = df[cs[-1]].values.tolist()
cs.pop(0)#index
#cs.pop(-1)#教師ラベル
#ids = df.index
X = df[cs].values
#y = df["144"].values#教師ラベル
meandic={}
sddic={}

for i in range(len(X)):
    id = idx[i]
    vs = X[i,:]
    #print(id)
    print(vs.shape)
    mc = np.nansum(np.array(vs))
    #print(mc)
    m = np.nansum(vs)/int(len(vs))
    sd = float(np.nanstd(vs))/int(np.sqrt(len(vs)))#standard_error
    meandic[id] = m
    sddic[id] = sd

fns = np.arange(144)
ms=[]
sds=[]
for f in fns:
    if f not in meandic.keys():
        ms.append(0)
        sds.append(0)
    else:
        ms.append(meandic[f])
        sds.append(sddic[f])

df2 = pd.DataFrame()

df2["fns"]=fns
df2["ms"]=ms
df2["sds"]=sds

filename = "figure3result_"+ args.input

df2.to_csv(filename,index= False)



fns1 = fns[:36]
fns2 = fns[36:72]
fns3 = fns[72:108]
fns4 = fns[108:144]


ms1 = ms[:36]
ms2 = ms[36:72]
ms3 = ms[72:108]
ms4 = ms[108:144]


sds1 = sds[:36]
sds2 = sds[36:72]
sds3 = sds[72:108]
sds4 = sds[108:144]

"""
fig, ax = plt.subplots()
ax.bar(fns,ms, color=color)
"""
error_bar_set = dict(lw = 1, capthick = 1, capsize = 4)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.bar(fns, ms, yerr = sd,error_kw=error_bar_set,color=color)
#正値のとき(i>0)は、青色('b')、そうでないなら赤色('r')を指定。
color1 = [('b' if i > 0 else 'r') for i in ms1]
ax.bar(fns1, ms1, yerr=sds1,error_kw=error_bar_set,color=color1)
ax.set_ylim([-0.05,0.05])
filename = args.input + "totalcomment.png"
#fig.show()
fig.savefig(filename)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.bar(fns, ms, yerr = sd,error_kw=error_bar_set,color=color)
#正値のとき(i>0)は、青色('b')、そうでないなら赤色('r')を指定。
color2 = [('b' if i > 0 else 'r') for i in ms2]
ax.bar(fns2, ms2, yerr=sds2,error_kw=error_bar_set,color=color2)
#ax.set_ylim([-0.15,0.15])
ax.set_ylim([-0.05,0.05])
filename = args.input + "rejectivecomment.png"
#fig.show()
fig.savefig(filename)


fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.bar(fns, ms, yerr = sd,error_kw=error_bar_set,color=color)
#正値のとき(i>0)は、青色('b')、そうでないなら赤色('r')を指定。
color3 = [('b' if i > 0 else 'r') for i in ms3]
ax.bar(fns3, ms3, yerr=sds3,error_kw=error_bar_set,color=color3)
#ax.set_ylim([-0.15,0.15])
ax.set_ylim([-0.05,0.05])
filename = args.input + "numlegends.png"
#fig.show()
fig.savefig(filename)




fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
#ax.bar(fns, ms, yerr = sd,error_kw=error_bar_set,color=color)
#正値のとき(i>0)は、青色('b')、そうでないなら赤色('r')を指定。
color4 = [('b' if i > 0 else 'r') for i in ms4]
ax.bar(fns4, ms4, yerr=sds4,error_kw=error_bar_set,color=color4)
#ax.set_ylim([-0.15,0.15])
ax.set_ylim([-0.05,0.05])
filename = args.input + "numfriends.png"
#fig.show()
fig.savefig(filename)
