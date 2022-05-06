import math
import pandas as pd
import scipy.stats as st
import os
import argparse
import datetime
import pandas as pd
import re
from datetime import datetime as dt
import numpy as np
from scipy import stats

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,help="input file")
#parser.add_argument('-di', '--di', type=str, required=True,help="input file")
args = parser.parse_args()

df = pd.read_csv(args.input)

cs = df.columns.values.tolist()


#cs = cs.pop(0)# tosidsの除去
print(cs)

ldf = df[df["m_toslegends"]==1]
nonldf =df[df["m_toslegends"]==0]

m_legends=[]
std_legends=[]
num_ldfs=[]
m_nonlegends=[]
std_nonlegends=[]
num_nonldfs=[]
t_values=[]
p_values=[]
for c in cs:
    a = ldf[c].dropna()
    b = nonldf[c].dropna()
    r= stats.ttest_ind(a,b, equal_var = False)
    m_legend = np.nanmean(a)
    m_nonlegend= np.nanmean(b)
    std_legend = np.nanstd(a)
    std_nonlegend= np.nanstd(b)
    num_ldf =a.count()
    num_nonldf = b.count()
    m_legends.append(m_legend)
    std_legends.append(std_legend)
    num_ldfs.append(num_ldf)
    m_nonlegends.append(m_nonlegend)
    std_nonlegends.append(std_nonlegend)
    num_nonldfs.append(num_nonldf)
    t_values.append(r[0])
    p_values.append(r[1])
    print(c,"m_legend",m_legend,"std_legend",std_legend,"num_ldf",num_ldf,"m_nonlegend",m_nonlegend,"std_nonlegend",std_nonlegend,"num_nonldf",num_nonldf,"ttest", r[0],"p-value",r[1])

df2=pd.DataFrame()
df2["cs"]=cs
df2["m_legends"]=m_legends
df2["std_legends"]=std_legends
df2["num_ldfs"]=num_ldfs
df2["m_nonlegends"]=m_nonlegends
df2["std_nonlegends"]=std_nonlegends
df2["num_nonldfs"]=num_nonldfs
df2["t_values"]=t_values
df2["p_values"]=p_values

filename = "table1result.csv"
df2.to_csv(filename,index=False)
