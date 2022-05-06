import argparse
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
from matplotlib.lines import Line2D

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
sns.set(style="white", palette="muted", color_codes=True,font_scale=0.8)
#font_scale=1 default
# データ生成
#np.random.seed(0)
#dt = np.random.normal(size=100)

#notice intimacy simulation was 1000times, network simulation was 200times

df = pd.read_csv(args.input, header=0)


cs = df.columns.values.tolist()

#cs.pop(0) # first filename
cs = cs[:-1] # last classname
print(cs)
#df2 = pd.read_csv(args.dinput, header=0)
#dt = df["vicrates"].values.tolist()
cl2 = df["cl2"].values.tolist()
cl2 = list(set(cl2))

sorted_cl2 = cl2.sort()

figname = os.path.splitext(os.path.basename(args.input))[0]

for c in cs:
    c1 = str(c)
    print(c1)
    if c1 == "nums" or c1 == "indegrees":
        pass
    else:
        continue
    legenddf = df[df["m_toslegends"]==1]
    nonlegenddf= df[df["m_toslegends"]==0]
    c1s = df[c1].values.tolist()
    log10c1s = np.log10(c1s)
    cname = "log10" + c1
    df[cname] = log10c1s
    line_plot = sns.catplot(x = "cl2", y = c1, data=df, hue = "m_toslegends",kind='bar',color = "blue",order = sorted_cl2, palette=sns.color_palette(['yellow','skyblue']),aspect=3)
    #line_plot = sns.catplot(x = "cl2", y = c1, data=df, kind='bar',color = "blue",order = sorted_cl2, palette=sns.color_palette(['yellow','skyblue']),aspect=3)
    #hueは必要

    clrs = ["skyblue","lightgreen"]
    def make_proxy(clr, mappable, **kwargs):
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)
    def make_proxy(clr, **kwargs):
        return Line2D([0, 1], [0, 1], color=clr, **kwargs)
    #proxies = [make_proxy(clr, h2, lw=5) for clr in clrs]
    proxies = [make_proxy(clr, lw=5) for clr in clrs]
    labels = ["abstinent gambler","non-abstinent gambler"]
    #plt.axis('off')#offにすると、外枠ができる。
    #plt.legend()
    #plt.legend(proxies, labels)
    #plt.legend(proxies, labels,bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)
    plt.legend(proxies, labels,bbox_to_anchor=(0, 1), loc='upper left', borderaxespad=0,fontsize='22')
    #plt.setp(ax.get_legend().get_texts(), fontsize='22')

    filename = figname + c1 + ".png"

    plt.savefig(filename)

    for c2 in cl2:
        df3 =legenddf[legenddf["cl2"]== c2]
        df3c1s = df3[c1]
        print(c1,c2,"legend_mean",df3c1s.mean(), "std",df3c1s.describe().loc['std'])
        nondf3 =nonlegenddf[nonlegenddf["cl2"]== c2]
        nondf3c1s = nondf3[c1]
        print(c1,c2,"nonlegend_mean",nondf3c1s.mean(), "std",nondf3c1s.describe().loc['std'])
