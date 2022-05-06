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
parser.add_argument("-di","--dinput", help="optional")
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

cs.pop(0) # first filename
cs = cs[:-1] # last classname
print(cs)
df2 = pd.read_csv(args.dinput, header=0)
#dt = df["vicrates"].values.tolist()

figname = os.path.splitext(os.path.basename(args.input))[0]
for c in cs:
    c1 = str(c)
    print(c1)
    #df2b = df2[df2['names'] == "vicrates"]
    df2b = df2[df2['names'] == c1]
    #sns.catplot(x = "cl2", y = "vicrates", data=df, kind='swarm',dodge=True)
    #line_plot = sns.catplot(x = "cl2", y = "vicrates", data=df, kind='violin',color = "skyblue")
    line_plot = sns.catplot(x = "cl2", y = c1, data=df, kind='violin',color = "skyblue",order = ["net1","net2","net3"])
    #line_plot = sns.pointplot(x="cl2", y="observed_score", data=df2b,join = False,color = "red",order = ["intimacy0","intimacy1","intimacy2","intimacy3","intimacy4","intimacy5"])
    #plt.legend()
    line_plot = sns.pointplot(x="cl2", y="observed_score", data=df2b,join = False,color = "red",order = ["net1","net2","net3"],hue = "cl2")
    # title
    new_title = 'Observed probability'
    line_plot.legend_.set_title(new_title)
    # replace labels
    new_labels = ['Distance 1', 'Distance 2',"Distance 3"]
    for t, l in zip(line_plot.legend_.texts, new_labels):
        t.set_text(l)
    #plt.show()
    #line_plot = snsplt.legend(labels=["Legend_Day1","Legend_Day2"])
    filename = figname + c1 + "20220203.png"
    #line_plot.savefig(filename)
    plt.savefig(filename)




#figure = line_plot.get_figure()

#figure.savefig(filename)

"""
fig, ax1 = plt.subplots()

binwidth=0.01
bins2 = np.arange(0.1, 0.5 + binwidth, binwidth)

# ヒストグラムプロットとビン情報の取得
n, bins, patches = ax1.hist(dt, alpha=0.7, color='blue', label='Frequency',orientation="horizontal", range =[0.1,0.5],bins=bins2)


#invert the order of x-axis values
ax1.set_xlim(ax1.get_xlim()[::-1])

#move ticks to the right
ax1.yaxis.tick_right()

# 第2軸用値の算出
#y2 = np.add.accumulate(n) / n.sum()
#x2 = np.convolve(bins, np.ones(2) / 2, mode="same")[1:]

y2 = 0.4
x2 = 0
# 第2軸のプロット
#ax2 = ax1.twinx()
lines = ax1.plot(x2, y2, ls='--', color='r', marker='o',markersize=20,
         label='Cumulative ratio')
#ax2.grid(visible=False)
#ax2.grid(visible=True)

plt.show()

"""
