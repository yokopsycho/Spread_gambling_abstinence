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
