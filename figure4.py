import networkx as nx
import argparse
import pandas as pd
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pickle
from datetime import datetime as dt
import math
import datetime
import matplotlib.colors as colors
import matplotlib.cm as cmx
from matplotlib.lines import Line2D


# まずオブジェクト生成
parser = argparse.ArgumentParser()
#引数設定
parser.add_argument("-i","--input", help="optional")
parser.add_argument("-di","--dinput", help="optional")
parser.add_argument("-ti","--time", help="optional")
args = parser.parse_args()

#legendtimedic
dicdf = pd.read_csv(args.dinput, header=0)

#nodedic1= dict(zip(dicdf["ids"].astype(int),dicdf["legends"].astype(int)))

#dicdf['ids'] = pd.to_numeric(dicdf['ids'], errors='coerce')
#dicdf["legends"] = pd.to_numeric(dicdf['legends'], errors='coerce')

dicdf = dicdf.dropna()

nodedic1= dict(zip(dicdf["legendnames2"].astype(int),dicdf["legendstartdays"].values.tolist()))

df = pd.read_csv(args.input, header=0)

"""
with open("changedict20201122", mode='rb') as f:
        changedic = pickle.load(f)
"""


pigg = nx.from_pandas_edgelist(df,source='froms2',
                                   target='tos2',edge_attr=["nums","chans","neuts","suss","thOKs","thneus","thnos"],
                                   create_using=nx.DiGraph())


timedf = pd.read_csv(args.time, header=0)

usertimedic = dict(zip(timedf["code"].astype(int),timedf["date"].values.tolist()))
"""
f = plt.figure(1)
ax = f.add_subplot(1,1,1)
"""
fnodes = df["froms2"].values.tolist()
tnodes = df["tos2"].values.tolist()
nodes3 = fnodes+tnodes
pnodes4 = list(set(nodes3))



#fig = plt.figure()
#ax = fig.add_subplot(111)


"""
for k in pnodes4:
    if pigg.nodes[k]["node"] < 10:
        removenodes.append(k)
"""


finaldate = '2020-3-10'
finaldate2 = dt.strptime(finaldate, '%Y-%m-%d')



removenodes = []

for k in nodes3:
    usertime = usertimedic[k]
    usertime2 = dt.strptime(usertime, '%Y-%m-%d')
    userstarttime = usertime2 - datetime.timedelta(days=1095)
    d = finaldate2 - userstarttime   # finaldate2のほうが後なら、dは正の数,つまりマイナスなら弾く必要が有る。
    d = d.total_seconds()
    if d < 0:
        removenodes.append(k)


pigg.remove_nodes_from(removenodes)





pnodes4 = list(pigg.nodes)


for k in pnodes4:
    if k in nodedic1.keys():
        continue
    else:
        nodedic1[k] = np.nan



#print(nodedic1.keys())
#print(pnodes3[0:10])
#legends
nodedic3 = {}
for k in pnodes4:
    nodedic3[k]={}
    #nodedic1[int(k)]
    nodedic3[k]["legends"]=nodedic1[int(k)]
print("nodedic3OK")


nx.set_node_attributes(pigg, nodedic3)


nodedic3 = {}
numdic ={}

for k in pnodes4:
	nodedic3[k]={}
	nodedic3[k]["node"]= nx.degree(pigg)[k]
#print("nodedic3OK")
nx.set_node_attributes(pigg, nodedic3)

#10以下は削る。
"""
removenodes = []
for k in pnodes4:
    if pigg.nodes[k]["node"] < 10:
        removenodes.append(k)

pigg.remove_nodes_from(removenodes)

pnodes4 = set(pnodes4) - set(removenodes)
pnodes4 = list(pnodes4)
"""




pos = nx.spring_layout(pigg)
#pos = nx.kamada_kawai_layout(pigg)
#change_edges = [(from_node,to_node,edge_attributes) for from_node,to_node,edge_attributes in pigg.edges(data=True) if edge_attributes['thnos'] > 0]

legend_edges = []
non_legend_edges =[]

pedges = list(pigg.edges)
legend_edge_sizes = []
non_legend_edge_sizes =[]

#list(list(pigg.edges(data=True))[0][-1].keys())

for fnode,tnode in pedges:
    #fnode = fnodes[i]
    #tnode = tnodes[i]
    #fnode = str(fnode)
    #tnode = str(tnode)
    #strにするとエラーになる。
    """
    if fnode1 not in pnodes4:
        continue
    if tnode1 not in pnodes4:
        continue
    """
    l = (fnode,tnode)
    f1 = nodedic1[fnode]
    t1 = nodedic1[tnode]
    try:
        edgesize = np.log10([int(pigg[fnode][tnode]["nums"])])[0]
    except:
        edgesize = 0
    #print(f1)
    if f1 != f1:
        #print("regarded nan")
        non_legend_edges.append(l)
        non_legend_edge_sizes.append(edgesize)
        continue
    if t1 != t1:
        non_legend_edges.append(l)
        non_legend_edge_sizes.append(edgesize)
        continue
    flegendtime = dt.strptime(f1, '%Y-%m-%d')
    tlegendtime = dt.strptime(t1, '%Y-%m-%d')
    d = tlegendtime - flegendtime # toのほうが後の場合正
    d = d.total_seconds()
    #print(flegendtime)
    #print(tlegendtime)
    #print(d)
    if d > 0:
        legend_edges.append(l)
        legend_edge_sizes.append(edgesize)
    else:
        non_legend_edges.append(l)
        non_legend_edge_sizes.append(edgesize)

print("num legend_edges", len(legend_edges))
print("num nonlegend_edges", len(non_legend_edges))

print("edges OK")
abcon = nx.draw_networkx_edges(pigg, pos, edgelist=non_legend_edges, alpha=0.8,edge_color='yellow',arrowsize=0.1, width=non_legend_edge_sizes,label = "abstinent-contigious relationship")
noncon = nx.draw_networkx_edges(pigg, pos, edgelist=legend_edges,alpha=0.8,edge_color='blue',arrowsize=0.1,width=legend_edge_sizes, label = "non-contigious relationship")

pcolors = []
for pnode in pnodes4:
    legendtime = nodedic1[pnode]
    if legendtime != legendtime:
        pcolors.append("yellow")
    else:
        pcolors.append("blue")

print("colors OK")


node_sizes = []


for pnode4 in pnodes4:
    #thnos
    try:
        node_sizes.append(np.log10([nx.degree(pigg)[pnode4]])[0])
    except:
        node_sizes.append(0)


#nx.draw_networkx_nodes(pigg, pos,nodelist=c4,node_color=colors,node_size=node_sizes,alpha=0.8)

nx.draw_networkx_nodes(pigg, pos,nodelist=pnodes4,node_color=pcolors, node_size= node_sizes, alpha=0.8)



#clrs = [c for c in _c[:m]]
clrs = ["blue","yellow"]
def make_proxy(clr, mappable, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)
def make_proxy(clr, **kwargs):
    return Line2D([0, 1], [0, 1], color=clr, **kwargs)
#proxies = [make_proxy(clr, h2, lw=5) for clr in clrs]
proxies = [make_proxy(clr, lw=5) for clr in clrs]
labels = ["abstinent-contagious relationship","non-contagious relationship"]
plt.axis('off')#offにすると、外枠ができる。
#plt.legend()
#plt.legend(proxies, labels)
#plt.legend(proxies, labels,bbox_to_anchor=(0, -0.1), loc='upper left', borderaxespad=0, fontsize=18)
plt.legend(proxies, labels,bbox_to_anchor=(0, 0), loc='upper left', borderaxespad=0)
filename2 = "10year20220202f" + str(args.input) + ".png"
plt.savefig(filename2)


print("num of non legend", pcolors.count("yellow"))
print("num of legend", pcolors.count("blue"))
print("usual edges", len(non_legend_edges))
print("legend contagion edges", len(legend_edges))
