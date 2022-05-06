import os
import argparse
import datetime
import pandas as pd
import MeCab
import re
import pickle
import pandas as pd
import networkx as nx
import argparse
import numpy as np
import math
from scipy.stats import entropy
import pickle
from node2vec import Node2Vec
from gensim.models import word2vec
import multiprocessing
import random
import json
from datetime import datetime as dt


# まずオブジェクト生成
parser = argparse.ArgumentParser()
#引数設定
parser.add_argument("-i","--input", help="optional")
#parser.add_argument("-oi","--oinput", help="optional")
#parser.add_argument("-vi","--vinput", help="optional")
parser.add_argument("-di","--dinput", help="optional")
parser.add_argument("-ui","--uinput", help="optional")
args = parser.parse_args()



df = pd.read_csv(args.input, header=0)
#df = pd.read_csv(args.input, header=0)
"""
with open(args.input, 'rb') as rf:
  df = pickle.load(rf)
"""
#df = pd.read_pickle(args.input)



pigg = nx.from_pandas_edgelist(df,source='froms2',
                                   target="tos2",
                                   create_using=nx.DiGraph())

fnodes = df["froms2"].values.tolist()
tnodes = df["tos2"].values.tolist()

p_chans = df["p_chans"].values.tolist()
p_neuts =df["p_neuts"].values.tolist()
p_suss =df["p_suss"].values.tolist()
p_thOKs =df["p_thOKs"].values.tolist()
p_thneus =df["p_thneus"].values.tolist()
p_thnos =df["p_thnos"].values.tolist()
p_numlegends = df["p_chans"].values.tolist()
p_accepts=df["p_accepts"].values.tolist()
f_intims = df["f_intims"].values.tolist()



nums = df["nums"].values.tolist()
chans = df["chans"].values.tolist()
neuts = df["neuts"].values.tolist()
suss = df["suss"].values.tolist()
thOKs = df["thOKs"].values.tolist()
thneus = df["thneus"].values.tolist()
thnos = df["thnos"].values.tolist()
numlegends= df["numlegends"].values.tolist()

talknumdic={}

talkdic={}
for i in range(len(fnodes)):
    fnode = fnodes[i]
    tnode = tnodes[i]
    k2 = str(fnode)+"_"+str(tnode)
    p_chan = p_chans[i]
    p_neut = p_neuts[i]
    p_sus = p_suss[i]
    p_thOK = p_thOKs[i]
    p_thneu = p_thneus[i]
    p_thno = p_thnos[i]
    p_accept = p_accepts[i]
    f_intim = f_intims[i]
    p_numlegend = p_numlegends[i]
    l2 = [p_chan,p_neut,p_sus,p_thOK,p_thneu,p_thno,p_numlegend,p_accept,f_intim]
    talkdic[k2] = l2
    num = nums[i]
    chan = chans[i]
    neut = neuts[i]
    sus = suss[i]
    thOK = thOKs[i]
    thneu = thneus[i]
    thno = thnos[i]
    numlegend = numlegends[i]
    l3 = [num,chan,neut,sus,thOK,thneu,thno,numlegend]
    talknumdic[k2] = l3




fnodes2 = fnodes + tnodes
fnodes3 = list(set(fnodes2))



dicdf = pd.read_csv(args.dinput, header=0)

cols= dicdf.columns.tolist()
col1=cols[1]
col2=cols[2]
legendtimedic = dict(zip(list(dicdf[col1]), list(dicdf[col2])))


for k in fnodes3:
    if k in legendtimedic.keys():
        continue
    else:
        legendtimedic[k] = np.nan

usertimedf=pd.read_csv(args.uinput, header=0)

cols= usertimedf.columns.tolist()
col0=cols[0]
col1=cols[1]
usertimedic = dict(zip(list(usertimedf[col0]), list(usertimedf[col1])))


for k in fnodes3:
    if k in usertimedic.keys():
        continue
    else:
        usertimedic[k] = np.nan

centraldic ={}
#indegreecentraly辞書の作成
#これはネットワークを維持した乱数では変更されない。
chcentraldic ={}
#こちらは変更しうる。

for k1, v1 in nx.in_degree_centrality(pigg).items():
    centraldic[k1] = v1
    edges1e = list(nx.bfs_edges(pigg, source = k1, depth_limit = 1))
    nodes1e = [v for u, v in edges1e]
    chl=[]
    neul=[]
    sussl=[]
    thOKl=[]
    thneul=[]
    thnol=[]
    for nd1e in nodes1e:
        k4 = str(k1)+"_"+str(nd1e)
        l4 = talkdic[k4]
        chl.append(l4[0])
        neul.append(l4[1])
        sussl.append(l4[2])
        thOKl.append(l4[3])
        thneul.append(l4[4])
        thnol.append(l4[5])
    l5 = [np.nansum(chl)/v1,np.nansum(neul)/v1,np.nansum(sussl)/v1,np.nansum(thOKl)/v1,np.nansum(thneul)/v1,np.nansum(thnol)/v1]
    chcentraldic[k1]=l5

legendvarr = list(legendtimedic.values())
legendkeys = list(legendtimedic.keys())


talkvarr = list(talkdic.values())
talkkeys = list(talkdic.keys())

talknumvarr = list(talknumdic.values())
talknumkeys = list(talknumdic.keys())

usertimevarr = list(usertimedic.values())
usertimekeys = list(usertimedic.keys())


chcentralvarr = list(chcentraldic.values())
chcentralkeys = list(chcentraldic.keys())


for i2 in range(0,1000):
    print(i2)
    random.seed(i2)
    legendvarr2 = random.sample(legendvarr,len(legendvarr))
    #vl2 = list(vicdic1.keys())
    legendtimedicb = {}
    for j in range(len(legendkeys)):
        legendtimedicb[legendkeys[j]] = legendvarr2[j]
    talkvarr2 = random.sample(talkvarr,len(talkvarr))
    #vl2 = list(vicdic1.keys())
    talkdicb = {}
    for j in range(len(talkkeys)):
        talkdicb[talkkeys[j]] = talkvarr2[j]

    talknumvarr2 = random.sample(talknumvarr,len(talknumvarr))
    #vl2 = list(vicdic1.keys())
    talknumdicb = {}
    for j in range(len(talknumkeys)):
        talknumdicb[talknumkeys[j]] = talknumvarr2[j]

    usertimevarr2 = random.sample(usertimevarr,len(usertimevarr))
    #vl2 = list(vicdic1.keys())
    usertimedicb = {}
    for j in range(len(usertimekeys)):
        usertimedicb[usertimekeys[j]] = usertimevarr2[j]
    #centraldicは変更されない。ネットワークを固定しているので
    chcentralvarr2 = random.sample(chcentralvarr,len(chcentralvarr))
    chcentraldicb = {}
    for j in range(len(chcentralkeys)):
        chcentraldicb[chcentralkeys[j]] = chcentralvarr2[j]


    #nd1
    n1idx=[]
    #change
    n1legend_b=[]

    n1chp_b=[]
    #neuts

    n1neup_b=[]

    n1suss_b=[]

    n1thOKs_b=[]

    n1thneus_b=[]


    n1thnos_b=[]

    n1lp_b=[]

    #nd2
    n2idx=[]
    #change
    n2legend_b=[]

    n2chp_b=[]
    #neuts

    n2neup_b=[]

    n2suss_b=[]

    n2thOKs_b=[]

    n2thneus_b=[]


    n2thnos_b=[]

    n2lp_b=[]

    #nd3
    n3idx=[]
    #change
    n3legend_b=[]

    n3chp_b=[]
    #neuts

    n3neup_b=[]

    n3suss_b=[]

    n3thOKs_b=[]

    n3thneus_b=[]


    n3thnos_b=[]

    n3lp_b=[]

    #add_argument

    n1num_b=[]
    n2num_b=[]
    n3num_b=[]

    n1central_b=[]
    n2central_b=[]
    n3central_b=[]

    n1chcentral_b=[]
    n2chcentral_b=[]
    n3chcentral_b=[]
    n1neucentral_b=[]
    n2neucentral_b=[]
    n3neucentral_b=[]
    n1suscentral_b=[]
    n2suscentral_b=[]
    n3suscentral_b=[]
    n1thOKcentral_b=[]
    n2thOKcentral_b=[]
    n3thOKcentral_b=[]
    n1thneucentral_b=[]
    n2thneucentral_b=[]
    n3thneucentral_b=[]
    n1thnocentral_b=[]
    n2thnocentral_b=[]
    n3thnocentral_b=[]





    for k, v in nx.degree_centrality(pigg).items():
        if k != k:
            continue
        #print(k)
        #forcheck
        """
        if k == 2298:
            break
        """
        k = int(k)
        n1idx.append(k)
        n2idx.append(k)
        n3idx.append(k)
        v1 = legendtimedicb[k]
        u1 = usertimedicb[k]
        #nd1
        edges1 = nx.bfs_edges(pigg, source = k, depth_limit = 1)
        nodes1 = [v for u, v in edges1] # edge1のnodeが分かる。
        #edges1 = pigg.in_edges(k)#入ってくるもののみを抽出
        #nodes1 = [u for u, v in edges1] # 入ってくるものなので、uを抽出。
        num_l = []
        chp_l = []
        neup_l = []
        suss_l = []
        thOKs_l = []
        thneus_l = []
        thnos_l = []
        #accepts_l=[]
        lp_l = []
        legend_l=[]
        user_l=[]
        username_l =[]
        chp_lb = []
        neup_lb = []
        suss_lb = []
        thOKs_lb = []
        thneus_lb = []
        thnos_lb = []
        num_lb = []
        #accepts_lb=[]
        lp_lb = []
        legend_lb=[]
        legendname_lb =[]
        talkname_lb = []
        #
        central_l=[]
        central_lb=[]
        chcentl=[]
        neucentl=[]
        susscentl=[]
        thOKcentl=[]
        thneucentl=[]
        thnocentl=[]
        chcentlb=[]
        neucentlb=[]
        susscentlb=[]
        thOKcentlb=[]
        thneucentlb=[]
        thnocentlb=[]
        for nd1 in nodes1:
            legend_l.append(legendtimedicb[nd1])
            user_l.append(usertimedicb[nd1])
            username_l.append(nd1)
            k3 = str(k)+ "_" + str(nd1)
            l3 = talknumdicb[k3]
            #l3a = talkdicb[k3]
            num_l.append(l3[0])
            chp_l.append(l3[1])
            neup_l.append(l3[2])
            suss_l.append(l3[3])
            thOKs_l.append(l3[4])
            thneus_l.append(l3[5])
            thnos_l.append(l3[6])
            lp_l.append(l3[6])
            central_l.append(centraldic[nd1])
            l6 = chcentraldicb[nd1]
            chcentl.append(l6[0])
            neucentl.append(l6[1])
            susscentl.append(l6[2])
            thOKcentl.append(l6[3])
            thneucentl.append(l6[4])
            thnocentl.append(l6[5])

            #accepts_l.append(l3a[6])
        #usertime&legendtimecheck
        #legend_lb
        #usertime
        usertime = dt.strptime(u1, '%Y-%m-%d')
        #legendtime
        if v1 != v1:
            legend_lb=[]
        else:
            legendtime = dt.strptime(v1, '%Y-%m-%d')
        for i in range(len(legend_l)):
            flegend = legend_l[i]
            username = username_l[i]
            if flegend != flegend:
                continue
            flegendtime = dt.strptime(flegend, '%Y-%m-%d')
            d = flegendtime - usertime #friendのlegendが自分の発話よりも後（新しい）なら、正になる
            d = d.total_seconds()
            if d > 0:
                chp_lb.append(chp_l[i])
                neup_lb.append(neup_l[i])
                suss_lb.append(suss_l[i])
                thOKs_lb.append(thOKs_l[i])
                thneus_lb.append(thneus_l[i])
                thnos_lb.append(thnos_l[i])
                talkname_lb.append(username)
                num_lb.append(num_l[i])
                central_lb.append(central_l[i])
                chcentlb.append(chcentl[i])
                neucentlb.append(neucentl[i])
                susscentlb.append(susscentl[i])
                thOKcentlb.append(thOKcentl[i])
                thneucentlb.append(thneucentl[i])
                thnocentlb.append(thnocentl[i])
            try:
                db = flegendtime - legendtime #friendのほうが自分よりも後（新しい）なら、正になる
            except:
                continue
            db = db.total_seconds()
            if db > 0:
                legend_lb.append(flegend)
                legendname_lb.append(username)
                lp_lb.append(lp_l[i])


        #network_effects
        try:
            n1legend_b.append(len(legend_lb)/len(legend_l))
        except:
            n1legend_b.append(0)
        try:
            n1chp=np.nansum(chp_lb)/np.nansum(chp_l)
            n1chp_b.append(n1chp)
        except:
            n1chp_b.append(0)
        try:
            n1neup = np.nansum(neup_lb)/np.nansum(neup_l)
            n1neup_b.append(n1neup)
        except:
            n1neup_b.append(0)
        try:
            n1suss = np.nansum(suss_lb)/np.nansum(suss_l)
            n1suss_b.append(n1suss)
        except:
            n1suss_b.append(0)
        try:
            n1thOKs = np.nansum(thOKs_lb)/np.nansum(thOKs_l)
            n1thOKs_b.append(n1thOKs)
        except:
            n1thOKs_b.append(0)
        try:
            n1thneus = np.nansum(thneus_lb)/np.nansum(thneus_l)
            n1thneus_b.append(n1thneus)
        except:
            n1thneus_b.append(0)
        try:
            n1thnos = np.nansum(thnos_lb)/np.nansum(thnos_l)
            n1thnos_b.append(n1thnos)
        except:
            n1thnos_b.append(0)
        try:
            n1lp = np.nansum(lp_lb)/np.nansum(lp_l)
            n1lp_b.append(n1lp)
        except:
            n1lp_b.append(0)
        try:
            n1num_b.append(np.nansum(num_lb)/np.nansum(num_l))
        except:
            n1num_b.append(0)
        try:
            n1central_b.append(np.nansum(central_lb)/np.nansum(central_l))
        except:
            n1central_b.append(0)
        #add_argument
        try:
            n1chcentral_b.append(np.nansum(chcentlb)/np.nansum(chcentl))
        except:
            n1chcentral_b.append(0)
        try:
            n1neucentral_b.append(np.nansum(neucentlb)/np.nansum(neucentl))
        except:
            n1neucentral_b.append(0)
        try:
            n1suscentral_b.append(np.nansum(susscentlb)/np.nansum(susscentl))
        except:
            n1suscentral_b.append(0)
        try:
            n1thOKcentral_b.append(np.nansum(thOKcentlb)/np.nansum(thOKcentl))
        except:
            n1thOKcentral_b.append(0)
        try:
            n1thneucentral_b.append(np.nansum(thneucentlb)/np.nansum(thneucentl))
        except:
            n1thneucentral_b.append(0)
        try:
            n1thnocentral_b.append(np.nansum(thnocentlb)/np.nansum(thnocentl))
        except:
            n1thnocentral_b.append(0)

        #nd2
        edges2 = list(nx.bfs_edges(pigg, source = k, depth_limit = 2))
        nodes2 = [v for u, v in edges2]
        nodes2a = set(nodes2)-set(nodes1)
        nodes2b = list(nodes2a)
        edges2a = set(edges2) - set(edges1)
        edges2b = list(edges2a)
        #print(edges2)
        #print(edges2a)
        #print(edges2b)
        #legend_l2=[]
        #legend_l2b=[]
        #legendcheck
        e2num_l = []
        e2chp_l = []
        e2neup_l = []
        e2suss_l = []
        e2thOKs_l = []
        e2thneus_l = []
        e2thnos_l = []
        e2lp_l = []
        e2legend_l=[]
        e2legendfrom_l=[]
        e2usernamefrom_l=[]
        e2user_l=[]
        #b
        e2chp_lb = []
        e2neup_lb = []
        e2suss_lb = []
        e2thOKs_lb = []
        e2thneus_lb = []
        e2thnos_lb = []
        e2lp_lb = []
        e2legend_lb =[]
        #e2user_l=[]
        e2legend_lb=[]
        talkname2_lb=[]
        legendname2_lb=[]
        username2_l=[]
        e2num_lb = []
        e2central_l=[]
        e2central_lb =[]
        #add_argument
        e2chcentl=[]
        e2neucentl=[]
        e2susscentl=[]
        e2thOKcentl=[]
        e2thneucentl=[]
        e2thnocentl=[]
        e2chcentlb=[]
        e2neucentlb=[]
        e2susscentlb=[]
        e2thOKcentlb=[]
        e2thneucentlb=[]
        e2thnocentlb=[]
        for e2 in edges2b:
            e2legendfrom_l.append(legendtimedicb[e2[0]])
            e2usernamefrom_l.append(e2[0])
            username2_l.append(e2[1])
            e2legend_l.append(legendtimedicb[e2[1]])
            e2user_l.append(usertimedicb[e2[1]])
            ke2 = str(e2[0])+"_" + str(e2[1])
            le2 = talknumdicb[ke2]
            e2num_l.append(le2[0])
            e2chp_l.append(le2[1])
            e2neup_l.append(le2[2])
            e2suss_l.append(le2[3])
            e2thOKs_l.append(le2[4])
            e2thneus_l.append(le2[5])
            e2thnos_l.append(le2[6])
            e2lp_l.append(le2[6])
            e2central_l.append(centraldic[e2[1]])
            #print(e2)
            l6 = chcentraldicb[e2[1]]
            e2chcentl.append(l6[0])
            e2neucentl.append(l6[1])
            e2susscentl.append(l6[2])
            e2thOKcentl.append(l6[3])
            e2thneucentl.append(l6[4])
            e2thnocentl.append(l6[5])

        for i in range(len(e2legend_l)):
            flegend2 = e2legend_l[i]
            username2 = username2_l[i]
            fromusername2 = e2usernamefrom_l[i]
            if flegend2 != flegend2:
                #print("nonflegend2")
                continue
            flegendtime2 = dt.strptime(flegend2, '%Y-%m-%d')
            #print(flegendtime2)
            #print(fromusername2)
            #fromusername2 = int(fromusername2)
            if fromusername2 not in talkname_lb:
                #print("nonusername2")
                continue
            usertime2 = usertimedicb[fromusername2]
            usertime2b = dt.strptime(usertime2, '%Y-%m-%d')
            #print(usertime2b)
            d2 = flegendtime2 - usertime2b #friendのfriendのlegendが自分の発話よりも後（新しい）なら、正になる
            d2 = d2.total_seconds()
            #print("d2", d2)
            if d2 > 0:
                e2chp_lb.append(e2chp_l[i])
                e2neup_lb.append(e2neup_l[i])
                e2suss_lb.append(e2suss_l[i])
                e2thOKs_lb.append(e2thOKs_l[i])
                e2thneus_lb.append(e2thneus_l[i])
                e2thnos_lb.append(e2thnos_l[i])
                talkname2_lb.append(username2)
                e2num_lb.append(e2num_l[i])
                e2central_lb.append(e2central_l[i])
                #
                e2chcentlb.append(e2chcentl[i])
                e2neucentlb.append(e2neucentl[i])
                e2susscentlb.append(e2susscentl[i])
                e2thOKcentlb.append(e2thOKcentl[i])
                e2thneucentlb.append(e2thneucentl[i])
                e2thnocentlb.append(e2thnocentl[i])
                #print("talk2OK")
            if fromusername2 not in legendname_lb:
                continue
            legendtime2 = legendtimedicb[fromusername2]
            legendtime2b = dt.strptime(legendtime2, '%Y-%m-%d')
            try:
                d2b = flegendtime2 - legendtime2b #friendのほうが自分よりも後（新しい）なら、正になる
            except:
                continue
            d2b = d2b.total_seconds()
            #print("d2b", d2b)
            if d2b > 0:
                e2legend_lb.append(flegend2)
                legendname2_lb.append(username2)
                #print("legend2OK")
                e2lp_lb.append(e2lp_l[i])
        try:
            n2legend_b.append(len(e2legend_lb)/len(e2legend_l))
        except:
            n2legend_b.append(0)
        try:
            n2chp_b.append(np.nansum(e2chp_lb)/np.nansum(e2chp_l))
        except:
            n2chp_b.append(0)
        try:
            n2neup_b.append(np.nansum(e2neup_lb)/np.nansum(e2neup_l))
        except:
            n2neup_b.append(0)
        try:
            n2suss_b.append(np.nansum(e2suss_lb)/np.nansum(e2suss_l))
        except:
            n2suss_b.append(0)
        try:
            n2thOKs_b.append(np.nansum(e2thOKs_lb)/np.nansum(e2thOKs_l))
        except:
            n2thOKs_b.append(0)
        try:
            n2thneus_b.append(np.nansum(e2thneus_lb)/np.nansum(e2thneus_l))
        except:
            n2thneus_b.append(0)
        try:
            n2thnos_b.append(np.nansum(e2thnos_lb)/np.nansum(e2thnos_l))
        except:
            n2thnos_b.append(0)
        try:
            n2lp_b.append(np.nansum(e2lp_lb)/np.nansum(e2lp_l))
        except:
            n2lp_b.append(0)
        try:
            n2num_b.append(np.nansum(e2num_lb)/np.nansum(e2num_l))
        except:
            n2num_b.append(0)
        try:
            n2central_b.append(np.nansum(e2central_lb)/np.nansum(e2central_l))
        except:
            n2central_b.append(0)
        #
        #add_argument
        try:
            n2chcentral_b.append(np.nansum(e2chcentlb)/np.nansum(e2chcentl))
        except:
            n2chcentral_b.append(0)
        try:
            n2neucentral_b.append(np.nansum(e2neucentlb)/np.nansum(e2neucentl))
        except:
            n2neucentral_b.append(0)
        try:
            n2suscentral_b.append(np.nansum(e2susscentlb)/np.nansum(e2susscentl))
        except:
            n2suscentral_b.append(0)
        try:
            n2thOKcentral_b.append(np.nansum(e2thOKcentlb)/np.nansum(e2thOKcentl))
        except:
            n2thOKcentral_b.append(0)
        try:
            n2thneucentral_b.append(np.nansum(e2thneucentlb)/np.nansum(e2thneucentl))
        except:
            n2thneucentral_b.append(0)
        try:
            n2thnocentral_b.append(np.nansum(e2thnocentlb)/np.nansum(e2thnocentl))
        except:
            n2thnocentral_b.append(0)
        #nd3
        edges3 = list(nx.bfs_edges(pigg, source = k, depth_limit = 3))
        nodes3 = [v for u, v in edges3]
        nodes3a = set(nodes3)-set(nodes2)
        nodes3b = list(nodes3a)
        legend_l3=[]
        legend_l3b=[]
        edges3a = set(edges3) - set(edges2)
        edges3b = list(edges3a)
        #legend_l2=[]
        #legend_l2b=[]
        #legendcheck
        e3num_l = []
        e3chp_l = []
        e3neup_l = []
        e3suss_l = []
        e3thOKs_l = []
        e3thneus_l = []
        e3thnos_l = []
        e3lp_l = []
        e3legend_l=[]
        e3legendfrom_l=[]
        e3usernamefrom_l=[]
        e3user_l=[]
        #b
        e3chp_lb = []
        e3neup_lb = []
        e3suss_lb = []
        e3thOKs_lb = []
        e3thneus_lb = []
        e3thnos_lb = []
        e3lp_lb = []
        e3legend_lb =[]
        #e3user_l=[]
        e3legend_lb=[]
        e3num_lb=[]
        e3central_l=[]
        e3central_lb=[]
        #
        #add_argument
        e3chcentl=[]
        e3neucentl=[]
        e3susscentl=[]
        e3thOKcentl=[]
        e3thneucentl=[]
        e3thnocentl=[]
        e3chcentlb=[]
        e3neucentlb=[]
        e3susscentlb=[]
        e3thOKcentlb=[]
        e3thneucentlb=[]
        e3thnocentlb=[]
        for e3 in edges3b:
            e3legendfrom_l.append(legendtimedicb[e3[0]])
            e3usernamefrom_l.append(e3[0])
            e3legend_l.append(legendtimedicb[e3[1]])
            e3user_l.append(usertimedicb[e3[1]])
            ke3 = str(e3[0])+"_" + str(e3[1])
            le3 = talknumdicb[ke3]
            e3num_l.append(le3[0])
            e3chp_l.append(le3[1])
            e3neup_l.append(le3[2])
            e3suss_l.append(le3[3])
            e3thOKs_l.append(le3[4])
            e3thneus_l.append(le3[5])
            e3thnos_l.append(le3[6])
            e3lp_l.append(le3[6])
            e3central_l.append(centraldic[e3[1]])
            #
            l6 = chcentraldicb[e3[1]]
            e3chcentl.append(l6[0])
            e3neucentl.append(l6[1])
            e3susscentl.append(l6[2])
            e3thOKcentl.append(l6[3])
            e3thneucentl.append(l6[4])
            e3thnocentl.append(l6[5])

        for i in range(len(e3legend_l)):
            flegend3 = e3legend_l[i]
            fromusername3 = e3usernamefrom_l[i]
            if flegend3 != flegend3:
                continue
            flegendtime3 = dt.strptime(flegend3, '%Y-%m-%d')
            if fromusername3 not in talkname2_lb:
                continue
            usertime3 = usertimedicb[fromusername3]
            usertime3b = dt.strptime(usertime3, '%Y-%m-%d')
            d3 = flegendtime3 - usertime3b #friendのfriendのlegendが自分の発話よりも後（新しい）なら、正になる
            d3 = d3.total_seconds()
            if d3 > 0:
                e3chp_lb.append(e3chp_l[i])
                e3neup_lb.append(e3neup_l[i])
                e3suss_lb.append(e3suss_l[i])
                e3thOKs_lb.append(e3thOKs_l[i])
                e3thneus_lb.append(e3thneus_l[i])
                e3thnos_lb.append(e3thnos_l[i])
                e3num_lb.append(e3num_l[i])
                e3central_lb.append(e3central_l[i])
                #
                e3chcentlb.append(e3chcentl[i])
                e3neucentlb.append(e3neucentl[i])
                e3susscentlb.append(e3susscentl[i])
                e3thOKcentlb.append(e3thOKcentl[i])
                e3thneucentlb.append(e3thneucentl[i])
                e3thnocentlb.append(e3thnocentl[i])

            if fromusername3 not in legendname2_lb:
                continue
            legendtime3 = legendtimedicb[fromusername3]
            legendtime3b = dt.strptime(legendtime3, '%Y-%m-%d')
            try:
                d3b = flegendtime3 - legendtime3b #friendのほうが自分よりも後（新しい）なら、正になる
            except:
                continue
            d3b = d3b.total_seconds()
            if d3b > 0:
                e3legend_lb.append(flegend3)
                e3lp_lb.append(e3lp_l[i])
        try:
            n3legend_b.append(len(e3legend_lb)/len(e3legend_l))
        except:
            n3legend_b.append(0)
        try:
            n3chp_b.append(np.nansum(e3chp_lb)/np.nansum(e3chp_l))
        except:
            n3chp_b.append(0)
        try:
            n3neup_b.append(np.nansum(e3neup_lb)/np.nansum(e3neup_l))
        except:
            n3neup_b.append(0)
        try:
            n3suss_b.append(np.nansum(e3suss_lb)/np.nansum(e3suss_l))
        except:
            n3suss_b.append(0)
        try:
            n3thOKs_b.append(np.nansum(e3thOKs_lb)/np.nansum(e3thOKs_l))
        except:
            n3thOKs_b.append(0)
        try:
            n3thneus_b.append(np.nansum(e3thneus_lb)/np.nansum(e3thneus_l))
        except:
            n3thneus_b.append(0)
        try:
            n3thnos_b.append(np.nansum(e3thnos_lb)/np.nansum(e3thnos_l))
        except:
            n3thnos_b.append(0)
        try:
            n3lp_b.append(np.nansum(e3lp_lb)/np.nansum(e3lp_l))
        except:
            n3lp_b.append(0)
        try:
            n3num_b.append(np.nansum(e3num_lb)/np.nansum(e3num_l))
        except:
            n3num_b.append(0)
        try:
            n3central_b.append(np.nansum(e3central_lb)/np.nansum(e3central_l))
        except:
            n3central_b.append(0)
        #
        #add_argument
        try:
            n3chcentral_b.append(np.nansum(e3chcentlb)/np.nansum(e3chcentl))
        except:
            n3chcentral_b.append(0)
        try:
            n3neucentral_b.append(np.nansum(e3neucentlb)/np.nansum(e3neucentl))
        except:
            n3neucentral_b.append(0)
        try:
            n3suscentral_b.append(np.nansum(e3susscentlb)/np.nansum(e3susscentl))
        except:
            n3suscentral_b.append(0)
        try:
            n3thOKcentral_b.append(np.nansum(e3thOKcentlb)/np.nansum(e3thOKcentl))
        except:
            n3thOKcentral_b.append(0)
        try:
            n3thneucentral_b.append(np.nansum(e3thneucentlb)/np.nansum(e3thneucentl))
        except:
            n3thneucentral_b.append(0)
        try:
            n3thnocentral_b.append(np.nansum(e3thnocentlb)/np.nansum(e3thnocentl))
        except:
            n3thnocentral_b.append(0)

    n1new_dir_path = "random20210514b/net1"

    try:
        os.makedirs(n1new_dir_path)
    except FileExistsError:
        pass

    n2new_dir_path = "random20210514b/net2"

    try:
        os.makedirs(n2new_dir_path)
    except FileExistsError:
        pass

    n3new_dir_path = "random20210514b/net3"

    try:
        os.makedirs(n3new_dir_path)
    except FileExistsError:
        pass

    filename1 = n1new_dir_path + "/random" + str(i2) + args.input
    filename2 = n2new_dir_path + "/random" + str(i2) + args.input
    filename3 = n3new_dir_path + "/random" + str(i2) + args.input

    df1=pd.DataFrame()
    df1["idx"]= n1idx
    df1["legend_b"]= n1legend_b
    #networkeffects
    df1["chp_b"]= n1chp_b
    df1["neup_b"]= n1neup_b
    df1["suss_b"]= n1suss_b
    df1["thOKs_b"]= n1thOKs_b
    df1["thneus_b"]= n1thneus_b
    df1["thnos_b"]= n1thnos_b
    df1["lp_b"]= n1lp_b
    df1["num_b"]=n1num_b
    df1["central_b"]=n1central_b

    df1["chcentral_b"] = n1chcentral_b
    df1["neucentral_b"] = n1neucentral_b
    df1["susscentral_b"] = n1suscentral_b
    df1["thOKcentral_b"] = n1thOKcentral_b
    df1["thneucentral_b"] = n1thneucentral_b
    df1["thnocentral_b"] = n1thnocentral_b


    df1 = df1.fillna(0)
    #filename = "genuine0514bnet1" + args.input
    df1.to_csv(filename1,index= False)





    df2=pd.DataFrame()
    df2["idx"]= n2idx
    df2["legend_b"]= n2legend_b
    #networkeffects
    df2["chp_b"]= n2chp_b
    df2["neup_b"]= n2neup_b
    df2["suss_b"]= n2suss_b
    df2["thOKs_b"]= n2thOKs_b
    df2["thneus_b"]= n2thneus_b
    df2["thnos_b"]= n2thnos_b
    df2["lp_b"]= n2lp_b
    df2["num_b"]=n2num_b
    df2["central_b"]=n2central_b



    df2["chcentral_b"] = n2chcentral_b
    df2["neucentral_b"] = n2neucentral_b
    df2["susscentral_b"] = n2suscentral_b
    df2["thOKcentral_b"] = n2thOKcentral_b
    df2["thneucentral_b"] = n2thneucentral_b
    df2["thnocentral_b"] = n2thnocentral_b

    df2 = df2.fillna(0)
    #filename = "genuine0514bnet2" + args.input
    df2.to_csv(filename2,index= False)



    df3=pd.DataFrame()
    df3["idx"]= n3idx
    df3["legend_b"]= n3legend_b
    #networkeffects
    df3["chp_b"]= n3chp_b
    df3["neup_b"]= n3neup_b
    df3["suss_b"]= n3suss_b
    df3["thOKs_b"]= n3thOKs_b
    df3["thneus_b"]= n3thneus_b
    df3["thnos_b"]= n3thnos_b
    df3["lp_b"]= n3lp_b
    df3["num_b"]=n3num_b
    df3["central_b"]=n3central_b

    df3["chcentral_b"] = n3chcentral_b
    df3["neucentral_b"] = n3neucentral_b
    df3["susscentral_b"] = n3suscentral_b
    df3["thOKcentral_b"] = n3thOKcentral_b
    df3["thneucentral_b"] = n3thneucentral_b
    df3["thnocentral_b"] = n3thnocentral_b

    df3 = df3.fillna(0)

    #filename = "genuine0514bnet3" + args.input
    df3.to_csv(filename3,index= False)
