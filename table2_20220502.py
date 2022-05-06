
from sys import stdout
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings("ignore")

import statsmodels.api as sm
import scipy
#import scipy.stats
from scipy.stats import t  # We only need the t class from scipy.stats
from scipy import stats
from scipy.signal import savgol_filter

from sklearn import preprocessing
from sklearn import datasets, linear_model
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.utils import shuffle
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--input', type=str, required=True,help="input file")
#parser.add_argument('-di', '--di', type=str, required=True,help="input file")
args = parser.parse_args()

df = pd.read_csv(args.input)

#cs = df.columns.values.tolist()

y = df["m_toslegends"].values


talks = ["nums","chans","neuts","suss","thOKs","thneus","thnos"]
desc = ["c_ages","c_males","c_years","c_debts"]
social =["indegrees","numlegends"]
symp = ["c_s1s","c_s3s","c_s4s","c_s7s","c_s9s","c_s10s","c_snums"]
#c_s0は存在しない#全員symptomが確認されているので。

cs2 = talks + desc + social + symp

X=df[cs2].values


def prepare_inputs2(x_data):
    #xdata = np.concatenate([x_train, x_test])
    #imp = IterativeImputer(max_iter=10, random_state=0)
    #imp = SimpleImputer(missing_values=np.nan, strategy='mean')
    imp = SimpleImputer(missing_values=np.nan, strategy='constant', fill_value=0)
    ss = preprocessing.StandardScaler()
    #imp.fit(xdata)
    #x_train2 = imp.transform(x_train)
    x_data = imp.fit_transform(x_data)
    #X_data = x_test2
    #x_test2 = np.array()
    #x_test2 = x_test2 + 1/2  #1/2足す
    #x_test2 = np.log2(x_test2)
    x_data = scipy.stats.zscore(x_data, axis=0)#列ごと

    return x_data

X = prepare_inputs2(X)
print(X)
X2 = sm.add_constant(X)
#X = sm.add_constant(x) としている部分は、説明変数の一列目に新しく全要素が1.0の列を追加しています。これは、もし切片を必要とする線形回帰のモデル式ならば必ず必要な部分で、これを入れないと正しく回帰式が作成されません。

cs3 = cs2 + ["legend"]
#cs3 = list(cs3)
n = X2.shape[0]

k = X2.shape[1]

def optimise_pls_cv(X, y, n_comp, plot_components=True):

    '''Run PLS including a variable number of components, up to n_comp,
       and calculate MSE '''

    #mse = []
    auc = []
    component = np.arange(1, n_comp)
    nums = []

    for i in component:
        pls = PLSRegression(n_components=i)

        # Cross-validation
        y_cv = cross_val_predict(pls, X, y, cv=10)

        #mse.append(mean_squared_error(y, y_cv))
        nums.append(i)
        auc.append(roc_auc_score(y, y_cv))

        comp = 100*(i+1)/40
        # Trick to update status on the same line
        stdout.write("\r%d%% completed" % comp)
        stdout.flush()
    stdout.write("\n")

    # Calculate and print the position of minimum in MSE
    #msemin = np.argmin(mse)
    aucmax = np.argmax(auc)
    bestnum = aucmax+1
    print("auc_score",auc[aucmax])
    print("Suggested number of components: ", aucmax+1)
    stdout.write("\n")

    if plot_components is True:
        with plt.style.context(('seaborn-colorblind')):
            plt.plot(component, np.array(auc), marker ="+", color = 'blue', mfc='blue')
            #plt.plot(component[aucmax], np.array(auc)[aucmax], 'P', ms=10, mfc='red')
            plt.plot(component[aucmax], np.array(auc)[aucmax], marker="o", ms=10, mfc='red')
            plt.xlabel('Number of PLS components')
            plt.ylabel('AUC')
            #plt.title('PLS')
            plt.xlim(left=-1)

        #plt.show()
        filename = "appendixauc_" + args.input + ".png"
        plt.savefig(filename)

    # Define PLS object with optimal number of components
    #pls_opt = PLSRegression(n_components=msemin+1)
    nums.append(bestnum)
    auc.append(auc[aucmax])

    pls_opt = PLSRegression(n_components=aucmax+1)

    # Fir to the entire dataset
    pls_opt.fit(X, y)
    y_c = pls_opt.predict(X)
    print("best_roc_auc_score",roc_auc_score(y, y_c))
    nums.append("best_roc_auc_score")
    auc.append(roc_auc_score(y, y_c))
    print("bestrmse",np.sqrt(mean_squared_error(y, y_c)))
    nums.append("best_rmse")
    auc.append(np.sqrt(mean_squared_error(y, y_c)))
    print("pls_r2",pls_opt.score(X, y))
    nums.append("best_pls_r2")
    auc.append(pls_opt.score(X, y))


    beta_hat3 = np.squeeze(list(pls_opt.coef_))
    print("beta3",beta_hat3)
    residual = y - np.matmul(X, beta_hat3)  # calculate the residual
    sigma_hat = sum(residual ** 2) / (n - k - 1)  # estimate of error term variance
    #variance_beta_hat3 = sigma_hat * np.linalg.inv(np.matmul(X.transpose(), X))  # Calculate variance of OLS estimate
    variance_beta_hat3 = sigma_hat * np.linalg.pinv(np.matmul(X.transpose(), X))
    se = np.sqrt(variance_beta_hat3.diagonal())


    #print(variance_beta_hat)
    t_stat = beta_hat3 / np.sqrt(variance_beta_hat3.diagonal())
    print("t_values3",t_stat)

    p_value = 1 - 2 * np.abs(0.5 - np.vectorize(t.cdf)(t_stat, n - k - 1))
    print("p_values3",p_value)
    return beta_hat3,se,t_stat,p_value,auc,nums

beta_hat3,se,t_stat,p_value,auc,nums = optimise_pls_cv(X2,y, 40, plot_components=True)

beta_hat3 = np.delete(beta_hat3,0)
se = np.delete(se,0)
t_stat = np.delete(t_stat,0)
p_value = np.delete(p_value,0)
#切片の値を抜く

filename = "Table2results20220208.csv"
df = pd.DataFrame()
df["index"]=cs2
df["plsbeta"]=beta_hat3
df["se"] = se
df["t_values"]=t_stat
df["p_values"]=p_value
df.to_csv(filename,index= False)

filename = "Appendixaucresults20220208b.csv"
df2 = pd.DataFrame()
df2["nums"]=nums
df2["auc"]=auc
df2.to_csv(filename,index= False)
