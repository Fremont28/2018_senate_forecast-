#10/21/18 *********

#--- updating rep and dem polls? 
import csv
import math
import pandas as pd 
import numpy as np 

senate_adv=5 #senate advantage is worth 5 points 
K=1.9 #speed of ratings change 

senate=pd.read_csv("senate_train.csv")
senate=senate.iloc[:,1:7]
senate1=senate.values
senate1=senate1[:,1:7]


##10/22/18 (tennis opponent ranking system??)*********************
slam18=pd.read_csv("atp_slam_2018.csv")
s=slam18[["tourney_name","surface","winner_name","winner_rank","winner_rank_points","loser_name","loser_rank","loser_rank_points","win"]]
s1=s[["winner_name","loser_name","winner_rank_points","loser_rank_points"]]

def elo_update(A,B):
    return 1/(1+10**((B-A)/400))

def elo(old,score,k=2):
    return old+k*(score-exp)


wr=s1['winner_rank_points'].mean()
lr=s1['loser_rank_points'].mean()
rank_threshold=(wr+lr)/2  

#leggero: https://github.com/HankMD/EloPy/blob/master/elopy.py
def match_update(rating1,rating2,win):
    #player1=s.winner_name
    #player2=s.loser_name 
    #rating1=s.winner_rank_points
    #rating2=s.loser_rank_points
    rank_threshold=1313 
    if win==1:
        score1=50 #+50 
        score2=-50 #-50 
    if win==0:
        score1=-50 
        score2=50 
    k=0.5
    newRating1=rating1+score1
    newRating2=rating2+score2 
    if rating1>rank_threshold:
        newRating1=newRating1+10
    else:
        newRating1=newRating1 
    if rating2>rank_threshold:
        newRating2=newRating2+10
    else:
        newRating2=newRating2 
    return newRating1,newRating2 

xx=np.vectorize(match_update)(s['winner_rank_points'],s['loser_rank_points'],s['win'])
xy=np.asarray(xx)

#ASADA??
df = pd.read_csv("schedule.csv")
col1 = "\xef\xbb\xbfAway"
col2 = "Home"



#10/23/18 ---- october pumpkin beer/pie???---------
from scipy import optimize
from scipy.stats import norm

players_win=list(s['winner_name']) 
players_lose=list(s['loser_name'])

#apply index to players?
s['winner_id']=s['winner_name'].apply(lambda x: players_win.index(x))
s['loser_id']=s['loser_name'].apply(lambda x: players_lose.index(x))

sX=s['loser_id'].unique() 
sX1=s['winner_id'].unique()

n_players=2494
col1="winner_name"
col2="loser_name"

def ratings(x):
    return np.mean(x)

def obj(x):
    err=0
    s['proj']=3+s.winner_id.apply(lambda i:x[i]) #adding 3 points to winner rating
    s['winner_pr']=1-norm.cdf(0.5,s['proj'],14.5)
    s['loser_pr']=1-s['winner_pr'] #subtracting winner points rating edge from loser points rating 
    w=np.zeros(shape=n_players)
    for i in range(len(s)):
        w[players_win.index(s[col1][i])]=w[players_win.index(s[col1][i])]+s['winner_pr'][i]
        #w[players_lose.index(s[col1][i])]=w[players_lose.index(s[col1][i])]+s['loser_pr'][i]
        error=((s['winner_rank_points']-w)**2).sum() 

x0=np.zeros(shape=n_players)

res = optimize.minimize(obj,x0, constraints=[{'type':'eq', 'fun':ratings}], method="SLSQP",
                        options={'maxiter':10000})



#----
players_win=list(s['winner_name']) 
s['winner_id']=s['winner_name'].apply(lambda x: players_win.index(x))


def pumpkin(x):
    s['winner_pr']=3+s.winner_id.apply(lambda i:x[i])
    return s['winner_pr']


#---update tennis players??
p1_rtg=s['winner_rank_points'].values 
p2_rtg=s['loser_rank_points'].values 
win=s['win'].values 

s1=s.values.tolist() 

p1=[]
p2=[]
def player_elo(win,p1_rtg,p2_rtg):
    #for i in range(0,len(s1)):
    p1_rtg=s['winner_rank_points'].values 
    p2_rtg=s['loser_rank_points'].values 
    win=s['win'].values 
    if win==1:
        p1=(p1_rtg+50)+(p2_rtg/30)
        p2=(p2_rtg-50)
    else:
        p1=(p1_rtg-50)
        p2=(p2_rtg+50)+(p1/30)
    return p1,p2 


player_elo(win,p1_rtg,p2_rtg)


### pumpkin brewing ********
#VaR and Expected shortfall??---------- leggero: https://quantdare.com/value-at-risk-or-expected-shortfall/
from pandas_datareader import data as pdr

coin_brew=pd.read_csv("coin_daily_histories.csv")

#VaR function
def value_at_risk(returns,confidence_level=0.05):
    return returns.quantile(confidence_level,interpolation="higher")

#general variables
ticker='BTC'
first_date='1/24/15'
last_date='1/31/18'
confidence_level=0.05 
k=0.02

btc=coin_brew[coin_brew['symbol']=="BTC"]
btc_returns=btc["value"]

#VaR (value at risk??)
var=value_at_risk(btc_returns)

#leggero: https://www.quantinsti.com/blog/calculating-value-at-risk-in-excel-python/
###hazelnut cerveza??? ***********************************
#i. 
btc=coin_brew[coin_brew['symbol']=="BTC"]
btc=btc[["value"]]
btc['returns']=btc.value.pct_change() 

#mean and std 
mean=np.mean(btc['returns'])
std=np.std(btc['returns'])

#vAr (point percentile funciton)
var_90=norm.ppf(1-0.9,mean, std) #90% 
var_95=norm.ppf(1-0.95,mean, std) #95%

#ii. 
#************************---------------------
### value added risk (everything??) *********
coin_brew1=coin_brew[coin_brew['value']>0]
value_avg=coin_brew1['value'].mean() 
coin_brew1['value']=coin_brew1['value'].fillna(0)
coin_brew1['returns']=coin_brew1.value.pct_change()
coin_brew1['returns']=coin_brew1['returns'].fillna(0)

#mean and std 
mean=np.mean(coin_brew['returns'])
std=np.std(coin_brew['returns'])
std=0.04
mean=13 

#VaR by coin?? ----------------
coin_mean_ret=coin_brew1.groupby('symbol')['returns'].mean() 
coin_mean_ret=pd.DataFrame(coin_mean_ret)
coin_mean_ret.reset_index(level=0,inplace=True)

coin_std_ret=coin_brew1.groupby('symbol')['returns'].std() 
coin_std_ret=pd.DataFrame(coin_std_ret)
coin_std_ret.reset_index(level=0,inplace=True)

combine=pd.merge(coin_mean_ret,coin_std_ret,on="symbol")
combine.columns=['symbol','coin_ret_mean','coin_ret_std']

#return confidence interval (68%)
combine['68%_CI_high']=combine['coin_ret_mean']+combine['coin_ret_std']
combine['68%_CI_low']=combine['coin_ret_mean']-combine['coin_ret_std']

#iii. ----- 95% confidence intervals (por coin?)
import numpy as np 
import scipy.stats 
    
data=coin_brew1['returns'].tolist() 
btc=coin_brew1[coin_brew1['symbol']=='BTC']

data1=btc['returns'].tolist() 


def mean_confidence_interval(data,confidence=0.95):
    a=1.0*np.array(data1)
    n=len(a)
    m,se=np.mean(a),scipy.stats.sem(a)
    h=se*scipy.stats.t.ppf((1+confidence)/2.,n-1)
    return m, m-h,m+h 

mean_confidence_interval(data1)










































class Forecast():
    def __init__(self):
        pass
    


    def forecast(self):
        dem=senate1[:,2]
        rep=senate1[:,1]
        if dem>rep:
            dem +=1
        else:
            rep -=1

if __name__=='__main__':
    f=Forecast() 
    f.forecast()





