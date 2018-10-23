#10/21/18 ---------- adjusting for poll quality and state clustering 

#import train and test senate datasets
sen16=pd.read_csv("senate16X.csv")
sen18=pd.read_csv("senate18.csv")
sen=pd.concat([sen16,sen18],axis=0)

#poll adjustments for 2012, 2014, and 2016 senate races 
sen16['margin']=sen16['Dem']-sen16['Rep']
sen16['margin_abs']=abs(sen16['margin'])
sen16['diff_margin']=sen16['Final_Margin']-sen16['margin']

#poll quality thresholds
sen16_poll_adj=sen16[sen16.diff_margin<=20]

sen16_poll_adj['diff_margin'].mean() #5.37
sen16_poll_adj['diff_margin'].quantile(0.25) #1.70 

def poll_adj(x,y):
    if x<=1.70:
        return y*1.5
    if x>1.70 and x<=5.37:
        return y*1
    if x>5.37:
        return y*0.5 

#adjusted Republican and Democratic vote percentages (based on poll quality)
sen16['adj_Rep']=np.vectorize(poll_adj)(sen16['diff_margin'],sen16['Rep'])
sen16['adj_Dem']=np.vectorize(poll_adj)(sen16['diff_margin'],sen16['Dem'])

sen16.to_csv("senate16_adj.csv")

#ranking poll quality
poll_rankings=sen16.groupby('Poll')['diff_margin'].mean()
poll_rankings=pd.DataFrame(poll_rankings)
poll_rankings.to_csv("poll_rankings.csv")

#poll ratings (using sampled data based on 2012, 2014, and 2016 senate races)
def poll_rankings(x,y):
    if x=="8 News NOW - Las Vegas" or x=="8 News NOW - Las Vegas*" or x=="AFP/Magellan (R)" or x=="Cincinnati Enquirer/Ohio News" or x=="Columbus Dispatch*" or x=="High Point/SurveyUSA" or x=="Marquette University":
        return y*1.5
    if x=="CBS News/NYT/YouGov" or x=="CBS News/NYT/YouGov*" or x=="CBS News/YouGov" or x=="CNN/Opinion Research" or x=="Chicago Tribune*" or x=="Franklin & Marshall" or x=="Franklin & Marshall*" or x=="Fairleigh Dickinson" or x=="FOX News" or x=="Gonzales Research" or x=="Harper (R)" or x=="Hartford Courant/UConn" or x=="Marquette" or x=="NY Times/Kaiser" or x=="Quinnipiac" or x=="Rasmussen Reports" orx=="Rasmussen Reports" or x=="Reuters/Ipsos" or x=="Washington Post":
        return y*0.5
    else:
        return y 
    
#apply poll ratings function to senate 2018
sen18['adj_Rep']=np.vectorize(poll_rankings)(sen18['Poll'],sen18['Rep'])
sen18['adj_Dem']=np.vectorize(poll_rankings)(sen18['Poll'],sen18['Dem'])

#538 poll ratings 
def five_38_poll_ratings(x,y):
    if x=="SurveyUSA" or x=="Mason-Dixon" or x=="Quinnipiac" or x=="Quinnipiac University" or x=="Marist College": 
        return y*2
    if x=="YouGov" or x=="PublicPolicyPolling":
        return y*1.5
    if x=="Rasmussen" or x=="AmericanResearchGroup" or x=="Rasmussen Reports" or x=="Reuters/Ipsos" or x=="Washington Post":
        return y*0.5 
    else:
        return y

#apply 538 poll ratings function to senate 2018 and (12',14',and 16' senate races)
sen18['adj_Rep_538']=np.vectorize(poll_rankings)(sen18['Poll'],sen18['Rep'])
sen18['adj_Dem_538']=np.vectorize(poll_rankings)(sen18['Poll'],sen18['Dem'])
sen16_adj['adj_Rep_538']=np.vectorize(poll_rankings)(sen16_adj['Poll'],sen16_adj['Rep'])
sen16_adj['adj_Dem_538']=np.vectorize(poll_rankings)(sen16_adj['Poll'],sen16_adj['Dem'])

#adjust polls for senate 2018 
sen18['margin']=sen18['Dem']-sen18['Rep']
sen18.to_csv("senate18.csv")

senate18=pd.read_csv("senate18.csv",sep=",") 
sen16_adj=pd.read_csv("senate16_adj.csv",encoding="latin-1")

####### adjusted poll model (senate 2018) ************************
senate18.head(3)
X_train=sen16_adj[["margin","adj_Rep_538","adj_Dem_538"]] 
y_train=sen16_adj['Win']
X_test=senate18[["margin","adj_Rep_538","adj_Dem_538"]]

#i. logistic regression model 
model_lr=LogisticRegression(random_state=0,solver='lbfgs').fit(X_train,y_train)
results=model_lr.predict_proba(X_test) 
results=pd.DataFrame(results)
results.columns=['Loss_Dem','Win_Dem']

#concat results 
model_results=pd.concat([senate18,results],axis=1)
model_state=model_results[["State","Loss_Dem","Win_Dem"]]

#democrats' win probability average by state 
dem_win_prob=model_state.groupby('State')['Win_Dem'].mean() 
dem_win_prob=pd.DataFrame(dem_win_prob) 
dem_win_prob.reset_index(level=0,inplace=True)
dem_win_prob.columns=['State','Win_Prob']

#deomcrats' win probability std by state
dem_win_std=model_state.groupby('State')['Win_Dem'].std() 
dem_win_std.fillna(1)
dem_win_std=pd.DataFrame(dem_win_std)
dem_win_std.reset_index(level=0,inplace=True)
dem_win_std.columns=['State','Std']
dem_win_std.Std.fillna(1)

final_senate18=pd.merge(dem_win_prob,dem_win_std,on="State")
final_senate18['High_Win']=final_senate18['Win_Prob']+final_senate18['Std']
final_senate18['Low_Win']=final_senate18['Win_Prob']-final_senate18['Std']

def adj_high_low_win(x):
    if x>1.00:
        return 1
    if x<0:
        return 0 
    else:
        return x

final_senate18['adjHigh_win']=final_senate18['High_Win'].apply(adj_high_low_win)
final_senate18['adjLow_win']=final_senate18['Low_Win'].apply(adj_high_low_win)

final_senate18_sub=final_senate18[["State","Win_Prob","adjHigh_win","adjLow_win"]]


###10/22/18 *******************************
#cluster states (similarity scores)
sen16=pd.read_csv("senate16X.csv")
sen18=pd.read_csv("senate18.csv")
sen=pd.concat([sen16,sen18],axis=0)

from sklearn.cluster import KMeans

cluster_vars=sen[["Dem","Rep"]]
kmeans=KMeans(n_clusters=3,random_state=0).fit(cluster_vars)
state_labels=kmeans.labels_
state_labels=pd.DataFrame(state_labels)
state_labels.columns=['Group']

state=sen['State']
state=pd.DataFrame(state,columns=['State'])
state.reset_index(level=0,inplace=True)

state_group=pd.concat([state,state_labels],axis=1)
state_group_avg=state_group.groupby('State')['Group'].mean() 

#apply state grouping to senate (12, 14, and 16) and senate 2018 
def state_group(x):
    if x=="AK" or x=="AR" or x=="AZ" or x=="CO" or x=="FL" or x=="GA" or x=="IA" or x=="IN" or x=="KS" or x=="KY" or x=="LA" or x=="ME" or x=="MT" or x=="MZ" or x=="NC" or x=="ND" or x=="NV" or x=="OH" or x=="PA" or x=="VA" or x=="WI" or x=="WV" or x=="NH":
        return 1
    if x=="CA" or x=="CT" or x=="DE" or x=="HI" or x=="IL" or x=="MA" or x=="MD" or x=="MI" or x=="MN" or x=="OR" or x=="RI" or x=="VT" or x=="WA":
        return 2
    else:
        return 0 

senate18['state_group']=senate18['State'].apply(state_group)
sen16_adj['state_group']=sen16_adj['State'].apply(state_group)

#primero write to csv
senate18.to_csv("senate_oct18.csv")
sen16_adj.to_csv("senate_past.csv")
senate_past=pd.read_csv("senate_past.csv")
senate_18=pd.read_csv("senate_oct18.csv")

####model with state groups
X_train1=senate_past[["margin","adj_Rep_538","adj_Dem_538","state_group"]] 
y_train1=senate_past['Win']
X_test1=senate_18[["margin","adj_Rep_538","adj_Dem_538","state_group"]]

#i. logistic regression model 
model_lr=LogisticRegression(random_state=0,solver='lbfgs').fit(X_train1,y_train1)
results=model_lr.predict_proba(X_test1) #target 0,1 
results=pd.DataFrame(results)
results.columns=['Loss_Dem','Win_Dem']

#concat results 
model_results=pd.concat([senate18,results],axis=1)
model_state=model_results[["State","Loss_Dem","Win_Dem"]]

#democrats' win probability average by state 
dem_win_prob1=model_state.groupby('State')['Win_Dem'].mean() 
dem_win_prob1=pd.DataFrame(dem_win_prob1) 
dem_win_prob1.reset_index(level=0,inplace=True)
dem_win_prob1.columns=['State','Win_Prob']

#poll count 
poll_count=senate_past['Poll'].value_counts() 
poll_count=pd.DataFrame(poll_count)
poll_count.reset_index(level=0,inplace=True)
poll_count.columns=['Poll','count']
#poll accuracy 
poll_acc=senate_past.groupby('Poll')['diff_margin'].mean() 
poll_acc=pd.DataFrame(poll_acc)
poll_acc.reset_index(level=0,inplace=True)

poll_final=pd.merge(poll_count,poll_acc,on="Poll")
poll_final=poll_final[poll_final['count']>=20]
































