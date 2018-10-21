
#Unadjusted 2018 Senate Forecast - 10/21/18 ---------------

import pandas as pd 
import numpy as np 
import sklearn 
from sklearn.linear_model import LogisticRegression

s_16=pd.read_csv("senate16.csv") #2012, 2014, and 2016 senate elections 

#margin republicans vs. democratic candidate 
s_16['margin']=s_16['Rep']-s_16['Dem']

#average by year and state 
s12=s_16[s_16.Year==2012] #2012

#group by state 
s12_margin=s12.groupby('State')['margin'].mean() #margin
s12_rep=s12.groupby('State')['Rep'].mean() #avg republican vote %
s12_dem=s12.groupby('State')['Dem'].mean() #avg democratic vote % 
s13_std=s12.groupby('State')['margin'].std() #avg standard deviation of polling
s12_std=s12_std.fillna(0)

#convert series to dataframe
s12_margin=pd.DataFrame(s12_margin)
s12_margin.reset_index(level=0,inplace=True)
s12_rep=pd.DataFrame(s12_rep)
s12_rep.reset_index(level=0,inplace=True)
s12_dem=pd.DataFrame(s12_dem)
s12_dem.reset_index(level=0,inplace=True)
s12_std=pd.DataFrame(s12_std)
s12_std.reset_index(level=0,inplace=True)
s12_std.columns=['State','Error']

#merge 2012 senate dataframes
merge_2012=pd.merge(s12_margin,s12_rep,on="State")
merge_2012=pd.merge(merge_2012,s12_dem,on="State")
merge_2012=pd.merge(merge_2012,s12_std,on="State")
merge_2012=merge_2012[["State","margin","Rep","Dem","Error"]]

s14=s_16[s_16.Year==2014] #2014 senate 

#group by state 
s14_margin=s14.groupby('State')['margin'].mean() #margin
s14_rep=s14.groupby('State')['Rep'].mean() #avg republican vote %
s14_dem=s14.groupby('State')['Dem'].mean() #avg democratic vote % 
s14_std=s14.groupby('State')['margin'].std() #avg standard deviation of polling
s14_std=s14_std.fillna(0)

#convert series to dataframe
s14_margin=pd.DataFrame(s14_margin)
s14_margin.reset_index(level=0,inplace=True)
s14_rep=pd.DataFrame(s14_rep)
s14_rep.reset_index(level=0,inplace=True)
s14_dem=pd.DataFrame(s14_dem)
s14_dem.reset_index(level=0,inplace=True)
s14_std=pd.DataFrame(s14_std)
s14_std.reset_index(level=0,inplace=True)
s14_std.columns=['State','Error']

#merge 2014 senate dataframes
merge_2014=pd.merge(s14_margin,s14_rep,on="State")
merge_2014=pd.merge(merge_2014,s14_dem,on="State")
merge_2014=pd.merge(merge_2014,s14_std,on="State")
merge_2014=merge_2014[["State","margin","Rep","Dem","Error"]]

s16=s_16[s_16.Year==2016] #2016 senate 

#group by state 
s16_margin=s16.groupby('State')['margin'].mean() #margin
s16_rep=s16.groupby('State')['Rep'].mean() #avg republican vote %
s16_dem=s16.groupby('State')['Dem'].mean() #avg democratic vote % 
s16_std=s16.groupby('State')['margin'].std() #avg standard deviation of polling
s16_std=s16_std.fillna(0)

#convert series to dataframe
s16_margin=pd.DataFrame(s16_margin)
s16_margin.reset_index(level=0,inplace=True)
s16_rep=pd.DataFrame(s16_rep)
s16_rep.reset_index(level=0,inplace=True)
s16_dem=pd.DataFrame(s16_dem)
s16_dem.reset_index(level=0,inplace=True)
s16_std=pd.DataFrame(s16_std)
s16_std.reset_index(level=0,inplace=True)
s16_std.columns=['State','Error']

#merge 2016 senate dataframes
merge_2016=pd.merge(s16_margin,s16_rep,on="State")
merge_2016=pd.merge(merge_2016,s16_dem,on="State")
merge_2016=pd.merge(merge_2016,s16_std,on="State")
merge_2016=merge_2016[["State","margin","Rep","Dem","Error"]]


#combine 2012, 2014, # 2016 dataframes 
final_senate=pd.merge(merge_2012,merge_2014,on="State")
final_senate.reset_index(level=0,inplace=True)
final_senate.columns=['index','State','Margin_12','Rep_12','Dem_12','Error_12','Margin_16',
'Rep_14','Dem_14','Error_14']
final_senate.to_csv("final_senate1214.csv")

combo_121416=pd.concat([merge_2012,merge_2014,merge_2016],axis=0)
combo_121416.to_csv("senate_train.csv")


##### 2018 senate forecast
s_2018=pd.read_csv("senate18.csv")
s_2018['margin']=s_2018['Rep']-s_2018['Dem']

#average by year and state***************
#2018
s18=s_2018[s_2018.Year==2018]

#group by state 
s18_margin=s18.groupby('State')['margin'].mean() #margin
s18_rep=s18.groupby('State')['Rep'].mean() #avg republican vote %
s18_dem=s18.groupby('State')['Dem'].mean() #avg democratic vote % 
s18_std=s18.groupby('State')['margin'].std() #avg standard deviation of polling
s18_std=s18_std.fillna(0)

#convert series to dataframe
s18_margin=pd.DataFrame(s18_margin)
s18_margin.reset_index(level=0,inplace=True)
s18_rep=pd.DataFrame(s18_rep)
s18_rep.reset_index(level=0,inplace=True)
s18_dem=pd.DataFrame(s18_dem)
s18_dem.reset_index(level=0,inplace=True)
s18_std=pd.DataFrame(s18_std)
s18_std.reset_index(level=0,inplace=True)
s18_std.columns=['State','Error']

#merge 2012 senate dataframes
merge_2018=pd.merge(s18_margin,s18_rep,on="State")
merge_2018=pd.merge(merge_2018,s18_dem,on="State")
merge_2018=pd.merge(merge_2018,s18_std,on="State")
merge_2018=merge_2018[["State","margin","Rep","Dem","Error"]]

merge_2018.to_csv("senate_test.csv")

####build a basic probability model (senate forecast)
#unadjusted for polls (using logistic regression)

#import train and test senate datasets
train_senate=pd.read_csv("senate_train.csv")
test_senate=pd.read_csv("senate_test.csv")

X_train=train_senate[["margin","Rep","Dem","Error","Senate"]]
y_train=train_senate['Win']
X_test=test_senate[["margin","Rep","Dem","Error","Senate"]]

#i. log loss (perfect score is 0)
model_lr=LogisticRegression(random_state=0,solver='lbfgs').fit(X_train,y_train)
results=model_lr.predict(X_test)
results=pd.DataFrame(results)
results.columns=['Win_predict']

#concat results 
model_results=pd.concat([test_senate,results],axis=1)
model_state=model_results[["State","Win_predict"]]
