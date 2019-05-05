#!/usr/bin/env python
# coding: utf-8

# ### Problem Statement
# **Food & Beverages Spend Prediction in Club Mahindra Resorts**
# 
# Club Mahindra (Club M) makes significant revenue from Food and Beverages (F&B) sales in their resorts. The members of Club M are offered a wide variety of items in either buffet or À la carte form. Following are some benefits that the model to predict the spend by a member in their next visit to a resort will bring:
# Predicting the F&B spend of a member in a resort would help in improving the pre-sales during resort booking through web and mobile app
# Targeted campaigns to suit the member taste and preference of F&B
# Providing members in the resort with a customized experience and offers
# Help resort kitchen to plan the inventory and food quantity to be prepared in advance
# Given the information related to resort, club member, reservation etc. the task is to predict average spend per room night on food and beverages for the each reservation in the test set.
#  
# 
# ### Data Description
# **train.zip**
# 
# train.zip contains train.csv and data_dictionary.csv.
# 
# train.csv contains the training data with details on a set of reservations with the average spend per room night
# 
# Data_Dictionary.xlsx contains a brief description of each variable provided in the training and test set.
#  
# **test.csv**
# 
# test.csv contains details of all reservations for which the participants need to predict the average spend on FnB per room night
#  
# 
# **sample_submission.csv**
# 
# sample_submission.csv contains the submission format for the predictions against the test set. A single csv/zip needs to be submitted as a solution.
#  
# 
# **Evaluation Metric**
# 
# Submissions are evaluated on 100 * Root Mean Squared Error (RMSE) on the variable amount_spent_per_room_night_scaled
# 

# In[57]:


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import date
pd.set_option('display.max_columns', None)


# In[58]:


train=pd.read_csv("train_5CLrC8b/train.csv")
test=pd.read_csv("test.csv")


# In[59]:


train.shape


# In[60]:


test.shape


# In[61]:


train.head()


# In[62]:


train['reservation_id'].nunique()


# In[63]:


train['memberid'].nunique()


# ### Convert the date columns from string to datetime and Extract date features from time columns

# In[64]:


def extractDateTimeFeature(date_column,data):
    data[date_column+"_year"]=data[date_column].dt.year
    data[date_column+"_day"]=data[date_column].dt.day
    data[date_column+"_month"]=data[date_column].dt.month
    data[date_column+"_dayofweek"]=data[date_column].dt.dayofweek
    data[date_column+"_isweekend"]=data[date_column+"_dayofweek"].apply(lambda x:1 if x==5 or x==6 else 0)
    return data


# In[65]:


date_columns=['booking_date','checkin_date','checkout_date']
for col in date_columns:
    train[col]=pd.to_datetime(train[col],dayfirst=True)
    test[col]=pd.to_datetime(test[col],dayfirst=True)
    train=extractDateTimeFeature(col,train)
    test=extractDateTimeFeature(col,test)


# In[66]:


train.head()


# ### Are there any missing values in the data?

# In[67]:


pd.isnull(train).sum()


# There are missing value in season_holidayed_code and state_code_residence in the train. Is this seeen in the test data as well

# In[68]:


pd.isnull(test).sum()


# Yup, the same columns are missing in the test data as well

# seaon_holidayed_code is the Season in which the member holiday. This we can fill probbaly by looking at the checkin date and the season code for the checkin date for that resort? Before that let us combine train and test dataset

# In[69]:


test['amount_spent_per_room_night_scaled']=0


# In[70]:


train['dataset']="train"
test['dataset']="test"


# In[71]:


missing_resort=train.loc[pd.isnull(train['season_holidayed_code']),'resort_id'].unique()


# In[72]:


test.loc[pd.isnull(test['season_holidayed_code']),'resort_id'].unique()


# In[73]:


missing_resort


# We can see that in test and train data the same set of resorts have missing season_holidayed_code

# In[74]:


missing_resort_data=train[(pd.notnull(train['season_holidayed_code'])) & (train['resort_id'].isin(missing_resort))]


# ### Let us now, use the month,year and resort id to get the missing season_holidayed_code

# In[75]:


resort_season=missing_resort_data.groupby(['resort_id','checkin_date','season_holidayed_code'])['reservation_id'].count().reset_index()


# In[76]:


resort_season.sort_values(by='checkin_date',ascending=False)


# In[77]:


def getSeasonHoliday(resort_id,date):
    rows=resort_season.loc[(resort_season['checkin_date']==date) & (resort_season['resort_id']==resort_id)]
    if rows.shape[0]>1:
        max_reservations=np.max(rows['reservation_id'])
        rows=rows[rows['reservation_id']==max_reservations]
        return rows['season_holidayed_code'].tolist()[0]
    if rows.shape[0]==1:
        return rows['season_holidayed_code'].tolist()[0]
    else:
        return 1 # Only 3 such cases found


# In[78]:


train['season_holidayed_code_new']=train.apply(lambda row:getSeasonHoliday(row['resort_id'],row['checkin_date']) if pd.isnull(row['season_holidayed_code']) else row['season_holidayed_code'],axis=1)


# In[79]:


train.head()


# In[80]:


test['season_holidayed_code_new']=test.apply(lambda row:getSeasonHoliday(row['resort_id'],row['checkin_date']) if pd.isnull(row['season_holidayed_code']) else row['season_holidayed_code'],axis=1)


# In[81]:


train.head()


# ### Are there particular members for which thew state ciode is missing?

# In[82]:


missing_state_members_train=train.loc[pd.isnull(train['state_code_residence']),'memberid'].unique().tolist()


# In[83]:


missing_state_members_test=test.loc[pd.isnull(test['state_code_residence']),'memberid'].unique().tolist()


# In[84]:


data=pd.concat([train,test])


# In[85]:


state_code_members=data.groupby(['memberid','state_code_residence'])['reservation_id'].count().reset_index()


# In[86]:


state_code_members.shape


# In[87]:


train['memberid'].nunique()


# In[88]:


142231 - 144823


# Few members have no info in the train or test dataset. Let us create missing as a seperate code for missing state residency. and this must be considered a category

# In[89]:


sorted(train['state_code_residence'].unique().tolist())


# We can see that state_residence code 17 is missing. we can replace the missing state_residence code with this number

# In[90]:


train['state_code_residence']=train['state_code_residence'].apply(lambda x:17 if pd.isnull(x) else x)


# In[91]:


test['state_code_residence']=test['state_code_residence'].apply(lambda x:17 if pd.isnull(x) else x)


# ### Feature Enginerring

# 1. Number of days, before checking in bookin was made?

# In[92]:


def diffbetweenDate(date_column1,date_column2):
    delta =  date_column1- date_column2
    return delta.days


# In[93]:


train['diff_booking_checkin_days']=train.apply(lambda row:diffbetweenDate(row['checkin_date'],row['booking_date']),axis=1)


# In[94]:


test['diff_booking_checkin_days']=test.apply(lambda row:diffbetweenDate(row['checkin_date'],row['booking_date']),
                                               axis=1)


# In[95]:


sns.kdeplot(train['diff_booking_checkin_days'])


# In[96]:


train[train['diff_booking_checkin_days']<0]


# In[97]:


test[test['diff_booking_checkin_days']<0]


# ### There are totally 14 rows, where the booking dat is much later than the check in data. In such cases, we can create a new column to indicate this anamoly and make the booking fate same as the booking date  - in the 2nd iteration

# In[98]:


train['days_of_stay']=train.apply(lambda row:diffbetweenDate(row['checkout_date'],row['checkin_date']),axis=1)


# In[99]:


test['days_of_stay']=test.apply(lambda row:diffbetweenDate(row['checkout_date'],row['checkin_date']),axis=1)


# In[100]:


train.shape


# 2. Is this a new member? - we have to get the first date the member booked using club mahindra - get all member info from train and test both

# In[101]:


data=pd.concat([train,test])


# In[102]:


data.shape


# In[103]:


member_first_booking=data.groupby(['memberid'])['booking_date'].min().reset_index().rename(columns={'booking_date':'first_booking_date'})


# In[104]:


member_first_booking


# In[105]:


train=pd.merge(train,member_first_booking,on="memberid",how='left')


# In[106]:


test=pd.merge(test,member_first_booking,on="memberid",how='left')


# In[107]:


train['is_first_booking']=train.apply(lambda row:1 if row['booking_date']==row['first_booking_date'] else 0,axis=1)
test['is_first_booking']=test.apply(lambda row:1 if row['booking_date']==row['first_booking_date'] else 0,axis=1)


# In[113]:


train.shape


# ### there are multiple booking by same member doe thw same checkin dates. Let us create a field, for if there is a booking by that member in same resort for than one

# In[114]:


member_bookings_for_same_day=train.groupby(['memberid','checkin_date'])['reservation_id'].count().reset_index().rename(columns={'reservation_id':'num_reservations_same_day'})


# In[115]:


train.shape


# In[116]:


member_bookings_for_same_day.shape


# In[117]:


train=pd.merge(train,member_bookings_for_same_day,on=['memberid','checkin_date'],how='left')


# In[118]:


train.shape


# In[119]:


member_bookings_for_same_day=test.groupby(['memberid','checkin_date'])['reservation_id'].count().reset_index().rename(columns={'reservation_id':'num_reservations_same_day'})


# In[120]:


test.shape


# In[121]:


test=pd.merge(test,member_bookings_for_same_day,on=['memberid','checkin_date'],how='left')


# In[122]:


test.shape


# ### What is the total number of visitors?

# In[123]:


train['total_visitors']=train['numberofadults'] + train['numberofchildren']


# In[124]:


test['total_visitors']=test['numberofadults'] + test['numberofchildren']


# #### What is the ratio of total_pax to total_visitors?

# In[125]:


train['visitors_pax_ratio']=train['total_visitors']/train['total_pax']
test['visitors_pax_ratio']=test['total_visitors']/test['total_pax']


# In[126]:


sns.kdeplot(train['visitors_pax_ratio'])


# ### What is the average number of visitors a memberm travels with?

# In[127]:


member_visitors_size=train.groupby(['memberid'])['total_visitors'].agg(['max', 'mean','min','median']).reset_index()
member_visitors_size=member_visitors_size.rename(columns={'max':'max_total_visitors','median':'median_total_visitors','min':'min_total_visitors','mean':'mean_total_visitors'})


# In[128]:


member_visitors_size


# In[129]:


train=pd.merge(train,member_visitors_size,on='memberid',how='left')


# In[130]:


train.shape


# In[131]:


member_visitors_size=test.groupby(['memberid'])['total_visitors'].agg(['max', 'mean','min','median']).reset_index()
member_visitors_size=member_visitors_size.rename(columns={'max':'max_total_visitors','median':'median_total_visitors','min':'min_total_visitors','mean':'mean_total_visitors'})


# In[132]:


test=pd.merge(test,member_visitors_size,on='memberid',how='left')


# In[133]:


test.shape


# ### Is the resort state same as the member state

# In[134]:


train['is_resort_state_same_member_state']=train.apply(lambda row:1 if row['state_code_resort']==row['state_code_residence'] else 0,axis=1)


# In[135]:


test['is_resort_state_same_member_state']=test.apply(lambda row:1 if row['state_code_resort']==row['state_code_residence'] else 0,axis=1)


# ### On a particular checkin date, how many reservations a resort has?

# In[136]:


data=pd.concat([train,test],sort=False)
data.shape


# In[137]:


data.head()


# In[138]:


num_reservations=data.groupby(['checkin_date'])['reservation_id'].count().reset_index().rename(columns={'reservation_id':'num_reservations_resort_on_that_day'})


# In[139]:


num_reservations.shape


# In[140]:


train=pd.merge(train,num_reservations,on='checkin_date',how='left')


# In[141]:


test=pd.merge(test,num_reservations,on='checkin_date',how='left')


# In[142]:


train.shape


# In[143]:


test.shape


# ### How many different resorts a member has booked??

# In[144]:


num_resorts_booked=data.groupby(['memberid'])['resort_id'].nunique().reset_index().rename(columns={'resort_id':'num_unique_resorts_booked_by_member'})


# In[145]:


train=pd.merge(train,num_resorts_booked,on='memberid',how='left')
test=pd.merge(test,num_resorts_booked,on='memberid',how='left')


# In[146]:


train.columns


# In[149]:


train['reservationstatusid_code'].value_counts()


# In[150]:


test['reservationstatusid_code'].value_counts()


# ## We can drop row with reservationstatusid_code= D from the datset

# In[152]:


train=train[train['reservationstatusid_code']!='D']
train.shape


# ### Number of resorts in the region?

# In[153]:


data=pd.concat([train,test])


# In[154]:


data.shape


# In[167]:


num_resorts_region=data.groupby(['resort_region_code'])['resort_id'].nunique().reset_index().rename(columns={'resort_id':
                                                                                                           'num_resorts_region'})


# In[168]:


train=pd.merge(train,num_resorts_region,on='resort_region_code',how='left')


# In[169]:


train.shape


# In[170]:


test=pd.merge(test,num_resorts_region,on='resort_region_code',how='left')


# ### Number of resorts booked on a particular checkin date?

# In[180]:


num_resorts_booked=data.groupby(['resort_region_code','checkin_date'])['resort_id'].nunique().reset_index().rename(columns={'resort_id':
                                                                                                                          'num_resorts_booked'})


# In[181]:


train=pd.merge(train,num_resorts_booked,on=['resort_region_code','checkin_date'],how='left')


# In[182]:


train.shape


# In[183]:


test=pd.merge(test,num_resorts_booked,on=['resort_region_code','checkin_date'],how='left')


# In[184]:


test.shape


# In[185]:


train.head()


# ### Percentage of resorts booked in a particular region

# In[186]:


train['percentage_resorts_booked_region']=train['num_resorts_booked']/train['num_resorts_region']
test['percentage_resorts_booked_region']=test['num_resorts_booked']/train['num_resorts_region']


# In[187]:


train.shape


# In[188]:


test.shape


# In[189]:


train['main_product_code'].unique()


# #### Days of stay vs room nights?

# In[203]:


train[train['roomnights']<0]=0
test[test['roomnights']<0]=0


# In[204]:


train['day_stay_roomnight_ratio']=train['days_of_stay']/(train['roomnights']+1)
test['day_stay_roomnight_ratio']=test['days_of_stay']/(test['roomnights']+1)


# In[205]:


sns.kdeplot(train['day_stay_roomnight_ratio'])


# In[206]:


train['day_stay_roomnight_ratio'].describe()


# In[208]:


train['booking_type_code'].value_counts()


# In[209]:


test['booking_type_code'].value_counts()


# In[210]:


train=train[train['booking_type_code']!=0]


# In[211]:


train.shape


# Let us build a model on this data and see how it performs on the leaderboard

# In[212]:


train.to_csv("Train_Preprocessed_version2.csv",index=False)
test.to_csv("Test_Preprocessed_version2.csv",index=False)


# In[215]:


import h2o
from h2o.automl import H2OAutoML


# In[214]:


h2o.cluster().shutdown()


# In[216]:


h2o.init()


# In[217]:


train=h2o.import_file("Train_Preprocessed_version2.csv")
test=h2o.import_file("Test_Preprocessed_version2.csv")


# In[218]:


x = train.columns
y = "amount_spent_per_room_night_scaled"
x.remove(y)
x.remove('dataset')
x.remove('reservation_id')
x.remove('memberid')
x.remove('season_holidayed_code')


# In[219]:


code_cols = ['channel_code','main_product_code','persontravellingid', 'resort_region_code',
       'resort_type_code', 'room_type_booked_code', 'season_holidayed_code_new', 'state_code_residence', 'state_code_resort',
             'booking_type_code'
            ]


# In[220]:


for col in code_cols:
    train[col] = train[col].asfactor()


# In[221]:


aml = H2OAutoML(max_models=30, seed=42,max_runtime_secs=7200,project_name="ClubMahindra_Version2",max_runtime_secs_per_model=400)
aml.train(x=x, y=y, training_frame=train)


# In[223]:


aml.leaderboard


# In[224]:


h2o.save_model(aml.leader, path = "Best_AutoModel_v2")


# In[225]:


for col in code_cols:
    test[col] = test[col].asfactor()


# In[226]:


preds=aml.predict(test)


# In[227]:


submission=pd.read_csv("sample_submission.csv")


# In[228]:


preds=preds.as_data_frame()


# In[229]:


submission['amount_spent_per_room_night_scaled']=preds['predict']


# In[230]:


submission.to_csv("Best_AutoML_v2.csv",index=False)


# In[ ]:




