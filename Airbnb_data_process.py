import pandas as pd
import numpy as np

# load dataset
train_users = pd.read_csv('train_users_2.csv')
test_users = pd.read_csv('test_users.csv')
# join data
users = pd.concat((train_users, test_users), axis=0, ignore_index=True)
# Set ID as index
users = users.set_index('id')
train_users = train_users.set_index('id')
test_users = test_users.set_index('id')

 # Drop columns
drop_list = ['date_account_created','date_first_booking','timestamp_first_active']
users.drop(drop_list, axis=1, inplace=True)
#clean data
user_with_year_age_mask = users['age'] > 1000
users.loc[user_with_year_age_mask, 'age'] = 2015 - users.loc[user_with_year_age_mask, 'age']
users.loc[(users['age'] > 100) | (users['age'] < 18), 'age'] = -1
users['age'].fillna(-1, inplace=True)

# categorical_feature
categorical_features = ['gender', 'signup_method', 'signup_flow',
'language','affiliate_channel', 'affiliate_provider', 'first_affiliate_tracked',
'signup_app', 'first_device_type', 'first_browser']
# one_hot_encode
#String to append DataFrame column names.
def one_hot_encode(data,categorical_features):
	for feature in categorical_features:
		data_dummy=pd.get_dummies(data[feature], prefix=feature)
		data.drop([feature], axis=1, inplace=True)
		data = pd.concat((data, data_dummy), axis=1)
	return data

users = one_hot_encode(users, categorical_features)

 # Split into train and test users
train_users = users.loc[train_users.index]
test_users = users.loc[test_users.index]

train_users.to_csv('train_users_processed.csv' )
test_users.to_csv('test_users_processed.csv' )