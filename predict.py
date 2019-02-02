import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder
from xgboost.sklearn import XGBClassifier



def generate_submission(y_pred, test_users_ids, label_encoder):
    
    ids = []
    cts = []
    for i, user in enumerate(test_users_ids):
        idx = user
        ids += [idx] * 5
        sorted_countries = np.argsort(y_pred[i])[::-1]
        cts += label_encoder.inverse_transform(sorted_countries)[:5].tolist()

    id_stacks = np.column_stack((ids, cts))
    submission = pd.DataFrame(id_stacks, columns=['id', 'country'])
    name = 'sub.csv'



train_users = pd.read_csv('train_users_processed.cs')
test_users = pd.read_csv(t'test_users_processed.csv')
y_train = train_users['country_destination']
train_users.drop(['country_destination', 'id'], axis=1, inplace=True)
train_users = train_users.fillna(-1)
x_train = train_users.values
label_encoder = LabelEncoder()
encoded_y_train = label_encoder.fit_transform(y_train)

test_users_ids = test_users['id']
test_users.drop('id', axis=1, inplace=True)
test_users = test_users.fillna(-1)
x_test = test_users.values

# XGBClassifier 
clf = XGBClassifier(
    max_depth=8,
    learning_rate=0.08,
    n_estimators=80,
    objective="rank:pairwise",
    gamma=0,
    min_child_weight=1,
    max_delta_step=0,
    subsample=1,
    colsample_bytree=1,
    colsample_bylevel=1,
    reg_alpha=0,
    reg_lambda=1,
    scale_pos_weight=1,
    base_score=0.5,
    missing=None,
    silent=True,
    nthread=-1,
    seed=1
)

#
clf=RandomForestClassifier(n_estimators=300,random_state=1)
clf.fit(x_train, encoded_y_train)
y_pred = clf.predict_proba(x_test)
generate_submission(y_pred, test_users_ids, label_encoder)



















