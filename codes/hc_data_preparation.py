from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import LabelEncoder
from scipy.stats import chi2_contingency
from scipy.stats import pointbiserialr
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import xgboost as xgb
import seaborn as sns
import pandas as pd
import numpy as np

from utils import param_data_preprocessing

train = pd.read_csv(param_data_preprocessing["train_path"],nrows=10000)
bureau = pd.read_csv(param_data_preprocessing["bureau_path"],nrows=10000)
bureau.drop_duplicates("SK_ID_CURR",keep="last",inplace=True)
bureau["bureau_presence"]=1

print(train.shape)
temp = train.merge(bureau[["SK_ID_CURR","bureau_presence"]],on="SK_ID_CURR",how="left")
train = temp[temp.bureau_presence!=1]
print(train.shape)

train['DAYS_BIRTH']  =  (train['DAYS_BIRTH']*(-1)/365).astype(int)
train['DAYS_BIRTH']

drop_list = ['FLAG_DOCUMENT_2','FLAG_DOCUMENT_4','FLAG_DOCUMENT_5','FLAG_DOCUMENT_7','FLAG_DOCUMENT_9','FLAG_DOCUMENT_10',
            'FLAG_DOCUMENT_11','FLAG_DOCUMENT_12','FLAG_DOCUMENT_13','FLAG_DOCUMENT_14','FLAG_DOCUMENT_15','FLAG_DOCUMENT_17',
            'FLAG_DOCUMENT_19','FLAG_DOCUMENT_20','FLAG_DOCUMENT_21','AMT_REQ_CREDIT_BUREAU_HOUR','AMT_REQ_CREDIT_BUREAU_DAY','AMT_REQ_CREDIT_BUREAU_WEEK',
             'AMT_REQ_CREDIT_BUREAU_MON','AMT_REQ_CREDIT_BUREAU_QRT','AMT_REQ_CREDIT_BUREAU_YEAR','bureau_presence','WEEKDAY_APPR_PROCESS_START','HOUR_APPR_PROCESS_START']

cat_features = ['NAME_CONTRACT_TYPE','CODE_GENDER','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE',
               'NAME_INCOME_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
               'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','NAME_EDUCATION_TYPE','NAME_FAMILY_STATUS']

ID = ["SK_ID_CURR"]
TARGET  = ["TARGET"]

protected_features1 = ['DAYS_BIRTH','CODE_GENDER','NAME_FAMILY_STATUS']

train.drop(drop_list, axis = 1,inplace=True)
train = pd.DataFrame(train.reset_index(drop=True))

numerical = list(set(train.columns.tolist())-set(cat_features+ID+TARGET))

for i in numerical:
  train[i]=train[i].fillna(-999)
  
label_encoder = LabelEncoder()
X_orig = train.copy()
X = np.zeros((len(train['CODE_GENDER']),1))
for i, name in enumerate(cat_features):
  
    train[name] = train[name].astype(str)
    x = label_encoder.fit_transform(train[name]).reshape(-1,1)
    X = np.hstack((X,x))

X = pd.DataFrame(X).drop([0],axis=1)
X.columns = cat_features

for i, name in enumerate(numerical):
    if name == 'AMT_INCOME_TOTAL':
        X = pd.concat([X,np.log(train[name])],axis=1)
    else:
        X = pd.concat([X,train[name]],axis=1)
        
X['AGE'] = train['DAYS_BIRTH'].apply(lambda x: 0 if x<=65 else 1 )
X['CODE_GENDER'] = train['CODE_GENDER'].apply(lambda x: 0 if x== 'F' else 1 )
X['MARITAL_STATUS'] = train['NAME_FAMILY_STATUS'].apply(lambda x: 0 if x== 'Single / not married' else 1 )

data = pd.concat([X,train['TARGET']],axis=1)
Y = data.iloc[:,-1:]

X['AGE_Raw'] = train['DAYS_BIRTH']
# X[X['AGE_Raw'] == 60]['DAYS_BIRTH']

indices=range(len(X))
X_train, X_test, y_train, y_test,indices_train,indices_test = train_test_split(X,Y,indices, test_size = 0.5, stratify=Y,
                                                    random_state=123)

print(X_train.shape)
print(X_test.shape)


X_test_raw = X_test.copy()

scaler = StandardScaler()
scaled_numfeats_train = pd.DataFrame(scaler.fit_transform(X_train[numerical]), 
                                     columns=numerical, index= X_train.index)
for col in numerical:
    X_train[col] = scaled_numfeats_train[col]
    
scaled_numfeats_test = pd.DataFrame(scaler.transform(X_test[numerical]),
                                    columns=numerical, index= X_test.index)

for col in numerical:
    X_test[col] = scaled_numfeats_test[col]
    
    
cat_features_model = ['NAME_CONTRACT_TYPE','FLAG_OWN_CAR','FLAG_OWN_REALTY','CNT_CHILDREN','NAME_TYPE_SUITE',
               'NAME_INCOME_TYPE','NAME_HOUSING_TYPE','OCCUPATION_TYPE','ORGANIZATION_TYPE','FONDKAPREMONT_MODE','HOUSETYPE_MODE',
               'WALLSMATERIAL_MODE','EMERGENCYSTATE_MODE','NAME_EDUCATION_TYPE']


numerical_model =['FLAG_EMP_PHONE',
 'NONLIVINGAREA_AVG',
 'ELEVATORS_MODE',
 'FLAG_DOCUMENT_8',
 'YEARS_BUILD_MODE',
 'FLOORSMAX_MODE',
 'LIVINGAPARTMENTS_MEDI',
 'FLAG_DOCUMENT_3',
 'FLOORSMIN_MODE',
 'DEF_30_CNT_SOCIAL_CIRCLE',
 'EXT_SOURCE_3',
 'CNT_FAM_MEMBERS',
 'REG_REGION_NOT_WORK_REGION',
 'NONLIVINGAPARTMENTS_MEDI',
 'DAYS_LAST_PHONE_CHANGE',
 'OBS_30_CNT_SOCIAL_CIRCLE',
 'FLAG_PHONE',
 'AMT_INCOME_TOTAL',
 'FLAG_DOCUMENT_18',
 'EXT_SOURCE_2',
 'FLAG_MOBIL',
 'DEF_60_CNT_SOCIAL_CIRCLE',
 'COMMONAREA_MODE',
 'NONLIVINGAREA_MODE',
 'YEARS_BUILD_AVG',
 'NONLIVINGAPARTMENTS_AVG',
 'REGION_POPULATION_RELATIVE',
 'LIVINGAPARTMENTS_AVG',
 'REGION_RATING_CLIENT_W_CITY',
 'APARTMENTS_MODE',
 'BASEMENTAREA_MODE',
 'ELEVATORS_MEDI',
 'FLOORSMAX_MEDI',
 'EXT_SOURCE_1',
 'DAYS_EMPLOYED',
 'BASEMENTAREA_AVG',
 'YEARS_BEGINEXPLUATATION_MEDI',
 'OWN_CAR_AGE',
 'APARTMENTS_MEDI',
 'LANDAREA_MODE',
 'YEARS_BEGINEXPLUATATION_AVG',
 'REGION_RATING_CLIENT',
 'APARTMENTS_AVG',
 'AMT_ANNUITY',
 'FLAG_WORK_PHONE',
 'LANDAREA_AVG',
 'OBS_60_CNT_SOCIAL_CIRCLE',
 'ENTRANCES_AVG',
 'YEARS_BEGINEXPLUATATION_MODE',
 'NONLIVINGAPARTMENTS_MODE',
 'LIVE_REGION_NOT_WORK_REGION',
 'YEARS_BUILD_MEDI',
 'REG_CITY_NOT_LIVE_CITY',
 'AMT_CREDIT',
 'LIVINGAREA_MEDI',
 'LIVINGAPARTMENTS_MODE',
 'FLAG_DOCUMENT_16',
 'FLAG_EMAIL',
 'REG_REGION_NOT_LIVE_REGION',
 'LANDAREA_MEDI',
 'REG_CITY_NOT_WORK_CITY',
 'LIVINGAREA_AVG',
 'BASEMENTAREA_MEDI',
 'FLOORSMIN_MEDI',
 'FLOORSMIN_AVG',
 'COMMONAREA_AVG',
 'NONLIVINGAREA_MEDI',
 'ENTRANCES_MODE',
 'FLOORSMAX_AVG',
 'DAYS_ID_PUBLISH',
 'AMT_GOODS_PRICE',
 'FLAG_DOCUMENT_6',
 'DAYS_REGISTRATION',
 'LIVINGAREA_MODE',
 'ENTRANCES_MEDI',
 'COMMONAREA_MEDI',
 'LIVE_CITY_NOT_WORK_CITY',
 'TOTALAREA_MODE',
 'FLAG_CONT_MOBILE',
 'ELEVATORS_AVG']

#for protect variables modeling
protected_features1 = ['DAYS_BIRTH','CODE_GENDER','NAME_FAMILY_STATUS']

#for fairness evaluation 
protected_features2 = ['AGE','CODE_GENDER','MARITAL_STATUS']

protected_features = list(set(protected_features1 + protected_features2))

non_protected_features = cat_features_model+numerical_model
print(non_protected_features)

model = xgb.XGBClassifier(num_class  =2, objective='multi:softprob', max_depth = 2, min_child_weight = 0.3, learning_rate = 0.1, subsample = 0.8)
model.fit(X_train[non_protected_features],y_train)
predicted_proba_train = model.predict_proba(X_train[non_protected_features])[:,1]
print('train roc is: ' ,roc_auc_score(y_train, predicted_proba_train))
#XGB Accuracy:  0.8442071785552022

predicted_proba_test = model.predict_proba(X_test[non_protected_features])[:,1]
print('test roc is: ' , roc_auc_score(y_test, predicted_proba_test))


feature_table = pd.DataFrame({'var':list(X_train[non_protected_features].columns), 'impotance': model.feature_importances_ })
feature_table = feature_table.sort_values(by =['impotance'],ascending = [False])
keep_feature  = list(feature_table[feature_table['impotance'] >= 0.01]['var'])


HC_train = X_train[keep_feature + protected_features]
HC_test  = X_test[keep_feature  + protected_features]
HC_train['TARGET'] = y_train['TARGET']
HC_test['TARGET']   = y_test['TARGET']

HC_train.to_csv(param_data_preprocessing["out_path"]+"HC_train.csv",index=False)
HC_test.to_csv(param_data_preprocessing["out_path"]+"HC_test.csv",index=False)
feature_table.to_csv(param_data_preprocessing["out_path"]+"HC_feature_imp.csv",index=False)
