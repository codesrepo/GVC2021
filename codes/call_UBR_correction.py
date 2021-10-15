# Input parameter section
from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from utils import param_data_preprocessing,param_UBR
import xgboost as xgb
import pandas as pd



HC_train =  pd.read_csv(param_data_preprocessing["ntc_train_path"]+"HC_train.csv")
HC_test  =  pd.read_csv(param_data_preprocessing["ntc_test_path"]+"HC_test.csv")

feature_table = pd.read_csv(param_data_preprocessing["ntc_test_path"]+"HC_feature_imp.csv")


protected_features1 = ['DAYS_BIRTH','CODE_GENDER','NAME_FAMILY_STATUS']


#fit the model - protect variables
model_p = xgb.XGBClassifier(num_class  =2, objective='multi:softprob', max_depth = 2, min_child_weight = 0.3, learning_rate = 0.1, subsample = 0.8, seed =123)
model_p .fit(HC_train[protected_features1],HC_train['TARGET'])

HC_train['p_score'] =  model_p.predict_proba(HC_train[protected_features1])[:,1]
HC_test['p_score'] =  model_p.predict_proba(HC_test[protected_features1])[:,1]

#fit the model - protect variables +  non protect variables
keep_feature  = list(feature_table[feature_table['impotance'] >= 0.01]['var'])
model = xgb.XGBClassifier(colsample_bytree= 0.6, gamma= 30, max_depth= 4, min_child_weight=14, n_estimators=441, subsample= 0.7, seed=123, learning_rate=0.1, n_jobs= -1)
model.fit(HC_train[keep_feature+ ['p_score']],HC_train['TARGET'])

HC_train['p_np_score'] =  model.predict_proba(HC_train[keep_feature+ ['p_score']])[:,1]
HC_test['p_np_score'] =  model.predict_proba(HC_test[keep_feature+ ['p_score']])[:,1]

#print('original train roc is: ' ,roc_auc_score(HC_train['TARGET'], HC_train['p_np_score']))
print('original test roc is: ' , roc_auc_score(HC_test['TARGET'], HC_test['p_np_score']))

#predit the score by set protect score = 0
#HC_train['p_score'] = 0
HC_test['p_score'] =  0

#HC_train['p_np_score_fix'] =  model.predict_proba(HC_train[keep_feature+ ['p_score']])[:,1]
HC_test['p_np_score_fix'] =   model.predict_proba(HC_test[keep_feature+ ['p_score']])[:,1]
HC_test.to_csv(param_UBR["out_path"]+"UBR_corrected_data.csv",index=False)

#print('correct train roc is: ' ,roc_auc_score(HC_train['TARGET'], HC_train['p_np_score_fix']))
print('correct test roc is: ' , roc_auc_score(HC_test['TARGET'], HC_test['p_np_score_fix']))
