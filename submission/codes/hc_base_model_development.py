from sklearn.metrics import accuracy_score,roc_auc_score,confusion_matrix
from utils import param_data_preprocessing
import xgboost as xgb
import pandas as pd


ID = ["index"]
TARGET  = ["TARGET"]
df_train = pd.read_csv(param_data_preprocessing["ntc_train_path"]+"HC_train.csv")
df_test = pd.read_csv(param_data_preprocessing["ntc_test_path"]+"HC_test.csv")

df_train["TARGET"] = df_train["TARGET"].apply(lambda x: 0 if x==1 else 1 )
df_test["TARGET"] = df_test["TARGET"].apply(lambda x: 0 if x==1 else 1 )

df_train["AGE"] = df_train["DAYS_BIRTH"].apply(lambda x: 1 if x>1.5 else 0 )
df_test["AGE"] = df_test["DAYS_BIRTH"].apply(lambda x: 1 if x>1.5 else 0 )


df_train = df_train.reset_index()
df_test = df_test.reset_index()


trn_columns = list(df_train.columns)
drop_columns = ["p_score_T","p_np_score_T","p_np_score_fix_T"]
protected_features1 = ['DAYS_BIRTH','CODE_GENDER','NAME_FAMILY_STATUS']
protected_features2 = ['AGE','CODE_GENDER','MARITAL_STATUS']
imp_cols = ['DAYS_EMPLOYED','ORGANIZATION_TYPE','DAYS_LAST_PHONE_CHANGE','FLAG_DOCUMENT_3','NAME_EDUCATION_TYPE','LIVINGAREA_MEDI']
protected_features = protected_features1+protected_features2
non_protected_features = list(set(df_train.columns.tolist())-set(ID+TARGET+protected_features+drop_columns))


df_train = pd.DataFrame(df_train.reset_index(drop=True))
df_test = pd.DataFrame(df_test.reset_index(drop=True))


X_test_raw = df_test.copy()
X_train = df_train[non_protected_features]
X_test = df_test[non_protected_features]
y_train = df_train[TARGET[0]]
y_test = df_test[TARGET[0]]

print(X_train.shape)
print(X_test.shape)

params = {"colsample_bytree":0.4,
          "gamma":13.196023151279942,
          "min_child_weight":3.2841380122371557,            
          "n_estimators":156,
          "seed":123,
          "subsample":0.4
         }

model = xgb.XGBClassifier(params=params,num_class  =2, objective='multi:softprob')
model.fit(X_train[non_protected_features],y_train)
predict = model.predict(X_test[non_protected_features])
predict_proba = model.predict_proba(X_test[non_protected_features])
acc = accuracy_score(y_test, predict)
roc_auc  = roc_auc_score(y_test, predict_proba[:,1])
print('XGB Accuracy:%s, ROC:%s '%(acc,roc_auc))


X_test = X_test[non_protected_features]
cols_ = X_test.columns
cols_ = [x+"_T" for x in cols_]
X_test.columns=cols_

final = pd.concat([X_test_raw ,X_test,pd.DataFrame(predict_proba[:,1],columns = ['prediction_prob']),pd.DataFrame(predict,columns = ['prediction'])],axis=1)
final.to_csv(param_data_preprocessing["out_path"]+"/final_outcome.csv",index=False)

#https://adb-5589335273424606.6.azuredatabricks.net/files/shared_uploads/c56708a@ascendplatform.experian.com.sg/scenario_1_hc_v11122021.csv


tn, fp, fn, tp  = confusion_matrix(final["TARGET"], final["prediction"]).ravel()
print("Confusion matrix before adjustments - tn=%s, fp=%s, fn=%s, tp=%s"%(tn, fp, fn, tp))