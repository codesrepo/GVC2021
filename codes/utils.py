from sklearn.metrics import accuracy_score,roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

param_data_preprocessing = {
        
        "train_path":"/home/coder/GVC2021/submission/data/application_train.csv",
        "bureau_path":"/home/coder/GVC2021/submission/data/bureau.csv",
        "out_path":"/home/coder/GVC2021/submission/output/",
        "ntc_train_path":"/home/coder/GVC2021/submission/output/",
        "ntc_test_path":"/home/coder/GVC2021/submission/output/",
        
        
        
        
        }

param_fairness = {
        
        "data_path":"/home/coder/GVC2021/submission/output/final_outcome.csv",
        "out_path":"/home/coder/GVC2021/submission/output/",
        "target":"TARGET",
        "score":"prediction_prob",
        "protected_attribute_list":['AGE',"CODE_GENDER","MARITAL_STATUS"],
        "privileged_group":1,
        "base_trheshold":0.92,
        "threshold_bins":100,
        "track_metrices":["equal_opportunity_difference","average_abs_odds_difference","disparate_impact"]     
        
       
        }

param_ROC = {
        
        "data_path":"/home/coder/GVC2021/submission/output/ROC_corrected_data_1.csv",
        "out_path":"/home/coder/GVC2021/submission/output/",
        "target":"TARGET",
        "score":"prediction_prob",
        "apply_ROC_attribute":'AGE',
        "privileged_group":1,
        "base_trheshold":0.91,
        "threshold_bins":100,
        "track_metrices":["equal_opportunity_difference","average_abs_odds_difference","disparate_impact"],
        "low_ROC_margin":0.01,
        "high_ROC_margin":0.05,
        "num_ROC_margin":100,
        "metric_lb":0.85,
        "metric_ub":0.95,
        "metric_name":"average_odds_difference"
        
        
        
        
        }


param_UBR = {
        
        
        "data_path":"C:/Veritas/Submission/APIX/Veritas/data/df_UBR.csv",
        "out_path":"C:/Veritas/Submission/APIX/Veritas/output/UBR/",
        "score":"p_np_score_fix"
        }


def get_bad_rate(df,target,score,threshold):
  bad_rate = 1-df[target][df[score]>=threshold].sum()/float(len(df[target][df[score]>=threshold]))
  acceptance_rate = (df[target][df[score]>=threshold].count()/float(len(df)))
  return("Bad rate = %s, Acceptance rate = %s, Gini=%s"%(bad_rate,acceptance_rate,2*roc_auc_score(df[target],df[score])-1))

def cost_benefit_analysis(df,target,score):
  total=[]
  goods = []
  bads = []
  thresh = []
  
  for i in range(0,100):
      try:
        threshold = i*0.01
        temp_total = df[target][df[score]>=threshold].count()
        temp_goods = df[target][df[score]>=threshold].sum()
        total.append(temp_total)
        goods.append(temp_goods)
        bads.append(temp_total-temp_goods)
        thresh.append(np.round(threshold,2))
      except:
        print(thresh)
  return pd.DataFrame({"threshold":thresh,"total":total,"goods":goods,"bads":bads})
        
def line_plot(df,xlabel,ylabel,title,x_list,y_list,x,y,vertical=None,save_path=None):
  plt.figure(figsize=(16, 7))
  sns.set(font_scale=1.8)
  sns.lineplot(x=x, y=y, hue='variable', linewidth = 4,
               data=pd.melt(df[y_list], x_list))
  
  if vertical!=None:
    plt.axvline(vertical, color='red',dashes=(2,2))
  
  plt.title(title)
  plt.xlabel(xlabel)
  plt.ylabel(ylabel)
  plt.grid()
  if save_path!=None:
    plt.savefig(save_path)
  plt.show()