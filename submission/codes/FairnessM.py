from sklearn.metrics import confusion_matrix
import pandas as pd
import numpy as np

class FairnessMaster(object):
  def __init__(self,df,target,prob,protected_var,privileged_group,base_threshold=1,n_bins=20):
    self.df = df
    self.target = target
    self.protected_var = protected_var
    self.privileged_group = privileged_group
    self.prob = prob
    self.base_threshold = base_threshold
    self.n_bins = n_bins
  
  
  def get_confusion_matrix(self,target,prediciton):
    tn, fp, fn, tp  = confusion_matrix(target, prediciton).ravel()
    return tn, fp, fn, tp
  
  def weird_division(self,n, d):
    return np.round(n / d,3) if d else 0
  
  def calculate_fairness_metrices(self):
    
    df_privileged = self.df[self.df[self.protected_var]==self.privileged_group]
    df_underprivileged = self.df[self.df[self.protected_var]!=self.privileged_group]
    prob=self.prob

    thresh_list = []
    dp_p = []
    eo_p = []
    pp_p = []
    di_tn_p = []

    dp_up = []
    eo_up = []
    pp_up = []
    di_tn_up = []
    
    absolute_equal_opportunity_difference = []
    equal_opportunity_difference = []
    average_abs_odds_difference = []
    average_odds_difference=[]
    statistical_parity_difference=[]
    
    #'absolute_equal_opportunity_difference'
    #equal_opportunity_difference
    
    
    #average_abs_odds_difference
    #average_odds_difference
    
    #statistical_parity_difference
    
    
    for i in range(0,100):
      try:
        thresh = i*0.01
        thresh_list.append(np.round(thresh,2))
        temp_p = df_privileged.copy()
        temp_up = df_underprivileged.copy()

        temp_p["prediction"] = temp_p[self.prob].apply(lambda x:1 if x>=thresh else 0 )
        temp_up["prediction"] = temp_up[self.prob].apply(lambda x:1 if x>=thresh else 0 )

        tn, fp, fn, tp = self.get_confusion_matrix(temp_p[self.target],temp_p["prediction"])
        dp_p.append(self.weird_division((fp+tp),(tn+ fp+ fn+ tp)))
        eo_p.append(self.weird_division((tp),( fn+ tp)))
        pp_p.append(self.weird_division(tp,fp+  tp))
        
        di_tn_p.append(self.weird_division((fp),( fp+ tn)))
        

        tn, fp, fn, tp = self.get_confusion_matrix(temp_up[self.target],temp_up["prediction"])
        dp_up.append(self.weird_division((fp+tp),(tn+ fp+ fn+ tp)))
        eo_up.append(self.weird_division((tp),( fn+ tp)))
        pp_up.append(self.weird_division(tp, fp+  tp))
        di_tn_up.append(self.weird_division((fp),( fp+ tn)))
      except:
        print(thresh)
        
    df = pd.DataFrame({"threshold":thresh_list,"di_p":dp_p,"eo_p":eo_p,"pp_p":pp_p,"di_up":dp_up,"eo_up":eo_up,"pp_up":pp_up,"di_tn_p":di_tn_p,"di_tn_up":di_tn_up})
    
    df["disparate_impact"] = df["di_up"]/df["di_p"]
    df["absolute_equal_opportunity_ratio"] = df["eo_up"]/df["eo_p"]
    df["pp_up_by_p"] = df["pp_up"]/df["pp_p"]  
    df["average_absolute_odds_ratio"] = ((df["eo_up"]+df["di_tn_up"])/(df["eo_p"]+df["di_tn_p"]))
    df["statistical_parity_ratio"] = df["di_up"]/df["di_p"]
    
    df["statistical_parity_difference"] = 1-(df["di_p"]-df["di_up"])
    df["equal_opportunity_difference"] = 1-(df["eo_p"]-df["eo_up"])
    
    df["absolute_equal_opportunity_difference"] =  1-abs(df["eo_p"]-df["eo_up"])
    df["average_odds_difference"] = 1-((((df["eo_p"]+df["di_tn_p"])-(df["eo_up"]+df["di_tn_up"]))*0.5))
    df["average_abs_odds_difference"] = 1-(abs(((df["eo_p"]+df["di_tn_p"])-(df["eo_up"]+df["di_tn_up"]))*0.5))
    

    return df 
  
 
  
  def calculate_harms_and_benefits(self,df_up):
    
      threshold_list = [np.round(self.base_threshold - i*0.01,2) for i in range(0,self.n_bins)]
      threshold_list = [i  for i in threshold_list if i>0]
      benefits = []
      harms = []
      temp_up = df_up.copy()
      temp_up["prediction"] = temp_up[self.prob].apply(lambda x:1 if x>=self.base_threshold else 0 )
      tn_base, fp_base, fn_base, tp_base = self.get_confusion_matrix(temp_up[self.target],temp_up["prediction"])
      
      for i in threshold_list:
        temp_up = df_up.copy()
        temp_up["prediction"] = temp_up[self.prob].apply(lambda x:1 if x>=i else 0 )
        tn, fp, fn, tp = self.get_confusion_matrix(temp_up[self.target],temp_up["prediction"])
        benefits.append(tp-tp_base)
        harms.append(fp-fp_base)
        
      df = pd.DataFrame({"threshold":threshold_list,"benefit":benefits,"harm":harms})
      return df  
    
  def apply_ROC(self,df,low_ROC_margin,high_ROC_margin,num_ROC_margin,metric_lb,metric_ub,metric_name):
      metric_val_old = -1
      df_privileged = self.df[self.df[self.protected_var]==self.privileged_group]
      df_underprivileged = self.df[self.df[self.protected_var]!=self.privileged_group]
      for ROC_margin in np.linspace(low_ROC_margin,high_ROC_margin, num_ROC_margin):     
        interval_low = self.base_threshold-ROC_margin
        interval_high = self.base_threshold+ROC_margin
        temp_p =df_privileged.copy()
        temp_up =df_underprivileged.copy()       
        temp_p["prediction"] = temp_p[self.prob].apply(lambda x:x if x>=interval_high or x<=interval_low else 0.01)
        temp_up["prediction"] = temp_up[self.prob].apply(lambda x:0.99 if x>=interval_low and x<=interval_high  else x ) 
        
        
        temp_p["prediction"] = temp_p["prediction"].apply(lambda x:1 if x>=self.base_threshold else 0 )
        temp_up["prediction"] = temp_up["prediction"].apply(lambda x:1 if x>=self.base_threshold else 0 )
        
        #print(temp_p["prediction"].sum(),temp_up["prediction"].sum(),"Prediction-SUM")

        tn, fp, fn, tp = self.get_confusion_matrix(temp_p[self.target],temp_p["prediction"])
        
        dp_p=(self.weird_division((fp+tp),(tn+ fp+ fn+ tp)))
        eo_p=(self.weird_division((tp),( fn+ tp)))
        pp_p=(self.weird_division(tp,fp+  tp))        
        di_tn_p=(self.weird_division((fp),( fp+ tn)))
        

        

        tn, fp, fn, tp = self.get_confusion_matrix(temp_up[self.target],temp_up["prediction"])
        dp_up=(self.weird_division((fp+tp),(tn+ fp+ fn+ tp)))
        eo_up=(self.weird_division((tp),( fn+ tp)))
        pp_up=(self.weird_division(tp, fp+  tp))
        di_tn_up=(self.weird_division((fp),( fp+ tn)))
        
        
        
        equal_opportunity_difference = 1-(eo_p-eo_up)    
        average_odds_difference = 1-((((eo_p+di_tn_p)-(eo_up+di_tn_up))*0.5))
        statistical_parity_difference = 1-(dp_p-dp_up)
        
        if metric_name=="equal_opportunity_difference":
            metric_val = equal_opportunity_difference
        elif metric_name=="average_odds_difference":
            metric_val = average_odds_difference
        elif metric_name=="statistical_parity_difference":
            metric_val = statistical_parity_difference
        else:
          raise ValueError("Please select correct metric (equal_opportunity_difference or average_odds_difference or statistical_parity_difference )")
        
        #print(ROC_margin,metric_val_old,metric_val,equal_opportunity_difference)
        
        if metric_val_old<metric_val:
          metric_val_old = metric_val
          
        if metric_val_old >=metric_lb:
          return ROC_margin,interval_low,interval_high
      raise ValueError("Please change the parameter range!!!!")
      
  def _get_corrected_value(self,x,interval_low,interval_high):
    #print(x[self.prob],interval_high)
    if x[self.protected_var]==self.privileged_group:
      if  x[self.prob]>=interval_high or x[self.prob]<=interval_low:
        return x[self.prob]
      else:
        return 0.01
      
    if x[self.protected_var]!=self.privileged_group:
      if  x[self.prob]>=interval_low and x[self.prob]<=interval_high:
        return 0.99
      else:
        return x[self.prob]
      
      
  def apply_ROC_correction(self,df,interval_low,interval_high):
    df["prediction_ROC"] = df.apply(lambda x:self._get_corrected_value(x,interval_low,interval_high),axis=1)
    return df
           
  