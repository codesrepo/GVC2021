from FairnessM import FairnessMaster
from utils import param_fairness,param_ROC
from utils import line_plot
import pandas as pd

#import pipeline as Pipeline

if __name__ == '__main__':
    final = pd.read_csv(param_ROC["data_path"])
    metric_summary_paths = []
    
    #Generate and save fairness report as csv
    for metric in param_fairness['protected_attribute_list']:     
        new_obj = FairnessMaster(final,param_fairness['target'],param_fairness['score'],metric,param_fairness['privileged_group'],param_fairness['base_trheshold'],param_fairness['threshold_bins'])
        df_metrices = new_obj.calculate_fairness_metrices()
        df_temp = df_metrices[["threshold"]+param_fairness['track_metrices']].sort_values("threshold",ascending=False)
        path_temp = param_ROC["out_path"]+"%s_ROC_fairness_report.csv"%(metric)
        print("..saving...%s_ROC_fairness_report.csv"%(metric))
        df_temp.to_csv(path_temp,index=False)
        metric_summary_paths.append(metric)
        
        
    #Save fairness plots
    for metric in metric_summary_paths:
        path_temp = param_ROC["out_path"]+"%s_ROC_fairness_report.csv"%(metric)
        save_path = param_ROC["out_path"]+"%s_ROC_fairness_plot.png"%(metric)
        df_metrices = pd.read_csv(path_temp)
        df_metrices=df_metrices[df_metrices.threshold>0.85]
        line_plot(df_metrices,'Score','Fairness values','Fairness report- Base model (%s)'%(metric),['threshold'],["threshold"]+param_fairness['track_metrices'],'threshold','value',vertical=0.93,save_path=save_path)
        
        
  
