from FairnessM import FairnessMaster
from utils import param_fairness,param_UBR,param_ROC
from utils import line_plot,cost_benefit_analysis
import pandas as pd

#import pipeline as Pipeline

data_path_list = [param_fairness["data_path"],param_UBR["data_path"],param_ROC["data_path"]]
data_type_list = ["original","UBR","ROC"]
vertical_list = [0.92,0.93,0.94]
if __name__ == '__main__':
    
    metric_summary_paths=[]
    #Generate and save fairness report as csv
    c=0
    for file_path in data_path_list:
        df_temp = pd.read_csv(file_path)
        metric = data_type_list[c]
        if metric=="UBR":
            print("Running UBR correction....")
            #df_temp[param_fairness['target']]=1-df_temp[param_fairness['target']]
            #df_temp['p_np_score_fix']=1-df_temp['p_np_score_fix']
            cba_UBR = cost_benefit_analysis(df_temp,param_fairness['target'],"p_np_score_fix")
        else:
            cba_UBR = cost_benefit_analysis(df_temp,param_fairness['target'],param_fairness['score'])
            
        cba_UBR["profit"] = cba_UBR["goods"]*80- cba_UBR["bads"]*1000
        print(cba_UBR[cba_UBR.profit==cba_UBR.profit.max()])
        
        cba_UBR["n_total"] = cba_UBR["total"]/float(cba_UBR["total"].max())
        cba_UBR["n_goods"] = cba_UBR["goods"]/float(cba_UBR["goods"].max())
        cba_UBR["n_bads"] = cba_UBR["bads"]/float(cba_UBR["bads"].max())
        cba_UBR["n_profit"] = cba_UBR["profit"]/float(cba_UBR["profit"].max())
        cba_UBR = cba_UBR[cba_UBR.threshold>0.85]
        
   
        
        df_temp = cba_UBR.sort_values("threshold",ascending=False)
        path_temp = param_fairness["out_path"]+"%s_cost_benefit_analysis_report.csv"%(metric)
        print("..saving...%s_cost_benefit_analysis_report.csv"%(metric))
        df_temp.to_csv(path_temp,index=False)
        metric_summary_paths.append(metric)
        c+=1
        
    c=0   
    #Save fairness plots
    for metric in metric_summary_paths:
        path_temp = param_fairness["out_path"]+"%s_cost_benefit_analysis_report.csv"%(metric)
        save_path = param_fairness["out_path"]+"%s_cost_benefit_analysis_plot.png"%(metric)
        df_metrices = pd.read_csv(path_temp)
        df_metrices=df_metrices[df_metrices.threshold>0.85]
        line_plot(df_metrices,'Score','Normalized values','Identifying maximum profit point- (%s)'%(metric),['threshold'],["threshold"]+["n_total","n_goods","n_bads","n_profit"],'threshold','value',vertical=vertical_list[c],save_path=save_path)
        c+=1
        
        
  
