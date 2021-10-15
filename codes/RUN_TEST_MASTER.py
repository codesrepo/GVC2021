
from utils import param_TEST_MASTER,param_fairness,line_plot,cost_benefit_analysis
from sklearn.metrics import roc_auc_score
from FairnessM import FairnessMaster
import pandas as pd
import numpy as np

if __name__ == '__main__':
    test_master = pd.read_csv(param_TEST_MASTER["test_master_data_path"])
    output_variable_list = ['prediction_prob','prediction_ROC','p_np_score_fix','score_gs']
    correction_type = ["BASE","ROC","UBR","GS"]
    vertical_list =  [0.919,0.926,0.94,0.945]
    c=0
    for i in output_variable_list:
        print("AUC on %s is %s"%(correction_type[c],roc_auc_score(test_master['TARGET'],test_master[i])))
        c+=1
    c=0   
    metric_summary_paths=[]
    for i in output_variable_list:
        for metric in param_fairness['protected_attribute_list']:  
            print("Generating report for - %s model on %s metric"%(correction_type[c],metric))   
            new_obj = FairnessMaster(test_master,param_fairness['target'],i,metric,param_fairness['privileged_group'],param_fairness['base_trheshold'],param_fairness['threshold_bins'])
            df_metrices = new_obj.calculate_fairness_metrices()
            df_temp = df_metrices[["threshold"]+param_fairness['track_metrices']].sort_values("threshold",ascending=False)
            path_temp = param_TEST_MASTER["fairness_out_path"]+"%s_%s_fairness_report.csv"%(metric,correction_type[c])
            print("..saving...%s_%s_fairness_report.csv"%(metric,correction_type[c]))
            df_temp.to_csv(path_temp,index=False)
            save_path = param_TEST_MASTER["fairness_out_path"]+"%s_%s_fairness_report.png"%(metric,correction_type[c])
            df_metrices=df_metrices[df_metrices.threshold>0.85]
            line_plot(df_metrices,'Score','Fairness values','Fairness report- %s-(%s)'%(correction_type[c],metric),['threshold'],["threshold"]+param_fairness['track_metrices'],'threshold','value',vertical=vertical_list[c],save_path=save_path)
            #print("AUC on %s is %s"%(correction_type[c],roc_auc_score(test_master['TARGET'],test_master[i])))
        
    
        cba_UBR = cost_benefit_analysis(test_master,param_fairness['target'],i)
        cba_UBR["profit"] = cba_UBR["goods"]*80- cba_UBR["bads"]*1000
        print("Optima..%s_%s_..."%(metric,correction_type[c]))
        print(cba_UBR[cba_UBR.profit==cba_UBR.profit.max()])        
        cba_UBR["n_total"] = cba_UBR["total"]/float(cba_UBR["total"].max())
        cba_UBR["n_goods"] = cba_UBR["goods"]/float(cba_UBR["goods"].max())
        cba_UBR["n_bads"] = cba_UBR["bads"]/float(cba_UBR["bads"].max())
        cba_UBR["n_profit"] = cba_UBR["profit"]/float(cba_UBR["profit"].max())
        cba_UBR = cba_UBR[cba_UBR.threshold>0.85]  
        df_temp = cba_UBR.sort_values("threshold",ascending=False)
        save_path = param_TEST_MASTER["cba_out_path"]+"%s_CBA_report.csv"%(correction_type[c])
        print("saving..%s"%(save_path))
        df_temp.to_csv(save_path,index=False)
        df_temp=df_temp[df_temp.threshold>0.85]
        save_path = param_TEST_MASTER["cba_out_path"]+"%s_CBA_report.png"%(correction_type[c])
        line_plot(df_temp,'Score','Normalized values','Identifying maximum profit point- (%s)'%(correction_type[c]),['threshold'],["threshold"]+["n_total","n_goods","n_bads","n_profit"],'threshold','value',vertical=vertical_list[c],save_path=save_path)
        c+=1  


    