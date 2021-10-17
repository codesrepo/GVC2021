from FairnessM import FairnessMaster
from utils import param_fairness,param_ROC
import pandas as pd

#import pipeline as Pipeline

if __name__ == '__main__':
       
        
    #POC correction parameters determined
    final = pd.read_csv(param_ROC['data_path'])
    new_obj = FairnessMaster(final,param_ROC['target'],param_ROC['score'],param_ROC["apply_ROC_attribute"],param_ROC['privileged_group'],param_ROC['base_trheshold'],param_ROC['threshold_bins'])
    _,interval_low,interval_high = new_obj.apply_ROC(final,param_ROC['low_ROC_margin'],
                                                     param_ROC['high_ROC_margin'],
                                                     param_ROC['num_ROC_margin'],
                                                     param_ROC['metric_lb'],
                                                     param_ROC['metric_ub'],
                                                     param_ROC['metric_name']
                                                    )
    print(_,interval_low,interval_high)
    final_ROC = new_obj.apply_ROC_correction(final,interval_low,interval_high)
    final_ROC.to_csv(param_ROC["out_path"]+"ROC_corrected_data_%s.csv"%(1),index=False)
    print("...saving",param_ROC["out_path"]+"ROC_corrected_data_%s.csv"%(1))
        
    
    

    
    
    
  
    
    
