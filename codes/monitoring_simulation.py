from utils import param_fairness,param_UBR
from FairnessM import FairnessMaster
from utils import line_plot
import pandas as pd
import random
random.seed(2021)

out_path =param_fairness["out_path"] #"C:/Veritas/Submission/APIX/Veritas/output/monitoring/"
final = pd.read_csv(param_fairness["data_path"])
final["p_np_score_fix"] = final["prediction_prob"]
new_obj = FairnessMaster(final,"TARGET","p_np_score_fix","AGE",1,0.92,100)
df_metrices = new_obj.calculate_fairness_metrices()
#display(df_metrices[["threshold","equal_opportunity_difference","average_abs_odds_difference","disparate_impact"]].sort_values("threshold",ascending=False))
df_base = df_metrices[df_metrices.threshold==0.94]
df_base["month"] = 0
for i in range(1,12):
  df_new_applications = final.sample(frac = 0.5, random_state=random.randint(0,100000))
  new_obj = FairnessMaster(df_new_applications,"TARGET","p_np_score_fix","AGE",1,0.92,100)
  df_metrices = new_obj.calculate_fairness_metrices()
  df_temp = df_metrices[df_metrices.threshold==0.94]
  df_temp["month"] = i
  df_base = pd.concat([df_base,df_temp])
#display(df_base[["month","disparate_impact"]])
df_base[["month","disparate_impact"]].to_csv(out_path+"DI_monitoring.csv",index=False)


import random
random.seed(2021)

new_obj = FairnessMaster(final,"TARGET","p_np_score_fix","AGE",1,0.92,100)
df_metrices = new_obj.calculate_fairness_metrices()
#display(df_metrices[["threshold","equal_opportunity_difference","average_abs_odds_difference","disparate_impact"]].sort_values("threshold",ascending=False))
df_base = df_metrices[df_metrices.threshold==0.94]
df_base["month"] = 0
for i in range(1,12):
  df_new_applications = final.sample(n = 1000, random_state=random.randint(0,100000))
  new_obj = FairnessMaster(df_new_applications,"TARGET","p_np_score_fix","AGE",1,0.92,100)
  df_metrices = new_obj.calculate_fairness_metrices()
  df_temp = df_metrices[df_metrices.threshold==0.94]
  df_temp["month"] = i
  df_base = pd.concat([df_base,df_temp])
#display(df_base[["month","average_abs_odds_difference"]])
df_base[["month","average_abs_odds_difference"]].to_csv(out_path+"AOD_monitoring.csv",index=False)


df_AOD = pd.read_csv(out_path+"AOD_monitoring.csv")
df_AOD.columns = ["month","AOD_1000_random_approved_applications"]

df_DI = pd.read_csv(out_path+"DI_monitoring.csv")
df_DI.columns = ["month","DI_all_new_approved_applications"]
df_DI["AOD_1000_random_approved_applications"] = df_AOD["AOD_1000_random_approved_applications"]


df_DI.to_csv(out_path+"DI_AOD_monitoring.csv",index=False)
line_plot(df_DI,'Month','Fairness values','Monitoring- DI and AOD for AGE',['month'],["month","AOD_1000_random_approved_applications","DI_all_new_approved_applications"],'month','value',vertical=None,save_path=out_path+"DI_AOD_monitoring.png")
