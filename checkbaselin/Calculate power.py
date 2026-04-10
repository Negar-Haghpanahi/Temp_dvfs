import pandas as pd import numpy as np
power_df=pd. read_csv("power.csv")
event_at=pd. read_sv ("Log. cv" )
time_array=power_df["time_s"]. values power_array=power_df["power_mi"]. values
def find_closest_index(target_time, time_array):
return np.argmin(np.abs(time_array-target_time))
events=dict (zip(event_df["Event"1, event_af["time"]))
id_w_star=find_closest_index(events ["WORKLOAD_START" ], time_array)
id_w_end=find_closest_index(events ["WORKLOAD_ENDS"], time_array)
id_1_star=find_closest_index(events[ "IDLE_START"], time_array)
idx_1_end=find_closest_index(events["IDLE_END"], time_array)
workload_avg=np.mean(power_array[idx_w_star:1dx_w_end])
ideal_avg=np.mean(power_array[idx_1_star:1dx_1_end])
print("workload power: ", workLoad_avg)
print ("idle power: ", ideal_avg)
