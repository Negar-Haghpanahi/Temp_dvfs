import multiprocessing
import time
import os
import signal
import csv
def cpu_stress():
while True:
pass
def log_event (writer, event_name) :
timestamp = time.time()
writer writerow([event_name, timestamp])
ifname== " main":
with open ("log. csv", "w", newline="" )as f:
writer=csv.writer(f)
writer writerow( ["Event", "time"1)
print("workload: ", time.time())
log_event (writer, "WORKLOAD_START")
processes=|]
for - in range (multiprocessing. cpu_count()): p=multiprocessing.Process(target=cpu_stress)
p.start()
processes. append (P)
time. sleep(10)
log_event (writer, "WORKLOAD_ENDS")
print("stopping : ",time. time())
for p in processes:
p.terminate()
p.join()
print ("ideal: "
20g_e(end(writer, "IDLE_START")
time. sleep(30)
log_event (writer, "IDLE_END")
print("cod ends: ",time, time())
os. system("pkill -f 'python3 data_logger.py")
