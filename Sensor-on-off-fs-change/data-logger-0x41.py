from ina219 import INA219

import sys

import time

import csv

SHUNT_OHMS = 0.1

ina = INA219(SHUNT_OHMS,address = 0x41 , busnum=1)

ina.configure()
 
header = ["time s","Current_mA","Voltage_V","power mW"]

filename = "PKL_Saved_Files/margin1.5/GB/wisdm_Power_Board.csv"

data_all = []
 
 
with open(filename, 'w', newline='') as csvfile:
 
    writer = csv.writer(csvfile)

    writer.writerow(header)

    while True:

        p = ina.power()

        t = time.time()

        #print(p)

        v=ina.voltage()
        A=ina.current()

        data = []

        data.append(t)
        data.append(A)
        data.append(v)
        data.append(p)
 
 
        # Write the data rows

        writer.writerow(data)

#        time.sleep(0.05)
 
