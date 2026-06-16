
import pickle ,os , time
from ReadFile import LoadData
import argparse , csv , numpy as np
from test_Board import TestBoardControlled
import smbus2
import csv
import time
from gpiozero import Button
from signal import pause
import os

print(time.time(), "start")

# --- Configuration ---
I2C_ADDR = 0x19          # LSM303DLHC accel
MAG_ADDR = 0x1E          # LSM303DLHC mag
INT1_GPIO = 24
CSV_FILE = "lsm303_data.csv"

bus = smbus2.SMBus(1)

SCALE = 1 / 64.0
G_TO_MS2 = 9.81
sample_count = 0
current_phase = "1hz"

def write_reg(reg, val):
    bus.write_byte_data(I2C_ADDR, reg, val)

def write_reg_mag(reg, val):
    bus.write_byte_data(MAG_ADDR, reg, val)

def read_fifo_chunked():
    addr = 0x28 | 0x80
    total_bytes = 144
    chunk_size = 24
    raw_data = []

    try:
        for _ in range(total_bytes // chunk_size):
            block = bus.read_i2c_block_data(I2C_ADDR, addr, chunk_size)
            raw_data.extend(block)

        converted_samples = []
        for i in range(0, len(raw_data), 6):
            x_raw = raw_data[i + 1]
            y_raw = raw_data[i + 3]
            z_raw = raw_data[i + 5]

            if x_raw > 127: x_raw -= 256
            if y_raw > 127: y_raw -= 256
            if z_raw > 127: z_raw -= 256

            x_ms2 = x_raw * SCALE * G_TO_MS2
            y_ms2 = y_raw * SCALE * G_TO_MS2
            z_ms2 = z_raw * SCALE * G_TO_MS2

            converted_samples.append((x_ms2, y_ms2, z_ms2))

        return converted_samples

    except Exception as e:
        print(f"I2C Error: {e}")
        return None

def interrupt_handler():
    global sample_count
    timestamp = time.time()
    batch = read_fifo_chunked()
    print("FIFO--------------")
   # if batch:
    #    with open(CSV_FILE, mode='a', newline='') as f:
    #        writer = csv.writer(f)
    #        for x, y, z in batch:
    #            writer.writerow([current_phase, timestamp, x, y, z])
    #            sample_count += 1

def init_sensor():
    # Force mag to sleep
    write_reg_mag(0x02, 0x00)  # MR_REG_M: normal mode

    write_reg(0x2E, 0x00)      # FIFO bypass (reset)
    write_reg(0x24, 0x40)      # FIFO_EN = 1
    write_reg(0x2E, 0x97)      # Stream mode, watermark = 24
    write_reg(0x22, 0x04)      # Route watermark to INT1
    read_fifo_chunked()        # flush

def set_odr_Acc(ODR):
    print(" ODR: ", ODR)
    if ODR == 1:
       odr_reg_val=0x17
    elif ODR==10:
       odr_reg_val=0x27
    elif ODR==25:
       odr_reg_val=0x37
    elif ODR==50:
       odr_reg_val=0x47
    elif ODR==100:
       odr_reg_val=0x57
    elif ODR==200:
       odr_reg_val=0x67
    elif ODR==400:
       odr_reg_val=0x77
    elif ODR==1344:
       odr_reg_val=0x97
    else:
       print("invalid ODR")
       return
    write_reg_mag(0x02, 0x00)  # MR_REG_M: normal mode
    write_reg(0x20, odr_reg_val)
    write_reg(0x24, 0x40)      # FIFO_EN = 1
    write_reg(0x2E, 0x97)      # Stream mode, watermark = 24
    write_reg(0x22, 0x04)      # Route watermark to INT1

def set_odr_mag(odr_reg_val):
    write_reg(0x00, odr_reg_val)
# --- Initialize CSV ---
with open(CSV_FILE, mode='w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(["phase", "timestamp", "x_ms2", "y_ms2", "z_ms2"])
def set_sensor_off():
    print("Sensor off")
    # Accelerometer: power down all axes
    write_reg(0x20, 0x00)  # CTRL_REG1_A = 0 → all axes off
    # Magnetometer: sleep mode
    write_reg_mag(0x02, 0x03)  # MR_REG_M = 0x03 → sleep
# --- GPIO Setup ---
int_pin = Button(INT1_GPIO, pull_up=False)
int_pin.when_pressed = interrupt_handler
phase_buffer = []

def log_phase_buffer(phase_name, start_ts, end_ts):
    phase_buffer.append([phase_name, start_ts, end_ts])

import csv
import time

# --- Buffer to store phase info ---
phase_buffer = []

def log_phase_buffer(phase_name, start_ts, end_ts):
    phase_buffer.append([phase_name, start_ts, end_ts])

 
def parse_args():
     
     p = argparse.ArgumentParser()
     p.add_argument("--dataset_name", type=str, default="wisdm")
     p.add_argument("--n_est", type=int, default=120)
     p.add_argument("--max_depth", type=int, default=5)
     p.add_argument("--num_exits", type=int, default=4)

     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
     p.add_argument("--th_combination", type=float, nargs="+", default=[0.44793, 1.34381, 0.89587])    # th_list
    #     
     p.add_argument("--fs_base", type=float, default=250.0)
     p.add_argument("--sensor_wakeup_sec", type=float, default=0.0)
     p.add_argument("--print_trace", action="store_true")
     return p.parse_args()
     

def write_content_to_file(file, content, header): 
    writer = csv.writer(file)
    for line in content:
        row = [line[key] for key in header]
        writer.writerow(row)


def add_header(file, header):
    writer = csv.writer(file)
    writer.writerow(header)
    
    
if __name__ =="__main__":
    print("t1 is --> " , time.time())
    time.sleep(2)
    print("ts is --> ", time.time())
    
    args = parse_args()
    classData = LoadData()
    classData.Read(args.dataset_name)
    classData.SplitData()

    
    with open(f"PKL_Saved_Files/margin1.5/GB/{args.dataset_name}_trained_model.pkl", "rb") as f:
        all_models = pickle.load(f,encoding='latin1')
        
   
    x_path = f"PKL_Saved_Files/margin1.5/GB/{args.dataset_name}_X_test.npy"
    y_path = f"PKL_Saved_Files/margin1.5/GB/{args.dataset_name}_y_test.npy"

    if os.path.exists(x_path) and os.path.exists(y_path):
        X_test = np.load(x_path, allow_pickle=True)
        y_test = np.load(y_path, allow_pickle=True)
    else:
        
        classData = LoadData()
        classData.Read(args.dataset_name)
        classData.SplitData()
        X_test = classData.GetXtest()
        y_test = classData.GetYtest()
        
    model = all_models[0]['models']  
    init_sensor()
    time.sleep(0.1)
    print("Sensor Initialized")
 
    
    window_len = int(X_test.shape[2])

    # ------------------------------
    # Run board-controlled evaluation
    # ------------------------------
    all_result = TestBoardControlled(X_test=X_test,y_test=y_test,model=model,args=args,sensor_on=set_odr_Acc,sensor_sleep=set_sensor_off,fs_base=args.fs_base,window_len=window_len,sensor_wakeup_sec=0.0,print_trace=args.print_trace,)

    output_file = (
        f"PKL_Saved_Files/margin1.5/GB/"
        f"{args.dataset_name}_accuracy_results.csv"
    )

    num_exits = len(model.split_points)

    header = ["t_start", "t_end"]
    for i in range(1, num_exits + 1):
        header.append(f"t{i}_acq")
    for i in range(1, num_exits + 1):
        header.append(f"t{i}_compute")

    header.extend([
        "window_sched_sec",
        "sensor_total_on_sec",
        "sensor_total_off_sec",
        "compute_total_sec",
        "t_start_prediction",
        "t_end_prediction",
        "total",
        "true_label",
        "prediction",
        "correctness",
        "exit_level",
        "window_num",
        "data%",
    ])

    with open(output_file, "w", newline="") as f:
        add_header(f, header)
        write_content_to_file(f, all_result, header)

    os.system("pkill -f 'python3 data-logger.py'")   



  
    # FOR GB:
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="wisdm")
    #     p.add_argument("--n_est", type=int, default=120)
    #     p.add_argument("--max_depth", type=int, default=5)
    #     p.add_argument("--num_exits", type=int, default=4)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.44793, 1.34381, 0.89587])    # th_list
    #     return p.parse_args()
    
    
    
    # def parse_args():
    # p = argparse.ArgumentParser()
    # p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
    # p.add_argument("--n_est", type=int, default=120)
    # p.add_argument("--max_depth", type=int, default=5)
    # p.add_argument("--num_exits", type=int, default=4)

    # p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    # p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    # p.add_argument("--th_combination", type=float, nargs="+", default=[0.5198603, 1.559581,1.03972])    # th_list
    # return p.parse_args()


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="ACCGyro")
#     p.add_argument("--n_est", type=int, default=150)
#     p.add_argument("--max_depth", type=int, default=4)
#     p.add_argument("--num_exits", type=int, default=4)
#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.47 ,0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.3, 0.41 ,0.56, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.6931, 0.1732, 0.34657])    # th_list

#     return p.parse_args()

#    def parse_args():
#         p = argparse.ArgumentParser()
#         p.add_argument("--dataset_name", type=str, default="Epilepsy")
#         p.add_argument("--n_est", type=int, default=250)
#         p.add_argument("--max_depth", type=int, default=4)
#         p.add_argument("--num_exits", type=int, default=2)

#         p.add_argument("--tree_splits", type=float, nargs="+", default=[0.47, 1])
#         p.add_argument("--proportions", type=float, nargs="+", default=[0.36, 1])   # split_points
#         p.add_argument("--th_combination", type=float, nargs="+", default=[0.693147])    # th_list
#         return p.parse_args()
    
    #  def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Shoaib")
    #     p.add_argument("--n_est", type=int, default=200)
    #     p.add_argument("--max_depth", type=int, default=5)
    #     p.add_argument("--num_exits", type=int, default=4)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.48, 0.59, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.35, 0.47, 0.61, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.4864, 1.45943, 0.9729550])    # th_list
    #     return p.parse_args()
    
    
    
    
    
    
    
    # ===================================================================================================
    # FOR RF
    
    
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Epilepsy")
    #     p.add_argument("--n_est", type=int, default=80)
    #     p.add_argument("--max_depth", type=int, default=20)
    #     p.add_argument("--num_exits", type=int, default=3)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.34657359027997264, 1.3862943611198906])    # th_list
    #     return p.parse_args()
    
    # def parse_args():
    #     p = argparse.ArgumentParser()
    #     p.add_argument("--dataset_name", type=str, default="Shoaib")
    #     p.add_argument("--n_est", type=int, default=75)
    #     p.add_argument("--max_depth", type=int, default=70)
    #     p.add_argument("--num_exits", type=int, default=3)

    #     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.31, 0.54, 1])
    #     p.add_argument("--proportions", type=float, nargs="+", default=[0.39, 0.57, 1])   # split_points
    #     p.add_argument("--th_combination", type=float, nargs="+", default=[0.48647753726382825, 1.945910149055313])    # th_list
    #     return p.parse_args()

# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="ACCGyro")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=60)
#     p.add_argument("--num_exits", type=int, default=4)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.32, 0.47 ,0.59, 1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.3, 0.41 ,0.57, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[0.6931471805599453, 0.17328679513998632, 0.34657359027997264])    # th_list
#     return p.parse_args()


# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="wharDataOriginal")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=15)
#     p.add_argument("--num_exits", type=int, default=2)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.33,  1])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.34, 1])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[1.0397207708399179])    # th_list
#     return p.parse_args()



# def parse_args():
#     p = argparse.ArgumentParser()
#     p.add_argument("--dataset_name", type=str, default="wisdm")
#     p.add_argument("--n_est", type=int, default=60)
#     p.add_argument("--max_depth", type=int, default=15)
#     p.add_argument("--num_exits", type=int, default=2)

#     p.add_argument("--tree_splits", type=float, nargs="+", default=[0.33 , 1.0])
#     p.add_argument("--proportions", type=float, nargs="+", default=[0.34, 1.0])   # split_points
#     p.add_argument("--th_combination", type=float, nargs="+", default=[1.0397207708399179])    # th_list
#     return p.parse_args()
