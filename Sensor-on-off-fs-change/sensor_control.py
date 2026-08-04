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
#CSV_FILE = "lsm303_data.csv"

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
    chunk_size = 30
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
   #     with open(CSV_FILE, mode='a', newline='') as f:
   #         writer = csv.writer(f)
   #         for x, y, z in batch:
   #             writer.writerow([current_phase, timestamp, x, y, z])
   #             sample_count += 1

def init_sensor():
    # Force mag to sleep
    write_reg_mag(0x02, 0x00)  # MR_REG_M: normal mode

    write_reg(0x2E, 0x00)      # FIFO bypass (reset)
    write_reg(0x24, 0x40)      # FIFO_EN = 1
    write_reg(0x2E, 0x97)      # Stream mode, watermark = 24
    write_reg(0x22, 0x04)      # Route watermark to INT1
    read_fifo_chunked()        # flush

def set_odr_Acc(ODR):
   # print(" ODR: ", ODR)
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
    write_reg(0x2E,0x00)
    write_reg_mag(0x02, 0x00)  # MR_REG_M: normal mode
    write_reg(0x20, odr_reg_val)
    write_reg(0x24, 0x40)      # FIFO_EN = 1
    write_reg(0x2E, 0x97)      # Stream mode, watermark = 24
    write_reg(0x22, 0x04)      # Route watermark to INT1

def set_odr_mag(odr_reg_val):
    write_reg(0x00, odr_reg_val)
# --- Initialize CSV ---
#with open(CSV_FILE, mode='w', newline='') as f:
#    writer = csv.writer(f)
#    writer.writerow(["phase", "timestamp", "x_ms2", "y_ms2", "z_ms2"])
def set_sensor_off():
    #print("Sensor off")
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

if __name__=="__main__":
# init_sensor()
# time.sleep(10)
# set_odr_Acc(100)

# while True:
# time.sleep(5)
# set_odr_Acc(400)
# time.sleep(10)
  # set_odr_Acc(1344)
  # time.sleep(3)
# set_sensor_off()
# print("off")
# time.sleep(10)
