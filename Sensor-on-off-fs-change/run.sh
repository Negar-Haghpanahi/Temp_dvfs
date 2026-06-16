#!/bin/bash

cleanup(){
 echo "Stopping data loggers..."

 kill  $LOGGER2_PID 2>/dev/null


}
trap cleanup EXIT

python3 data-logger-0x41.py &
LOGGER2_PID=$!

#python3 main_Board_2.py
python3 sensor_control.py
