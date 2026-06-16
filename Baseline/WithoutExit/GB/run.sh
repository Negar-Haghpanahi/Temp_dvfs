#!/bin/bash

cleanup(){
 echo "Stopping data loggers..."

 kill  $LOGGER1_PID 2>/dev/null


}
trap cleanup EXIT

python3 data-logger-0x41.py &
LOGGER1_PID=$!

python3 main_Board.py
