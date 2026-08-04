#!/bin/bash

cleanup(){
 echo "Stopping data loggers..."

 kill  $LOGGER2_PID 2>/dev/null


}
trap cleanup EXIT

python3 data-logger.py &
LOGGER2_PID=$!

python3 main_Board_2.py
#python3 main_Board.py
