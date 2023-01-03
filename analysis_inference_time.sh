#!/bin/bash
START=0
END=30
for interval in $(seq $START $END)
do  
    echo "The interval of template is $interval."  
    python main.py --interval $interval
done  