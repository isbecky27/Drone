#!/bin/bash
END=4
for interval in $(seq 4 $END)
do  
    echo "The interval of template is $interval."  
    python main.py --interval $interval
done  