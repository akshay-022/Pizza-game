#!/bin/bash
for ((n=1; n<=50; n++)); do
    python main.py -s $n -g False -p 1 -num_top 3 | grep -i 'Total Score S' >> output4.txt;
done

