#!/bin/bash
# sweep.sh JOBS: the harness on each "NAME SHA" line, two at a time; one line of JSON a head in sweep.log
cd /tmp/cir208
xargs -P 2 -L 1 sh -c 'python3 check.py "$0" "$1" >> /tmp/cir208/sweep.log 2>> /tmp/cir208/sweep.err' < "$1"
echo "DONE $(date -u +%FT%TZ)" >> /tmp/cir208/sweep.log
