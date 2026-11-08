#!/usr/bin/env bash
# b32prog.sh RUN: a neutral.sh run's summary and the newest lines of its neutrality log.
R=/workspace/research/runs/$1/out
tail -4 $R/summary.txt | cut -c1-200; tail -c 3000 $R/neutrality.log | tr -s ' ' | tail -6 | cut -c1-250
ps -eo etimes,rss,args --sort=-rss | head -4 | cut -c1-200
