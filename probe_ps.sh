R=/workspace/research/runs/$1
ls $R/out 2>&1 | head; cat $R/out/summary.txt 2>&1 | tail -5; tail -5 $R/stdout.log 2>&1 | cut -c1-250
ps -eo pid,pgid,etimes,stat,rss,args --forest | awk -v g="$2" '$2==g' | cut -c1-250 | head -20
