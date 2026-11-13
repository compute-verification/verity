[2026-11-13T00:44Z] 712ac003bc27
From the vllm lead, 00:46Z: one addition to tonight one-card check, on VLLM-380 (00:45Z): run the SM clock sampler (nvidia-smi --query-gpu=index,clocks.current.sm,temperature.gpu --format=csv -l 5) to a file under out/ beside the job, stopped by its PID after; post its first lines and line count with your result.

