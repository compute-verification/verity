[2026-11-17T00:18Z] be6f50e0331c
Coordinator 4:20 PM PST: infra rule for node A: before any run that writes over 10 GB to /, read `df -h /` and do not start under 75 GB free. Apply it to your card-0 run for #10615 and say the free figure in your return.

