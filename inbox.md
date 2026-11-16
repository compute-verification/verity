[2026-11-16T12:02Z] efb76ca8e954
Coordinator 4:10 AM PST: infra read node A at 4:00 AM as "load near zero", yet run cb5a entered python-verify at 3:55 AM and should hold 36 cores. Check now on the node that python-verify workers are alive and burning CPU on 0-19,60-75 (JOBS, nproc as serve.sh saw it, per-process CPU). If it is stalled or running on few workers, say so on TOP-402 and to me at once with the cause.

