[2026-11-22T01:54Z] d06790bb02f0
From vllm lead (read the clock yourself): a launch tip from infra (TOP-450, 5:53 PM PST): a quick run of a change that touches the adapter runs 87 suites on THREE test processes with the default job (4 CPUs, 16 GiB) and takes about two and a half hours. Ask for memory: research run --on auto --cpus 12 --mem-gb 54 gives the same run about twelve processes. Use that for any run of suites.py --changed, and name the test files directly (not the changed-suites run) when a few files are what you need.

