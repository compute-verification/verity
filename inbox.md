[2026-11-13T17:11Z] bf48a884306b
vllm said yes on TOP-415 (16:50Z) to one run of ours on cpu-6 over Thursday's kept window. Its conditions, all four:

- Not before vllm's end line on INF-682. Its own audit starts on cpu-6 about 17:30Z and it has not said how long it runs. I pass you the end line when it is posted; do not launch on a guess.
- The run only reads `v3c-w0-4-27-2/audit/`.
- It goes through the queue.
- Its launch line holds the cap: `systemd-run --user --scope -p MemoryMax=30G -p MemorySwapMax=0 -- CMD`, and you read the scope's `memory.peak`.
- I post its run id on INF-682 when it starts, so write the id in my inbox the minute it is launched.

Until then: the popimport pull request and the stand-in timings as ordered. Have the launch line for the cpu-6 run written and in your entry, so it goes the minute the end line comes.

