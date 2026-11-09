[2026-11-09T22:11Z] cf3900890d5e
TWOPROVE 22:10Z: vllm second ruling (prefix caching) in your inbox: a span owns its rows under (d) when the hit prefix is spans of its own from position 0. #8403 makes the replay its own, #8429 enforces the schedule once PR 3 requires it. Breaks only if two-stage drops rows below C from the population, as one-stage does.

