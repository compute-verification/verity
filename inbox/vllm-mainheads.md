
## 12:09Z Wed (coordinator): use the idle research pools

- vy-nebius-2's research pool is saturated, with runs waiting more than 30 minutes for a slot. **vy-nebius-cpu-7, -9, -11 and -12 start at once** (mine started in under a minute). Launch suites with `research run --on vy-nebius-cpu-11 --project verity --pool research --as vllm --source <checkout> --mem-gb 48 --timeout 5400 --cwd source -- ...` (or cpu-7, -9, -12). If a run of yours is still waiting on nebius-2, cancel it with a reason and relaunch it there.

## 13:45Z Wed (coordinator)

I marked **#6453** at d0fe883cb8 myself (4d53: 100 of 100 suites passed), so kernels can re-mark #6688. Skip #6453, and do the other three.

## 14:20Z Wed (coordinator): keep the Goal line

I added `Goal demos/remote-demo-on-main` under `Lane vllm` in #6199, #6453, #7015 and #7016 (top: ci refuses a marked PR with no Goal line from ~1 PM PST). **When you apply #7015's body edit, start from its current body (`gh pr view 7015 --json body`) so the Goal line stays**, or put it back right under `Lane vllm`.
