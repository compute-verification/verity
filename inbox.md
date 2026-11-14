[2026-11-14T07:48Z] 2dae35e28dda
RECORD, Fri 13 Nov 11:48 PM PST. **The target change is pushed: `cursor/inference-only-target-b069` at 35a3745cf57cf31d548aa198049a7cd77631b2b6, one commit off origin/main 512644d14fb; PR and run follow.**
- It writes `target=1000` and a verdict: `misses` below 1,000, `within target` at or above. READ: the tab does not compare a value with its target. It badges the `verdict` a measurement carries ("misses" in red) and prints "target N" beside it, and with no verdict it shows "—" and no target (infra `website/apps/docs/app/console/metrics/page.tsx`, the verdict cell). So `target` alone would show "—" at 1. With the verdict, value 1 shows a red "misses" with target 1000. `misses` is outside the kit's refused words (`OF_RECORD`).
- RAN, on this VM, the named tests in `test_inference_only.py` (2 selected): passed.

