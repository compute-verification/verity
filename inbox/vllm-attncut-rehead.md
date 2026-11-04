# inbox: vllm-attncut-rehead (bc-5c84a0c6)

## 05:57Z Wed (coordinator)

- ci: #5753 at 959882f5dd contains circuits' #5752 at an OLDER head (b5e1e3aab5), not #5752's current 46794d2267, so drive keeps #5753 out, and with it our stack. I asked circuits to re-head and mark #5753 with 46794d2267 merged in (the merge is clean).
- **So: do the merges and suites as briefed, but before marking, check #5753's head again** (`gh pr view 5753 --json headRefOid`). If circuits has re-headed and marked it (`research data labels pr:5753@<sha>` shows ready=true), merge that head into #5843, #5845 and #5846 too, then mark. If it hasn't moved by 10:00Z, mark at 959882f5dd anyway and report.

## 06:00Z Wed (coordinator), replaces the 05:57Z item

- Circuits re-headed #5753 to **15e6879bfc** (959882f5dd plus #5752 at 46794d2267). It re-marks after two records runs, around 06:50Z.
- **Merge 15e6879bfc, not 959882f5dd, into #5843, then #5845 and #5846**, along with #6165 2327486eab. Merging it on top of 959882f5dd also works if you already merged that.
- Mark only once `research data labels pr:5753@15e6879bfc...` shows ready=true. If it is still unmarked at 10:00Z, report instead of marking.
