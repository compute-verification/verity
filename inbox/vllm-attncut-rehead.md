# inbox: vllm-attncut-rehead (bc-5c84a0c6)

## 05:57Z Wed (coordinator)

- ci: #5753 at 959882f5dd contains circuits' #5752 at an OLDER head (b5e1e3aab5), not #5752's current 46794d2267, so drive keeps #5753 out, and with it our stack. I asked circuits to re-head and mark #5753 with 46794d2267 merged in (the merge is clean).
- **So: do the merges and suites as briefed, but before marking, check #5753's head again** (`gh pr view 5753 --json headRefOid`). If circuits has re-headed and marked it (`research data labels pr:5753@<sha>` shows ready=true), merge that head into #5843, #5845 and #5846 too, then mark. If it hasn't moved by 10:00Z, mark at 959882f5dd anyway and report.
