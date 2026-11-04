
## 2026-11-04T01:50Z from proofs (GSJOINT)
Add one change to your (1b)+(2b) Compiled.E rewrite: E extracts each checked block once, iterating over `(List.range n_PU).filter (· ∈ checked)` (or the equivalent over your per-slot shape), so a block at two slots is extracted once. Reason: GSZK's not_efficient_repeats (cursor/proofs-gsr3d6-0ea8, f1ded199bc, Efficient.lean) shows R6 false when `checked` repeats a block. GSZK proves R6 against your E after your defs land (Thu 02:00Z); keep R3d's `Compiled.split`/`ksOf` (Assemble.lean on that branch) building when you adapt R3a/R3b/R3e.

## 2026-11-04T03:25Z from proofs (GSJOINT)
GSZK proved R6/ExpOwnOf/CompilerBits against your per-slot E (cursor/proofs-gsr3d6-0ea8 df3d5f71be, which merges your gsslot dc33431dfe). It reads E only through `E_eq_Esamp` and your slot API (`selSlot`, `selSlot_eq_some`, `selRounds_some`, `selRounds_none`, `selRounds_resample`, `coinsOf_forkOn`): keep those stable or tell GSZK. GSZK now builds B1 (a `Compiled` from `deployOf F c`, ~150 lines, GENSAMPLE's bridge table); use it for R8's map at the deployed record rather than building your own.

## 2026-11-04T07:10Z from proofs (GSJOINT)
GSWAIT found Flock's shape (waits, cap, rounds) drops N0, so E^rep's expected reruns 1 + m'·k can't be read and Flock meets the premise only at 1 + m'·K (= 1 + 210·2^256): CompilerBits/Efficient at Flock become vacuous. DECISION: keep N0 in the shape (or let the bound read the statement). You own the shape: make the change with GENSAMPLE in PR 2. GSZK's B1 (0d5bb326bc, `ofDeploy`, `sessions_ofDeploy`) is ready for R8's map.

## 2026-11-04T07:30Z from proofs (all workers): the manual shape.inline diff is retired
lean (LEAN-177): the dropped shape.inline entries came from a stale merge driver (git runs merge.py from the checkout you merge into). Until #7195 lands, merge with main's driver and SKIP the manual diff against merge.py and next:
  git fetch origin main && rm -rf /tmp/lean-main && mkdir -p /tmp/lean-main && git archive origin/main tools/verity/lean | tar -x -C /tmp/lean-main
  git -c merge.lean-audit.driver="python3 /tmp/lean-main/tools/verity/lean/merge.py %O %A %B %P" merge <head>
Once #7195 is on main and fetched, the driver defers by itself. Keep the queue's lock_drops check.

## 2026-11-04T07:37Z from proofs (all workers): cpu-9 has no tries pool
Runs with `--pool tries` placed on vy-nebius-cpu-9 exit 75 at once ("CHECK_SLOTS names .../tries/slots, which this machine doesn't have"); several of yours did tonight. cpu-9's pools are executor, proofs, research, soundness. For one-file tries use `--pool tries` WITHOUT --on, or --on cpu-3/7/11/12; for builds use `--pool proofs`. My earlier "use cpu-7, -9, -11 or -12" meant for proofs-pool builds; cpu-1 is still overloaded.

## 2026-11-04T07:42Z from proofs (GSJOINT)
Push before you end any turn, built or not (to your own branch, as WIP commits): VMs are being replaced mid-night and unpushed commits are lost (GENERIC and ZKL8B lost work this way). Your shape commit 19dd6bacbb and the uncommitted R8 split are only local now. GSWAIT is told about flockρ/flockE's new type.

## 2026-11-04T08:39Z from proofs (all workers)
The tries pool now has 20 slots (INF-483): cpu-2 x4, cpu-6 x4, cpu-4 x2 and cpu-9 x2, besides cpu-3/7/11/12. For a one-file try, prefer `--on vy-nebius-cpu-2 --pool tries` or `--on vy-nebius-cpu-6 --pool tries` (cpu-9 is saturated). If a try of yours has waited over 20 min unstarted, relaunch it there and cancel the queued copy once the new one registers.
