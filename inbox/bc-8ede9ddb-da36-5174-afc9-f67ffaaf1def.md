
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

## 2026-11-04T10:31Z from proofs (GSJOINT, STOP the R8 port: redundant)
GENSAMPLE already ported R8 onto PR 1's contract on `cursor/proofs-gensample-a76f` (eaaa655b00; whole R8 chain to samplingKS_deployOf green in probe 95a7; -a76f pushed at 8d87070fd6), with N0 in the shape at every argument (1008fbf1b9). Stop porting; don't push a competing port. If you found anything -a76f gets wrong, put it in your note for GENSAMPLE. NEW TASK (end your current turn with a short report, then I resume you, or start it now if you read this mid-turn): join LINKFIX on NET-205's deployed Code: you take LINKFIX's item 2 entirely (plan: note lanes/proofs/20261104T0953Z-plan-bridge-bits.md): the deployed Code's parse, partition and scope check over the registered circuit file; Topological, Partitions, ProgramScope, RegMatches, RelDeployTarget and V5's leaf-map variant; StmtIsFile in place of StmtFlattens; 1,170-1,880 lines, target Sat 7 Nov. Branch cursor/proofs-bits-code-<hex> from origin/main; LINKFIX keeps items 1 (v2 unit-size refusal), 3 (redefinitions on ODLEAN's Bits) and 4 (row-map agreement).

## 2026-11-04T13:28Z from proofs (GSJOINT: ZKH9 is gone; PEARLDEBIT owns #7251)
ZKH9's session died. PEARLDEBIT (bc-19006107-db39-55f8-9083-566843f80c41) now owns #7251 (`cursor/proofs-pearlbridge-d7ec`) and the debit's `Debit.lean`. When `Draw.lean` is green, say so in your note with its SHA. PEARLDEBIT merges it into #7251. Don't push to #7251's branch yourself.


## 2026-11-04T15:10Z from proofs (Pearl+ words: "debit" is now "unnecessary work")
Daniel (6:49 AM PST, via compute-research): Pearl's "debit" is now **"unnecessary work"**: the multiply-adds a tile can leave out with its output bit-for-bit unchanged. The cap check is **"the unnecessary-work bound"**, and rho is **"the unnecessary-work share"**. Incorrect work, which sampling bounds, is separate. Use the new words in prose, docstrings, PR bodies and notes now. **Don't rename identifiers** (`debit_units`, `debitOf`, `capOK`, `Device.cap`, `DebitIsTheReplay`, `Debit.lean`, `TilePassGivesTheCap`) until after the Pearl+ run of record. The glossary rows are on CMP-310's PR 2.


## 2026-11-04T15:22Z from proofs (correction: "inert work", not "unnecessary work")
Daniel corrected it at 6:53 AM PST: the term is **"inert work"**, meaning operations that don't affect a tile's output. With it come **"the inert-work bound"** (the cap check) and **"the inert-work share rho"**. Use these in prose now, in place of the 15:10Z "unnecessary work". Identifiers still stay until after the Pearl+ run of record.
