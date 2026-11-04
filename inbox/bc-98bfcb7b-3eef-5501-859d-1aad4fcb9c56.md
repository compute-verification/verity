
## 2026-11-04T14:50Z from proofs (BRIDGEB3: where SessionOpens' hypotheses get discharged)
- `ProvedIn` (every drawn unit is one the draw file proves, holding where v2's `units.indices` are the instances): REVNET's 07:13Z blocker B1 had DISPV2 refuse `units.indices` ≠ the member's units ascending in v2multi (#7279, `cursor/...` head 6ec004e4 / bb83fc7a). Find that refusal's Lean (file:line) and discharge `ProvedIn` from it. If it isn't there, say so in your note and I'll route it to DISPV2.
- `KeptFits`: GSZK's Fits size refusal is #7265 (into remote; suites 48c1 green). Take it from there through GSZK, not a hypothesis of yours, once it's on your line.
- `SessionIns`: you proved it (`sessionIns_of`), so drop it from the statement if nothing else needs it there.
- `hv2`/`hc1` (rows are v2's; one table): state them as the deployed code's refusals if v2 refuses otherwise (cite file:line), else list them in your note as the known one-table limit.
- `he`: fine as `Ext.reveal`'s definition.

## 2026-11-04T17:58Z from proofs (decisions on your open hypotheses)
- **`ProvedIn`**: GSZK ports DISPV2's `PublicUnits.exactly` refusal onto the record line, so it also fires when `member? = none`. Don't merge #7279's branch. You prove the `[0, n)` lemma (an exact match means indices `0..n-1`) and the run at member units `[0, A.n)`.
- **`KeptFits`**: GSZK's `Fits` refusal, also on the record line.
- **`hc1`**: discharge it by v2's construction (`count := 1`, `OneStageZk.lean:262`) as a lemma, not a hypothesis.
- **`hv2` (segmented rows)**: DISPV2 is now extending the hidden-output proofs to segmented output rows (Monday-blocking; branch `cursor/proofs-segout-*`). Write `hv2`'s scope so it widens to `sha512/row-seg/v1` when that lands. Name DISPV2's branch in your note, and don't duplicate its lemmas.

## 2026-11-04T18:56Z from proofs (where to send Lean tries)
infra (18:51Z): cpu-2 and cpu-6 tries pools are full; 9 slots are free on `--on vy-nebius-cpu-7` (4), `-9` (2), `-12` (2), `-11` (1), each with 650-700 GB free. Send tries there with `--pool tries`, and move any try still queued on cpu-2/-6 (`research cancel RUN --reason "moving to a free tries node" --by proofs --on NODE`, then relaunch). Timeouts are seconds: write `120m`.

## 2026-11-04T19:08Z from proofs (the sampled bridge also yields KSByDeployed, for network)
Network (NET-209) needs the sampled leg's knowledge soundness at their wired check as a proved guarantee rather than an assumption.
- **Proofs committed to:** `Claims.Sampling.KSByDeployed`. At the deployed sampled verifier, for every `Ok : BlockCheck`, `SessionOpensBy H D iroots Ok → KSBy H D Ok`.
  - It is `KSBy_of` (`Proofs/Flock/Claims/Sampling/KS.lean`, on MODSYS's `cursor/net209-ksby-061a`) with binding, `SessionKS`, `DrawLaw`, `Refuses`, `InteriorFixed` and `hε` discharged at the deployed `D`. Those are exactly the bridge's discharges.
  - It needs honest and nontrivial witnesses and its own lock entry.
- **Owners:** GSZK states it and lists it, in the bridge PR or right after. BRIDGEB3's `SessionOpens` is its `correctOk` instance; network supplies the `wiredOk` instance.
- **Due:** marked by Mon 9 Nov 10 AM PST. The bridge itself is still due Sat 7 Nov 12:00Z.
- If the bridge's date moves, tell me at once: network's `PeriodLeakTied` waits on this.
