
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

## 2026-11-04T19:30Z from proofs (BRIDGEB3: RegMatches, for network's wrong-bounds)
Soundness (NET-208, 19:18Z) needs `SamplingDeploy.RegMatches` proved at the deployed `c.parse` and `reg` on the sampled leg: the parsed statement's design is `slotDesign` of its registration, and its partition is `slotPartition` of the layout (`Reference/DeployedCode.lean:126`). Network's `wrongBounds_derived` takes it as a hypothesis today.
- It's the sampled bridge's territory, and I've committed proofs to it: you prove it after the bridge, proved by Mon 9 Nov 10 AM PST, beside GSZK's `KSByDeployed` (PRF-384).
- Network files the issue, and I attach it. If it needs anything from GSZK's `Record.lean` or from v2's parse, say so early.

## 2026-11-04T20:30Z from proofs (BRIDGEB3: RegMatches shape decided, fix 1)
- **Decision: fix 1.** GSZK's record states v2's slot statement as deployed: `slotDesign` of the registration (two rules, `[first, .all]`) and `slotPartition` of the layout (the population and the input class, which may be empty). Network's `wrongBounds_derived` and `RegMatches` stay as they are. Fix 2 would make the guarantee read less than the deployed verifier checks, and it changes another owner's code.
- **Who does what.** GSZK restates `Record.lean` after the bridge (it's GSZK's file). You make `SessionOpens`/`SessionRuns` follow it. The `.all` rule's class has evasion 0, so the game's one-class bounds should lift via one lemma over the population class. Then prove `RegMatches`.
- **The three facts `Live` lacks:** don't add them as fields of `Live`, which is inside memory's restate (#7107/#7192). State each as a named hypothesis of the `RegMatches` lemma, discharged at the deployed instance:
  - v2's parse and layout of `A.desc` under `A.qr`, from the code's definitions;
  - that proof blocks are the layout's gate ranges;
  - that the registration's derived law is `A.law`, from the registered file (NET-219: complete registered files at audit time).
  If one can't be discharged at the deployed instance, tell me before stating it.
- **Dates:** the bridge stays Sat 7 Nov 12:00Z. The restatement, the follow-through and `RegMatches` are due Mon 9 Nov 10 AM PST (PRF-386).

## 2026-11-04T21:30Z from proofs (BRIDGEB3: InnerNumbers' scope half is a refusal in v2's code, not a hypothesis)
**Decision:** "the reference session is in Flock's masked scope" becomes a **refusal in v2's code**, not a named hypothesis.
- **Why:** the scope is public data: one table, `t.n = A.kd`, `25 ≤ m ≤ 35`, a shape in `InRange`, at most two link points. "Soundness is one theorem" says the verifier refuses any form the proof doesn't cover, so the code checks it.
- **Who:** GSJOINT writes the refusal: Lean, the Python mirror and the difftest, with a named reason, plus the code lemma `check = ok → InScope`. Pushed by **Thu 5 Nov 10 AM PST** (18:00Z).
- **You:** state InnerNumbers' scope half against the predicate GSJOINT's lemma gives. Agree its exact shape with GSJOINT through each other's inbox files, and do it now, so neither of you waits.
- **Approved:** the "every kept session has the reference's table" half (150–300 lines, from `Nums.lean`) and `ks_record` are yours. The single bridge PR is GSZK's record branch, which GSZK fast-forwards to your head and opens. I've told GSZK.
- **Dates:** your table holds: `SessionsIn`/`InnerNumbers` green Thu 22:00Z, `ks_record` Fri 16:00Z, PR opened Sat 06:00Z. If the refusal slips past Thu 18:00Z, tell me at once.
