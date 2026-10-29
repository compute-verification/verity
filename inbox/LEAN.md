# LEAN: the PoUW lead proved in Lean (from 16:10Z, 29 Oct)

**Daniel, 9:00 AM PDT (16:00Z):** he accepts ShaWrites, that a SHA-512 compression takes about 3,500 registers (credited
per form: 3,515.38 for the words, 3,449.05 for A), as a premise milder than SAK. "Let's get this done as fast as
possible." Tile grain stays the copy model (decision 2's default): he hasn't ruled against it, and the statement names
it, so it is one line to change.

## Done when

On `main`, one listed guarantee for the lead design (`tag-blc4a-t64-rot/served/s8/k32/w/l1w/l1a/min`, at the layout the
kernel serves), stated in the Pouw area's `Guarantees` module and proved in `security_proofs`:

- **The statement:** for every prover strategy, its priced W1 work over a call is at least the honest work less the gap.
  The gap is a defined function of the layout's parameters, and a separate lemma evaluates it at T 512 × 64, T 512 × 512
  and T 8,192. If the proof gives 0.42% rather than 0.04–0.07%, the statement says 0.42%.
- **Assumptions:** named `Prop`s only, each in `Security/Definitions/Pouw/Assumptions/` with its claim id, and listed in
  `Security/lean-audit.json` with an owner:
  - the device and model premises: H32, W1, tile grain (CopyGrain at the tile width, under model W), LaneSemantics,
    OneInstructionMinimal and CardWithinAccel W1;
  - ShaWrites (Daniel's ruling above);
  - `trusted-nebius-host`: the certifier draws G and streams the coin lanes after it holds A_j;
  - any protocol premise the code can't be checked against yet, named, with an owner.
- No `sorry` and no axiom, and the Lean check passes. The honest and the nontrivial lemma are listed. The record lands,
  and Daniel's DM of it is the check.

## Rules

- **Read `.agents/skills/lean-proofs/SKILL.md` first.** In particular: the statement comes first, never make a
  guarantee easier, assumptions are named Props, and keep out of each other's files and build directories.
- **The statement comes first.** N-PREM puts the guarantee's statement on the integration branch, with every open
  piece's statement as a `sorry` stub naming its owner. Nobody proves against a statement before N-LEAN has merged it.
  Until then, port your own scratch proofs and get your build working.
- **Files:**
  - Each lane owns `Security/Proofs/Pouw/Lead/<Lane>/`, the namespace `Pouw.SecurityProofs.Lead.<Lane>` and one
    aggregator, `Lead/<Lane>.lean`.
  - N-LEAN owns `Lead.lean` and every edit to a `lean-audit.json`.
  - N-PREM owns `Security/Definitions/Pouw/Lead/`, where the definitions a statement reads go.
  - A lane that needs a new definition or premise asks N-PREM in its checkpoint.
- **Builds:** use your own checkout. Build files in the proofs package (it needs ArkLib) on your node:
  `verity lean try FILE --on vy-nebius-cpu-N`. Records and `--update` runs are N-LEAN's alone.
- **Branches:** branch from the integration branch,
  `cursor/compute-research-np-lean-39e5` (N-LEAN merges `main` into it first), and name yours
  `cursor/compute-research-lean-<lane>-<suffix>`. Push after every commit. N-LEAN merges the lanes in, and only N-LEAN
  opens a PR.
- **No Slack and no Linear.** Report to the coordinator by finishing your task. Each checkpoint is a note under
  `lanes/compute-research` (`kind: report`), published with `research notes sync --path lanes/compute-research`.
- GPU work only on vy-dc-h200-5, through gpu-lease, and only once the coordinator confirms a window. Never touch
  vy-dc-h200. **Window UP (infra, 18:09Z): vy-dc-h200-5, one H200, GPU 0, through gpu-lease; lease to 20:40Z, hard stop 20:45Z.** Earlier: start once infra
  posts that it is up; the coordinator relays it. Have GPU runs done by 20:30Z.

## Settled during the program

- **16:45Z, coin-addressed stores** (N-W1's review): MH100w covers branching provers cell by cell (`adaptive_floor`), but
  not a store to a coin-computed address. So "for every prover strategy" can't rest on MH100w alone. The statement covers
  such stores through `CardWithinAccelW1` (the ACCEL premise Daniel signed off in ASK-169), at cw = 32 with
  `w1EntryWordsAt`, which the commitment writes read. N-PREM states it so. N-RED searches scatter-store strategies
  against it.
- **No run ids, PR numbers, tracker ids, dates or lane names in Lean docstrings or comments** (`tests/lint/test_repo_lane_text.py`).
  The coordinator's earlier "with run ids" was wrong. Evidence goes in your checkpoint note or the PR body.
- **`verity lean try` from a worktree builds the checkout the installed tool lives in,** not your worktree. Run the
  worktree's own copy of the tool, as N-W1 did (its note 20261029T1630Z-report-lean-mh100w has the command), and check
  that the run's tree is your head.

- **17:08Z, the statement is pushed:** `Pouw.Guarantees.Lead.G` in `Security/Definitions/Pouw/Guarantees/Lead.lean`,
  on `cursor/compute-research-lean-statement-4457` at 9a7028a28e. Its premises are in
  `Security/Definitions/Pouw/Assumptions/Lead.lean`, each lane's stub is in `Security/Proofs/Pouw/Lead/<Lane>/`, and the
  gap list is note `compute-research/20261029T1706Z-report-lean-statement`. N-PREM is revising three points before
  N-LEAN merges, all on `G`'s side, with the stubs unchanged unless it lists a change: whole tiles and the single-use
  coin stream become verifier refusals rather than hypotheses; unused premises go; the gap is itemized against the card.
  Lanes start proving their stubs once N-LEAN merges.
- **17:14Z, network (N-EGRESS, note 20261029T1711Z-report-lean-network):**
  - `OutsourceWithinEgressLead` is false as stated, because nothing bounds `egress` by the cap (F1). It is restated with
    egress bounded by the cap the certifier enforces, accounted per forward (F2).
  - Step 1's coin lanes raise n_max to 157, so the network term is about 0.066% / 0.66% at caps 32 / 320 (F4).
  - certifier-coins is a named assumption under `trusted-host` (the tests refuse provider names), and the old
    `TrustedNebiusHost` goes (F6).
  - N-PREM applies these after its current fix and adds a NETWORK stub. N-EGRESS proves against its own restatement
    meanwhile.
- **17:22Z, the statement's fixes** (N-PREM, `cursor/compute-research-lean-statement-4457` at dd30c5c347, note
  20261029T1722Z-report-lean-statement-fixes):
  - `G` has no input hypotheses and covers every call. A call is committed over rows padded to 16 and columns padded
    to 8; padding rows have zero data but carry their coin lanes, tag and coins. **KERNEL: the served kernel must do the
    same.**
  - `CoinStream` gives each coin once by definition. `DataInRange`, `LaneSemantics` and `TrustedNebiusHost` are gone
    from `G`.
  - Decode is stated (`GapAtDecode`): 0.265% of padded stock.
  - Stub changes:
    - COINBRANCH: `distinct_le_writes` takes `CoinStream`.
    - TAGS: `repeats_le` drops the tile and data-range hypotheses.
    - PSTAR: `table_le` drops the tile hypotheses and takes `CoinStream`.
    - COMPOSE: `perCoin` drops the tile hypotheses.
    - COMMIT: `CommitL1w` counts A's real rows and the padded words.
  - Prove against the merged head, not 9a7028a28e.
- **17:24Z, COUNTER on 0c9b58b686** (an older merge; note 20261029T1721Z-report-lean-counter-tile):
  - Tiny calls broke `G` through a pre-coin table. N-PREM's padding rows that carry coins should close that, and it is
    being rerun on dd30c5c347.
  - The flat 2⁻¹⁰⁰ table chance fails on step-0 tiles, which have about 2⁻⁷⁶·⁹ coin settings. dd30c5c347's 49⁻⁸ at
    step 0 is looser than that. PSTAR: make `table_le` hold with N-PREM's per-step constants, not a flat 2⁻¹⁰⁰.
  - Exact whole-tile repeats are 0 at every size searched.
  - **Open, for Daniel's copy model:** `TileGrain` charges a patched word the whole tile's 128 registers. The
    measurements back that in effect (patching loses 6.4–7.6% once selection is priced), but not to the letter (W1:
    one register a patch). COUNTER is computing the gap under measured prices.
- **17:23Z, COMMIT is done** (N-HIDE, `cursor/compute-research-lean-commit-67bb` at 7fa8be0e2a):
  - proved: `RowBound` and `LeafFirstBlock` (a row digest binds its position), and `RoundsOpenAfterRoots` (g-private's
    ordering);
  - named assumptions `SubtreeLayout` and `SaltDerivation`, which `G` doesn't read;
  - carried-words needs no premise;
  - hiding's hybrid bound is moved in.
  - On the integration head, two things are broken: Compose's `G_of_wordsFloor` lacks the coin-stream binder, so
    COMPOSE must fix it against `CoinStream`; and `tests/test_lean_vocabulary.py` fails on `Pouw.Lead.Coins`,
    `Pouw.Lead.Layout` and `CoinBranch.Priced.Prog`, which their owners list or rename.
- **17:42Z, network restated** (N-PREM, statement branch at aa1d45c779, note 20261029T1742Z-report-lean-statement-network):
  - `Accepted` refuses a call whose egress exceeds `callBits`, the call's share of its forward's cap. This is a
    certifier requirement: it counts per call, where today it charges per window.
  - `OutsourceWithinEgressLead` is per bit, with β = 256/3·W_mac.
  - **The layout changed:** step 0 now carries the three coin lanes and 21 data lanes, so n_max is 256/3 at every step.
  - `HonestHashRuled` applies at egress 0. `CertifierCoins` is a named `trusted-host` assumption.
  - The totals are unchanged at the three points: 0.165–0.190% at cap 32 and 0.488–0.513% at cap 320. Decode is
    0.232% at cap 32 and 0.252% at cap 320, of padded stock.
  - **Per lane:**
    - NETWORK: prove the stubs in `Network/Term.lean` and import it from your `Network.lean`.
    - TAGS: step 0's words now include the coin lanes, so the exactness range has to cover them.
    - PSTAR: re-check the step-0 stream term (32/7⁸), since step 0 now has lanes.
    - KERNEL: step 0's three coin lanes and 21 data lanes, plus the padding rows.
    - COMPOSE: `G` and `honest` changed: the per-bit premise, the egress refusal, and `HonestHashRuled` at egress 0.
  - N-PPL is measuring T3 on this layout.
- **COUNTER, search first:** the coin-branch figure (0.084%) and the per-tile table constants (49⁻⁸ at step 0, 7⁻⁸ for at
  most 32 stream tiles, 2⁻¹⁰⁰ after). Both are analysis, not proved.


- **18:55Z, KERNEL on the H200** (runs 182946-137d, 183517-14b3, 183816-c88c; note 20261029T1851Z-report-lean-kernel):
  - The final layout serves with 0 mismatches across ten layouts, decode and ragged calls included.
  - T2 served, sync / async: 10.90× / 21.27× (64 rows), 37.62× / 65.61× (512) and 62.50× / 75.14× (8,192), within 1% of
    last night's.
  - Decode costs what a 64-row call does, because of a fixed per-call path: L1's seed hashing of A_j.
  - **Decided: the certifier draws every coin raw after A_j's root** (G by rejection sampling, b, a₁ and the lanes).
    Nothing is derived on the card from a seed that follows root(A_j): that path put grinding of A_j outside the
    statement, against live coins only. `Coins` stays as stated. The coin feed changes before the freeze, and
    N-KERNEL writes its spec.

- **19:37Z, Daniel decided two things:**
  - **The copy model is tile grain, restated to what's measured** ("Sure"). N-RED's restatement, backed to the letter,
    replaces `TileGrain`'s wording.
  - **Hiding is A, the keyed leaf** (ASK-192, "A sounds fine"). The per-call commitments are keyed leaves,
    SHA-512 from S_w = f(IV, ks_w ‖ salt_w), not hm96 leaves with L1 + (a) salts.
    - The lead's hiding term goes to 0 (N-PREM drops it).
    - `Pouw.Salt.keyed_leaf_hides` is stated in Lean and locked (N-SALT).
    - The committer (kernels) and the kernel's seed path (N-KERNEL) change.
    - The AGENTS.md line becomes "Commitments are hm96 rows or keyed leaves in frame-v3-sha512 trees" in the PR that
      builds it.

- **20:05Z:**
  - **The coin feed and layout.h are frozen** at d5687db8d5 (spec: note 20261029T1943Z-report-lean-kernel-coin-feed).
    The certifier draws every coin, G is rejection-sampled, and nothing is derived on the card.
  - **T2's denominator is the CUDA-graphs stock step,** with eager beside it. Last night's 512-row T2 was overstated:
    13.6× / 19.2× against graphs, not 37.6× / 65.6×.
  - Decode T2 is 15.6× / 23.3× at 1 row and 14.3× / 20.4× at 16 rows against graphs, or 3.2× against eager.
  - Freeze 3, the keyed-leaf bytes, waits on N-SALT.

## Owners' answers on landing the code (18:32Z)

- **compute-accounting (CMP-295):** the lead is a new protocol id, `verity/pouw/v2`, in the `PouwCheck` package beside
  v1, which stays until v2 credits a served window. Our lanes write the PRs and they review them, on three conditions:
  - the fold's 2²² range check is in the Definition;
  - a CPU test covers a ragged call's padding-row coins;
  - nothing is credited until a `Properties.Pouw.Lead` statement ties the verifier case to `Accepted`.
  soundness then moves `Core.Remote.components` to v2.
- **circuits:** reviews `pouw/circuit/lead.py`, its circuit-check sidecar and its pins. To land: the circuit-check
  report in the PR body, outputs of at most 16 bits, every piece stating its ANDs, and a test of its size-limit refusal.
- **kernels:** owns the committer (`native_rows.cu`, step_rows.py, row_tables*, acquire/) and reviews our kernel. It
  starts on CPU now, and codes after the layout freeze.
- **vllm:** owns the vLLM integration (the lead method beside H1's in engine/int8_vllm.py). It starts when the
  kernel's layout.h and the coin feed are frozen in the statement's terms. Ping vllm when the freeze lands.
- **network-accounting (NET-145):** no answer yet.
- **Ours:**
  - the kernel PR (N-KERNEL);
  - the PoUW v2 protocol: `lead.py`'s Definition with circuit-check, and the v2 case in `PouwCheck`;
  - the certifier, if network-accounting agrees;
  - `Properties.Pouw.Lead` (the verifier case is `Accepted`).

## Lanes

| lane | agent | piece | node |
|---|---|---|---|
| STATEMENT | N-PREM | the guarantee, its assumptions, each open piece's statement and the gap list (each lemma's name, size and owner), and the statements of the honest and nontrivial lemmas | cpu-2 |
| INTEGRATE | N-LEAN | merge `main` into the integration branch; move tonight's Pouw proofs (Alphabet, Expansion, MassFloor, Tag, Grind, Adaptive, Pointwise, Ssn) under the lanes that use them; merge each lane; records; the PR, which also adds the ShaWrites ruling as one line under the friction skill's "Standing rulings" | cpu-1 pool, cpu-7..9 records |
| MH100W | N-W1 | take `Security/Definitions/Pouw/Assumptions/Dimension/H100W.lean` (and `Embed.lean`) out of "draft, pending review": check each definition against the measurements (the 58 lane forms, the census, h200Measured), fix or flag any mismatch, write the review into the docstring, remove the marker | cpu-3 |
| COINBRANCH | N-ADAPT | the coin-branch bound per path, revisits (A2r) included, on the `-rot` layout at tile grain under model W | cpu-4 |
| TAGS | N-EXACT | tag distinctness at tile grain and shift separation for the wide tag and `-rot`'s lanes; the fold is exact while \|A·Bᵀ\| ≤ 2²² | cpu-5 |
| PSTAR | N-LIT | the exact p* for `blc4a-t64-e7`'s three lanes when they share coins, under `-rot`'s weights (127, 1, 1) / (1, 127, 1), and the cases the 127 weight opens | cpu-6 |
| COMPOSE | N-COMPOSE | the guarantee's proof from the pieces, each taken as a hypothesis until it lands: the matmul floor, hash-compose under ShaWrites, the network term, and the period inventory with hosts counted | cpu-8 |
| NETWORK | N-EGRESS | the network term at caps 32 and 320 under `-rot` (OutsourceWithinEgress); certifier-coins as a property of `NetworkCertifier` where that fits, else a named assumption | cpu-9 |
| COMMIT | N-HIDE | row-bound, subtree-layout, carried-words, salt-derivation and g-private: prove what the Lean code can see, and state the rest as named assumptions with owners; move hiding's hybrid bound in | cpu-7 |
| KERNEL | N-KERNEL | `-rot`'s parity fold in the kernel, checked on CPU against the Python fold (N-EXACT's); measured on the H200 when a window is confirmed | CPU first |
| COUNTER | N-RED | for each frozen statement, before it is proved, search small instances on CPU for a prover strategy that falsifies it, and report each with a run id | cpu nodes via `research run` |
| REVIEW | N-SMELL | smell-check the statement against the design and the card: does it say what it reads like, and is no guarantee made easier? | — |

## Checkpoints

- **18:00Z (11 AM PDT):**
  - N-PREM's statement, with its stubs and gap list, is on its branch.
  - N-W1's review has made a first pass.
  - Every other lane posts its plan and the size of its piece.
- **18:30Z:** N-LEAN has merged the statement. The coordinator gives Daniel an ETA from the gap list's sizes.
- Then each lane posts a checkpoint whenever it closes a lemma or changes its plan.
