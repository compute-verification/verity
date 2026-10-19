---
name: smell-check
description: >
  How to run a smell check in Verity, and the catalogue of smells every lane has found so far, grouped by the question
  that catches them: work that reads like what Daniel asked for but isn't. Credit for a side computation beside the served
  one, a "served" or "end to end" number from a model, a harness or stand-in values, a rule-following path that is opt-in,
  something public that should be private, numbers that don't compare, one value done twice, a check that passes for the
  wrong reason, a theorem or bound that says less than it reads, something deferred or only marked but stated as done, a
  claim relayed unread, an action beyond what was granted. Each entry quotes the words that hid it and its fix. Use when
  Daniel or a lead asks for a smell check, before you call something served, end to end, proved, exact, measured,
  credited or landed, when you relay another lane's claim, when a result looks too good, when you write or review a brief,
  or when you found a smell and want to record it.
---

# Smell check

Daniel sees the work only through what agents tell him, so he can't ask about a shortcut he doesn't know exists. On 19 Oct
the served Dither window turned out to credit receipts computed in a torch pass beside vLLM's own int8 GEMM. Its lane
called it "served", three lanes relayed that to him, and he found out only by asking a different question: "the smelliest
thing I've heard all day". A smell check is how agents find these first. Daniel's ruling (18 Oct, 10:31 PM PDT): he asks
every agent for a smell check from time to time, and each one reports what it finds and adds it to the catalogue below.
Run one too whenever a claim you're about to rely on or pass on carries one of the words below.

## Run one

1. **List the claims.** Write down what you were told (your brief, other lanes' posts, PR titles and bodies, labels) and
   what you told others (reports, cards, notes, PR titles, labels), and mark every claim that uses one of these words:
   *served, end to end, measured, proved, guarantees, exact, bit-exact, credited, accepted, PASS, committed,
   zero-knowledge, hiding, live, enforces, landed, works, pinned*.
2. **Read what backs each one, not its summary.** For code, the path that actually runs on the served or proved path
   (follow the call from the entry point; a docstring says what was meant). For a number, the run's own outputs and
   labels. For a theorem, its statement and the definitions it reads. For "landed", `main`. Ask each question in "What to
   ask" below.
3. **Ask the one question that matters:** would Daniel, reading the code and the run rather than the summary, recognize
   it as what he asked for? If you'd have to explain a "but", it's a smell.
4. **Report every smell you find,** yours or another lane's, in the format below. A smell in another lane's work goes to
   that lane's owner too. Report "none found" with the list of what you read, so the asker can see what wasn't.
5. **Act on it.** Fix your own smells without asking (the friction skill's 6 Oct ruling). Send another owner's to that
   owner. One that changes what a claim says or rests on goes to Daniel (`.agents/skills/asking-daniel/SKILL.md`). Relabel
   what's already published (`research data label`), and correct anything you told Daniel in the same report.
6. **Add it to the catalogue** below, under its question, newest first: in the PR that fixes it or in a one-line PR.

The report:

~~~text
SMELL <one line: what it looks like versus what it is>
where:  <file and line, PR, run id or art id>
quote:  "<the words that hid it, verbatim>"
bends:  <Daniel's intent or the AGENTS.md design rule it bends>
costs:  <what it changes: a claim, a number someone cites, compute, a decision already made on it>
fix:    <the fix and its owner; started or not>
~~~

## What to ask

Each question heads a section of the catalogue.

1. **Is what's credited or proved the thing that was served?** A second kernel beside the served one, a side computation,
   stand-in values, a reference pass, or a claim narrowed until the bound holds.
2. **Does "served", "end to end" or "measured" mean the real path?** Pinned build, compiled, the record's settings, the
   real workload and values, interference included, the verifier of record; not a harness, a model, a label, Python
   standing in for Lean, or a number nobody measured for this case.
3. **Is the rule-following path the record, or opt-in?** One hash, hiding leaves, live coins, everything committed at
   serve time, verifiers of record in Lean, Boolean circuits.
4. **Is anything public that should be a value?** A partition, a count, a salt or a field that follows the workload.
5. **Do the numbers compare?** Same baseline, metric, model, card, build and scheme; the scheme named in the label; a
   model's number labeled as one; a spec that reproduces it.
6. **Is anything done or kept twice?** Two GEMMs for one linear, two trees over one value with nothing tying them, two
   implementations of one format.
7. **Does each check fail when it should?** A negative refused for an unrelated reason, a gate that can't fail, a test
   that reads only its own fixture or never ran where it matters, a comparison of a thing with itself.
8. **Does the theorem or bound say what it reads as?** A definition that makes it vacuous, a premise known false or
   holding only for an honest machine, the honest prover's speed where the fastest adversary's is needed.
9. **Is anything deferred, or only marked, but stated as done?** Queued isn't landed; "will" isn't "does".
10. **Did you relay it without reading it?** A claim from another lane's summary is a quote, not a fact, until you've read
    what backs it, and a hypothesis goes out marked as one.
11. **Does an action or a default reach beyond what was granted?** An automated action on things it wasn't given, or a
    default that turns silence into a grant.
12. **Is there code on the soundness path that nothing runs?**

## Catalogue

Under each question, newest first; the lane that found or reported it is in brackets. Add yours.

### 1. Credited or proved isn't what was served

- **Two products in one launch, called fused** (19 Oct) [kernels]. kernels-dither-rtx2's h1f gate passed: one kernel
  a linear, everything bit-exact. But that kernel computed step s's plain z and step s−1's noised receipts, so the
  credited work was still beside the served product; #2684 credited it, since it checks ops, not kernels. The lag came
  from kernels' own brief ("z committed before each step's coin"). Fix: the one-stream form, where the served output is
  O = O' − Σ_d 128^d D_d of the noised mainloop (cgemm.cu), with per-call seeds under the 17 Oct exception; #2684
  refuses "two products in one launch".
- **The proved tile itself credited work beside the served product** (19 Oct) [compute-accounting]. PouwH1Tile_v2
  computes z by a separate EXACT call, "which no noise reaches", while its receipts fold the noised sets. Fix:
  PouwH1Tile_v3 derives z from the receipted words (H1V3).
- **A Commit measured without the values its openings need** (19 Oct) [kernels-maxtput]. The max-throughput arms
  keep no values (retain none), and openings are answered by replay, which nobody has run on a 1,014-wide batch. Fix:
  open one unit per arm at max throughput and check its replay (circuits, not started).
- **Dither credited a side computation, not the served GEMM** (19 Oct) [compute-accounting's W2 from the code, CMP-50,
  after kernels' question J11]. SERVE-H1 (#2639, draft) served through vLLM's int8 GEMM (`torch._int_mm`), ran a second
  `_int_mm` for z beside it, and computed the H1 receipts in a torch pass (457 s a window) that credit counted. Quotes:
  the receipts.py docstring, "as a side computation on a served int8 call ... z, the plain A Bᵀ cell, is the served y's
  integer product and no part of this computation"; #2639's title, "a served H1 + fx1 window"; compute-accounting, "the
  first padded static H1 + fx1 windows served on vLLM" and "W1 MET"; its 04:14Z rundown to Daniel listed it under
  *Served* and the side pass lower down, as a cost; top's morning page, "Padded static H1 windows are served on vLLM with
  live 64-byte coins"; kernels to Daniel, "Dither is already served in vLLM". Fix: one fused H1 + fx1 kernel produces y
  and z and commits z (kernels); a credit gate refuses a unit whose linear ran a second GEMM (compute-accounting,
  SERVE-GATE); all six runs relabeled "H1 receipts beside the served GEMM; not useful-work credit".
- **"Accepted" sessions that credited no memory** (18 Oct) [vllm's red team; vllm, memory-accounting]. VLLM-24's
  sessions reported "the session was accepted off-node, γ=0.9996/0.9989" with ε = 1.0 omitted: the ε was the harness's
  own reading, not remote_bandwidth's audit or the Lean's, and the ledger had no requirement link, so it credited the
  store's reads, not the circuit's. Nearly cited as Milestone 3. Fix: the done-condition is a served window, a
  requirement link and ε from the audit or the Lean; paced passes (MEM-39).
- **A8's "proved by M0 and accepted by Lean" ran on stand-in values** (19 Oct) [circuits]. "18 of 28 kinds lower through
  M0, each with a unit proved --zk and accepted by the Lean verifier", but the GLM Commit kept no values (retain=host), so
  the units were the Match capture's values. Fix: labeled stand-in; T1R's record Commit keeps values and the audits rerun.
- **A "Commit" benchmark row that isn't the deployed Commit** (18 Oct) [benchmarks]. "Commit arm passes quick 2/2 ...
  1.385x" committed the committer's own class selection under vllm-v1 (SHA-256, unsalted), not the registered units
  with salted rows-sha512; a 24-point sweep was dispatched on it. Fix: labeled "not the deployed Commit"; the arm moves to
  registered units with rows-sha512 in its identity.
- **census/v1's "served MACs" counted products that aren't served** (19 Oct) [circuits]. `served_macs_by_precision`
  counted GemmAccumulatorS8_v2's z products and missed PouwSegDot's, and network's datacenter credited them, so an H1
  tile's compute was over-counted. Fix: Boolean AND counts per facet are the credit (CENSUSAND); MACs informational.
- **Credited tensor-core work read a family's declared shape, not the Boolean body** (19 Oct) [network-accounting].
  "a tensor-core step's products (mac:<precision>, the census's PRODUCTS)" made σ's work preservation a hidden premise.
  Fix: NET-31, Boolean gates by facet.
- **"Stop crediting" a unit to make the bound hold** (15 Oct) [compute-accounting; Daniel's find]. A red team's fix,
  "stop crediting by cone", would have dropped attention's compute from the account while the bound read as covering the
  model. Daniel: "'stop crediting' feels weird to me. Like, eventually, we want to account for all compute use". Fix:
  noise on attention (decision 109, #2473).
- **"The cache works" with no restore ever recorded** [lean]. "cache.py get … work[s] on the nodes", but no Security
  build had been published since the layout move, so every fresh worker built Security cold. Fix: publish, then one
  recorded restore on a fresh tree, before anyone says it works (LEAN-31).

### 2. "Served", "end to end" or "measured" means a harness, a model or a label

- **"Native compiled vLLM" never cross-checked against vLLM's own entry point** (19 Oct) [kernels-maxtput]. The
  baseline is verity_vllm's offline wrapper; if it costs anything, every loss is against a slower baseline. Fix: one
  `vllm bench throughput` per card at the same settings.
- **A "realistic mixed" load that is two lengths with ignore_eos** (19 Oct) [kernels-maxtput]. Both native
  throughput and the hash's share depend on the mix. Fix: a trace-length run beside it.
- **"Served Boolean-first, Commit PASS" meant Python checks the Boolean lift** (19 Oct) [circuits]. "Llama-3.2-1B served
  Boolean-first has Match PASS and Commit PASS", but the replay ran the word accelerators and the Lean verifier doesn't
  check the word-to-Boolean lift yet. Fix: `sigma_check` ported to Lean (proofs); until then "lift checked in Python".
- **"Exact" FP8 and FP4 resting on our own measurements, not the vendor's** (17 Oct) [compute-accounting]. C1's security
  "carries over unchanged ... given the format's exactness", which rested on 462M and 504M outputs of our own runs; the
  H200's FP8 adder turned out to be an undocumented 14-bit window. Bends: no guarantee rests on a GPU attesting itself.
  Fix: FPX-MACHINE states the weakest FP premise per device with each source graded, and RT-FP attacks it.
- **"19.30% of node memory credited" with openings in the clear and a Python verdict** (18 Oct) [memory-accounting].
  Fix: every citation carries the caveat; ZK openings sampled (#2589, #2619); the Lean rerun agrees (#2635).
- **An overhead card resting on units nobody had agreed to** (18 Oct) [Daniel's find]. Kernels' card v1 priced the
  commit at per-layer units with a model that left out interference with serving. Daniel: "What do you mean per model
  layer instead of per kernel call -- I would talk to Proofs about this." Fix: the only overhead number is the pinned
  served loop, end to end, one change an iteration; the red team voided card v2's served-overhead cells
  (note:hash/20261018T2024Z-finding-redteam-numbers).
- **"G2: <5% at frontier decode" was a label, not a measurement** (18 Oct) [circuits]. Measured GLM-5.3 TP8 decode:
  1.40–1.45x for full rows, 1.13x per layer at 4 KiB. Fix: G2 is the measured curve plus the best achievable.
- **"Toward ~27 ms" was arithmetic** (19 Oct) [compute-accounting]. Kernels' #2629 message said the cut "should take the
  step toward ~27 ms"; measured, it was 33.39 ms. Fix: measure on the served harness before citing.
- **A memory floor nobody had measured for its case** [lean]. "its build peaks at 15–24 GB" set floors and advice; a
  whole-Security audit peaked at 189 GiB, and two runs were OOM-killed. Fix: floors from measured peaks per step.
- **A setting reported live while the binary that ran ignored it** (18 Oct) [infra]. "30-minute source-tree keep in
  effect on n1 and n2", but n1's pinned tool had no such flag. Fix: check the tool's own output ("deleted N").

### 3. The rule-following path is opt-in, or the record bends a rule

- **"Live coins" that Python's os.urandom drew, under no claim** [proofs]. A3, "The verifier draws the units it
  audits from IO.getRandomBytes", while the served draw's Stream defaulted to os.urandom. Fix: the Lean verifier of
  record answers the coins (#2666), and LiveChallenger's entropy is required (#2654).
- **A "public" draw whose rate the prover sets** [training's red team, proofs]. "a uniform kd-subset of the n
  units", with n read from the prover's own record field. Fix: a rate law, k = ⌈q·N⌉ with q public (SIGMA1).
- **NciCheck took run-time inputs as the registered baseline** [training's red team, proofs]. "The registered
  baseline is fixed before training and reused across declarations", yet recommitting it gave efficiency 0.3–1. Fix:
  refuse it as theta-not-registered, with a difftest vector (VC1).
- **Opening positions by Fiat–Shamir from the prover's own root** [vllm's red team]. c2_rules accepted the prover's
  command-line "verifier challenge" as "post-commitment by construction"; a cheat on 10% of leaves hides from the 64
  openings in about 1,000 tries (0.1 s). Bends: live coins only. Fix: VLLM-17 draws from the service challenger after the
  root is recorded.
- **A seeded-coin window passing the same gate as a live one** (18 Oct) [compute-accounting]. "The static windows,
  r20261018-215110-6e4a (live) and r20261018-214320-80a8 (seeded), both passed (SERVED-H1 OK)". Fix: the record's live line
  stays live; a device-seeded window is credited only labeled seeded, with GRIND's premium (p_eff ≤ G·p).
- **The Commit of record wasn't the design's hash** (18 Oct) [kernels, vllm]. The GLM-5.3 Commit of record used vllm-v1,
  SHA-256 position leaves with no hiding salt, and "the SHA-512 serving rows commit beside it" as an opt-in arm. Every
  overhead number on the record path understated the real one about 4x (262 vs 1,090 ms a Qwen3-8B run). Fix: SHA-512
  rows the only record (VLLM-17, circuits' land2).
- **A "PoUW" row measured through a side path** [benchmarks]. "PoUW overhead row: 3.34x" ran an experimental arm with no
  token control against stock FP8 and 0.55 GPU memory against 0.80; the protocol option was refused by the harness. Fix:
  BENCH-17, a protocol arm with stock FP8 as its control.
- **"The Lean keeps up live" only with an opt-in flag** [memory-accounting]. With the default settings every verdict came
  34–200 s after its draws. Fix: overlapped openings by default.
- **Records of record stale against the night's rules** (19 Oct) [circuits]. "Commit PASS on its AOT Program" for three
  records served with EOS honoured, under three known completeness holes. Fix: re-judged and re-served under the GLM v2
  train before they count.
- **A verifier deciding in Python that imports verity.ml** (18 Oct) [memory-accounting; Daniel's find]. "our verifier
  would import verity.ml.circuits. That code is already on the soundness path". Daniel: "This is currently the case? That
  smells bad... verifiers I think should only import verity.core code". Fix: edges frozen (#2522); Lean agrees on 4,430
  windows (#2558).
- **The datacenter's audit is a Python model, not a verifier of record** [network-accounting]. Fix: the README says so;
  the account's Lean is open.
- **NCI's check walked every gate of the unrolled circuit** (18 Oct) [network-accounting; Daniel's find]. Daniel: "fine
  with verifier doing O(circuitdescriptionlength) stuff -- but not O(unrolledcircuitlength)". Fix: per-Definition
  memoized check (#2552).
- **The accounting layer's circuits were word-level** (18 Oct) [network-accounting; Daniel's find]. Daniel: "Is our
  current thing not just literally a Boolean circuit?" Fix: Boolean models (#2587).

### 4. Public what should be a value

- **A weighted draw that leaked the live/dead partition** (18 Oct) [compute-accounting; proofs' RT10]. "weight >= total/K
  = fully proved" was neither fully proved nor private: weights from private annotations reveal the partition. Fix:
  uniform draws, with same-shape zero-knowledge emptiness proofs for dead slots.
- **The salt's frame carried the salt** (18 Oct) [network-accounting's red team]. "the window's record holds the
  commitment in the slot the frame takes, and never the salt", but the frame carried it. Fix: #2637.
- **Workload-dependent fields public in the audit's record** (18 Oct) [memory-accounting]. Appends, released, post_seed
  and per-pass steps went out through public(). Fix: declared outputs (#2548), γ per fixed window (#2566), public()
  trimmed.

### 5. Numbers that don't compare

- **"Negligible at a 1% budget", at a draw cost about 10⁴ times too low** (19 Oct) [proofs]. "q = s/(r + rho/w)
  = 6.1e-11" priced batched throughput against one-unit statements and left out serve-time commitments, which are 6–159
  times the budget. Fix: the register corrected (q about 1e-14), and training reissued.
- **Slot time reported as time to land** (19 Oct) [ci]. "The long pole is circuit-check ... 158 min per train",
  but circuit-check is deferred and never gated landing. Fix: report time to land separately from slot time.
- **Losses quoted against a baseline that isn't native** (19 Oct) [kernels-maxtput]. The pinned B200 stock runs
  the build's FLASH_ATTN, 13.4% under native FlashInfer (per-layer −7.0% reads −19.5% against native), and the RTX
  stock ran on the Commit's shrunk 58 GB KV pool. Fix: relabeled, and the table quotes true native.
- **TTFT read as latency when it is queue wait** (19 Oct) [kernels-maxtput]. Under an all-at-once burst, TTFT
  (47–131 s) scales with the request count. Fix: labeled; latency needs paced arrivals.
- **A plot that takes pin state per row and GPU, not per line** (19 Oct) [kernels-maxtput]. Fix: per line.
- **A decode metric that measured the whole run, "verified" against itself** (16 Oct) [benchmarks; caught by
  compute-accounting's W4]. "#2136: decode-only ratios match time per output token", but vLLM renames request ids, so
  every prompt token counted as decode, and the "match" compared one quantity with itself; the same renaming made
  "3.34x eager" and "11.55x eager" disagree. Fix: the harness refuses the metric when decode tokens exceed generated
  ones (VLLM-32).
- **ε read as a share of capacity** [compute-accounting]. "PoUW leaves at most ε of monitored compute unattested", but
  ε_W is a share of credited work; r is about 0.18–0.26, not 0.0074. Training quoted the same, "At today's values every tier gives an efficiency of 0.74%, set by the
  residual", so the dampener read about 135 where it is 1–1.8. Fix: ε_W and ε_s named apart; r = 1 − W_cert/C.
- **A PR title quoting numbers the Lean no longer states, "as Lean guarantees"** (19 Oct) [compute-accounting]. #2482's
  title read as proved while resting on a conjectured premise (OperandHMu) and one granted on a card. Fix: retitled with
  the Lean's numbers and premises.
- **−5% quoted as the dedup's value** (19 Oct) [compute-accounting]. Kernels' row 2, "71.31 → 67.66 ms (−5.1%)", came from
  a harness with no circuit Commit; the dedup's value (about 40 ms a step) shows only on the full path. Fix: labeled.
- **Quoted numbers from a retired scheme** (18 Oct) [kernels]. Two Pearl-C windows (3.097x, 1.630x) were the retired
  BLAKE3 -h2, and three overhead metrics disagreed. Fix: the scheme in every label; one metric against pinned stock.
- **Published overheads no spec reproduces** [benchmarks]. "All five quoted overheads reproduced inside the rerun's 95%
  intervals" came from hand-written wrappers. Fix: a spec per published table (BENCH-8).
- **A wipe's slack counted twice, at two confidence levels** [memory-accounting, training]. "slack = 1 −
  delta^(1/k) + floor", with a per-wipe δ of 1/100 beside 2^-40 elsewhere: at 2^-40, 86 rounds leave 27.6%, not 5.2%.
  Fix: counted once; one δ for the lifetime pool is open.

### 6. Done or kept twice

- **One value committed twice, by four implementations of one format** (18 Oct) [kernels]. A linear's input was committed
  by the Commit (BF16 rows) and by Pearl-C (FP32 row segments, its own tree) with nothing tying them, and four lanes kept
  a SHA-512 kernel for the same hm96-sha512 format. Fix: kernels' shared library; A-BIND opens Pearl-C's A from the
  Commit's table.

### 7. Checks that pass for the wrong reason, or can't fail

- **A gate killed at its cap read as "never ran", and the train landed** (18 Oct) [ci]. "touched suites passed;
  it lands" after a gate run exited 143 at its 90-min cap. Fix: a capped gate reruns once, then holds the train
  (#2583).
- **"4 checks of next running" while runs nobody followed held the slots** (18–19 Oct) [ci]. Three leaks kept
  runs in slots for 3–21 h, and a cancel left a job's workers alive (570 GB). Fix: #2553, #2565, CI-85, INF-93.
- **"L5 kernel census passes" never ran on any row of record** (19 Oct) [circuits]. No row of record has a
  per-step census, and L5 never gated on launch counts (an untraced in-place overwrite passed). Fix: CENSUSL5 refuses
  a kernel whose launch count differs from its declaring events; per-step census in every record run (CIR-45).
- **A custody wait that could never pass from another VM** (19 Oct) [kernels-maxtput]. Fix: fetch first
  (4123afff05).
- **A guard announced as enforcing that never ran** (19 Oct) [infra]. "task_hook.py (a coordinator's foreground local
  Task refused; proofs' rule 1)", but Task reached hooks as CallDynamicTool, so the matcher never fired and nothing was
  logged. Fix: #2678, verified only once a live log shows a record.
- **Negative tests refused for the wrong reason** (19 Oct) [compute-accounting, top]. "3 of 3 planted negatives REFUSED"
  and, on Daniel's morning page, "and refused all three planted negatives": all three failed at the statement digest
  (S2/R7), so none reached the in-proof check. Fix: in-proof negatives, refused on the batched constraint
  (r20261019-045751-1c6b, #2675).
- **An unbound root let the receipt verify for any B** [compute-accounting, CMP-38]. "the receipt verifies", but the
  noise seed hashed a public input the prover fills, so A = −b zeroes the noised input. Fix: the verifier fills the
  roots from the Commit tap (ROOTS1, #2605).
- **A `meets` rule that `theorem KS_meets : True` passes** [lean]. "each claim has witnesses that its antecedents hold at
  a deployed record". Fix: the witness's statement must mention the record's constant and the claim's hypothesis heads.
- **An "explicit allowlist" that is all of Init** [lean]. "nothing on the verifier's path uses implemented_by or extern,
  apart from an explicit allowlist of core primitives", with 19 implemented_by and 148 extern constants allowed. Fix:
  allow by name the constants the walk reaches.
- **"Map agrees" compared the prover's map with its own trace** [vllm]. Fix: check against circuits' requirement units.
- **Commit C2 PASS read as soundness** [vllm]. "the Commit passes C2" is completeness: an honest prover passes. Fix:
  report the verifier's ε and γ over a served window, never a Commit PASS.
- **"All gates pass" on a gate set that skipped the conflicted suite** (19 Oct) [circuits]. Fix: the touched suites of
  every merged branch, verity/ml/circuits always.
- **"Definitions disagree with the capture" because the harness read the wrong layout** (18 Oct) [circuits]. It nearly
  sent a Definition change. Fix: DSAFIX, 257 of 257 agree.
- **A fix "closed" by tests that never ran where it mattered** (14 Oct) [benchmarks]. "verify re-baseline bug (b) is
  closed by resume-by-identity": fake-engine tests passed, but on the queue no point ran. Fix: #2048; BENCH-10.
- **Wipes not timed: the plan's interval taken on trust** [network-accounting]. Fix: NET-30.
- **A count that read only Slack** (17 Oct) [comms]. "25 unanswered asks", of which 23 were answered on Linear. Fix:
  Linear state counts (#2441).
- **A run-id heuristic that matched English** [comms]. A bare 4-letter hex word counted as a run id, so comms got false
  rings all night. Fix: a bare suffix needs a letter and a digit.
- **"Tested" outside the environment that runs it** [comms]. "PR-owner and handles output unchanged (tested)", but drive
  runs where the routes may not be readable. Fix: the registry keeps its fallback.
- **A warning standing in for a fix** [comms]. "a routed post warns when the checkout is stale" still let the ask drop.
  Fix: route from origin/main at send time.

### 8. Theorems and bounds that say less than they read

- **A recursion soundness statement only a SHA-512 collision could prove** (19 Oct) [proofs]. "Claim it, (d).
  Every change RT5 and RT9 named is applied", on `RecDeploy.KSBoundRoute` (#2546). Fix: REC1 restates it with staging
  and holder-agreement facts, then proves it.
- **"RecursiveSound is proved", with an event that is nearly empty** [proofs]. The lock lists it as proved. Fix:
  the knowledge form (PRF-40).
- **A lifetime pool missing its per-window and lag terms** (19 Oct) [proofs; training's red team]. "ε_total =
  Pool/C falls as 1/C", while an hour of lag is about 40% of χ. Fix: Pool = (1 + Δ)·ℓ + ln(1/δ)/κ (PRF-45).
- **ρ₂ cited as Lean-proved where, with dark compute, it's on paper** (19 Oct) [proofs; training's red team].
  "ρ₂ is the thread theorem's slowdown (`ThreadedThrottled`)". Fix: the note says T1′ is on paper.
- **A premise false as instantiated, behind a proved theorem** (19 Oct) [training]. "ThreadedThrottled is
  proved in Lean", but κ counted the LM head's forward products, so the threading premise was false at the deployment.
  Fix: κ 0.655, and a conjecture entry for the deployed circuit (PRF-82).
- **A theorem with no instance in any declaration we produce** (19 Oct) [training]. "ContainedTraining is proved
  in Lean ... Covers one forward chain.", but Programs compute matmuls as Dot16/Dot64 chunks, so no Chain exists. Fix:
  STATUS says so; a Chain over chunk gates is open (proofs).
- **A modelled residual used as the default** (19 Oct) [training]. "the machine model gives r about 1%", while
  honest kernels run 1.72 times the floor; host CPUs at INT8 AMX (11.8% of the server's FP8 rate) were neither
  credited nor bounded. Fix: residual 0.326 + clock 0.064 + host 0.118.
- **A sampling plan that holds zero-work templates at one draw** [proofs; training's red team]. "every step that
  can change a value is drawn with probability at least share / (overhead + replay)", with bits = 0 by default. Fix:
  harm is the downstream credited work, and value-changing templates at the floor are refused (OPT5).
- **A tier called necessary that isn't** (18 Oct) [proofs; Daniel's find]. Daniel: "why is this the case? this
  doesn't seem necessary, remember? … I'm really looking for clarity on what the necessary and sufficient objects are".
  Fix: re-tiered; NEC1 proved static structure is never necessary (#2647).
- **"Actual versus declared" stated position by position** (18 Oct) [proofs; Daniel's find]. Daniel: "actual
  versus declared doesn't require talking about position versus position … this would also be true if I replaced the
  term circuit with program". Fix: restated format-neutrally.
- **A bound "for every adversary" that one hard-coded collision falsifies** (18 Oct) [compute-accounting]. "for every
  adversary within T, every r and every oracle H, it requires prob (Bind w) <= δb" is false at any δb < 1, and was about
  to land with Compose. Fix: the Fiat–Shamir files dropped; binding is a per-prover term.
- **`Held` proved under premises that hold only for an honest node** (18 Oct) [memory-accounting's red team]. "the Lean
  Held guarantee is proved", but its rate premises held only for a node that never rebuilds, reads disks or uses links.
  Fix: token-bucket premises and `w2_meets` (#2633).
- **"Proved" on a premise falsified two weeks earlier** [memory-accounting]. PoUS retention "proved in the
  ideal-permutation model", a permutation falsified on 4 Oct. Training's tiers.py quoted "The slack of 1/19 is
  proved (SecureErasureMeetsFamily ...)", and the reference Feistel-SHAKE256 lets a developer keep 1.53%. Fix: a
  streamed-label wipe resting on SHA-512 as a PRF (MEM-51).
- **Committed bits leaving a spatial unit were free in the covert budget** [network-accounting]. Fix: NET-34.
- **A rebuild-time bound using our measured rate, not the verifier's ceiling** [network-accounting]. Fix: NET-30.
- **"NCI changes by 0" stated as proved** (18 Oct) [network-accounting's red team]. Fix: revised with a proved bound.
- **Vacuous forms, and Complete, ZK and Efficient trivially met** [lean; Daniel's find]. Found by review, not the check.
  Fix: LEAN-26's `fails` witnesses.
- **A theorem made vacuous by a definition** [process-design skill]. #118 redefined `MerkleScheme.Collision` and made
  `merkle_binding` vacuous; the axiom audit passed it. Fix: guarantees' records in the lock, read by a person.
- **A masked-sum credit rule that escaped its charge** [compute-accounting]. "credit = masked sum of selected tiles": a
  mask after the forming escapes the gated-cone charge. Fix: an always-proved tally and forming closure (CMP-40).

### 9. Deferred, or only marked, but stated as done

- **A post-landing regression that refused every deferred landing, silently** (17–19 Oct) [ci]. "research merge:
  after the landing, started post_train.py" read as if it ran; about a day of landings had no regression verdict. Fix:
  pass drive's landed attempt to post_train (CI-81).
- **"The certifier enforces" a check no proved step makes** [network-accounting]. "The certifier enforces it: a constant
  number of frames per bucket", but no gate compared a window's counts to the profile. Fix: #2652, NET-35.
- **"The served Program never pulls weights on a TP row" before the switch shipped** (19 Oct) [circuits]. Fix: say "after
  the static MoE switch" until it ships (#2665).
- **"The gate now kernel-checks the whole import closure" while its PR was queued** (19 Oct) [lean]. Fix: "ready, closes
  LEAN-28 on landing".
- **An issue moved to In Review when its PR fixed one item of three** (18 Oct) [infra]. Fix: move it only with the PR link.
- **Committing less at serve time than the claim needs** (18 Oct) [Daniel's find]. A plan to commit only the replay
  blocks at serve time. Daniel: "the kind of like, Lazy commitment scheme where, like, at serving time, we don't commit
  the IO of the proof blocks, only the IO of the replay blocks. This is like basically not going to work." Fix:
  everything committed at serve time (ASK-25).

### 10. Relayed unread

- **A grep relayed as "can't reproduce"** (19 Oct) [ci]. "I can't reproduce either on next's current tip"; vllm
  then found a real NameError. Fix: "reproduced" only from a test run, otherwise "not seen in the code".
- **"main is stuck", read from a commit time** (17 Oct) [proofs]. "origin/main tip c1e1c28838 (committer 18 Oct
  12:37 PM PDT)" raised a false alarm. Fix: read mergedAt or ci's tick before flagging.
- **A rule relayed into two briefs, the opposite of the verifier's** (19 Oct) [circuits]. "Split them to <=64 bits; no
  wider row form is planned", while M0, C-Flock and the Lean refuse a unit whose rows all fit one SHA-512 block. Proofs had
  written "wide rows → circuits split to ≤64 bits (no new verifier form)". Fix: at least one row over 888 bits
  (HmRow.lean:560); the worker is redirected.
- **A hypothesis relayed as the likely cause** (19 Oct) [infra]. A worker's first hour went to ruling it out. Fix: leads
  go out marked unverified.
- **"A-BIND is building" when nobody had launched it** (18 Oct) [compute-accounting]. Kernels waited three hours on
  nothing. Fix: check a worker exists before relaying its status.
- **"Sat unanswered for about 45 hours" for issues filed already Done** (18 Oct) [comms]. Fix: open one example before
  relaying a number about another lane.
- **A hole said to exist that didn't** (18 Oct) [memory-accounting]. "servedMacs feeds the stacking property's unit cap...
  unsound for any KV credit", but the cap charges no MACs. Fix: corrected; read the code before relaying.
- **A step named as the cause before reading the run's records** [lean]. Fix: read the run's own records first.

### 11. Actions and defaults beyond what was granted

- **A watch for Verity's agents that acted on every agent the key could see** (19 Oct) [infra]. It restarted Daniel's own
  Calendly agent. Fix: never act on an agent whose repos don't name compute-verification/verity (#2646).
- **A key card posted default-yes, then shown as "Daniel chose: yes"** (18 Oct) [infra]. Keys are Daniel's alone, and a
  default turned his silence into a grant. Fix: `research msg ask-daniel` to refuse a default-yes keys card (not started).

### 12. Code on the soundness path that nothing runs

- **Dead legacy kernels** (18 Oct) [kernels]. `verity/ml/kernels/hash_gpu` holds Poseidon2, BLAKE3 and SHA-256 Merkle
  kernels from archived B-Ligero, imported by nothing live but inside a unit core may import. Fix: proofs' ARCH1 retires
  the one caller and moves them to archive/ in three PRs (started 19 Oct).
