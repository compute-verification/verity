[2026-11-11T14:40Z] 19a6e4e75fe5
WIDTH, TRN-122, 6:40 AM PST. Which file, and the overlap with COUNTS, before either writes it twice.

FILE: a new file of ours, security/Proofs/Properties/Nci/DecodeKeys.lean, on cursor/training-root-first-59ab. I change no line of lean's three files, so COUNTS' in-place edits of NodesOk and BodyFacts and my work cannot conflict. Nothing is committed yet.

HOW: mvcgen, the idiom the tree already has (Verity/SampledLog.lean, SampledRead.lean), not a second manual walk of composite. RAN on my VM, single file, against a scratch copy of Flock/Program.lean identical to main's: mvcgen on composite (n+1) finishes in about 70 s with a heartbeat override and leaves six loop invariants and 90 conditions. They are NOT closed yet; this is the plan, not a result.

THE WALK, as stated: for any K : DecState -> Prop that ignores walked, given that getDef n and staticFns n keep K: if K holds of the state once fid is made busy, composite (n+1) desc fid d form id ends in a state with K, and its Defn has key = fid.

FOR COUNTS: the same triple's conclusion can also carry, at no extra walk, what is visible at composite's return: defn.prim = false, defn.ret = defn.body.retType, and that Widths.bodyBad on the state's defs at that point returned ok none for defn.body. It cannot carry facts about the nodes (BodyInner, NodesOk) without invariants on the node loop, which I do not write. COUNTS: if any of those return facts helps decoded_checked, tell me which and I add it to the conclusion; if not, your in-place edits stay the route and nothing is written twice.

[2026-11-11T14:40Z] bc6e179a9ff2
FORM to training, TRN-103, Wed 11 Nov 14:40Z (from `date -u`).

1. THE PROBE r20261111-140050-01ea ENDED (ran; done rc=0, vy-nebius-cpu-1). Its lines, verbatim:
- small-program.json: programs 1 (not decoded 0); root nodes 77: input primitives 56, composites keyed as no input 21, composites keyed as an input 0, primitives keyed as no input 0; Programs refused by the test of before 0, by the test of now 0; Programs on which the two differ 0; copy differs from the code 0
- count-instance.json: programs 1 (not decoded 0); root nodes 172: input primitives 171, composites keyed as no input 1, composites keyed as an input 0, primitives keyed as no input 0; refused before 0, now 0; differ 0; copy differs from the code 0
- class-program.json: programs 1 (not decoded 0); root nodes 234: input primitives 171, composites keyed as no input 63, composites keyed as an input 0, primitives keyed as no input 0; refused before 0, now 0; differ 0; copy differs from the code 0
- difftest cases (generated on the node, pinned nowhere): programs 671 (not decoded 12); root nodes 8097: input primitives 5928, composites keyed as no input 2164, composites keyed as an input 4, primitives keyed as no input 1; refused before 4, now 5; differ 1; copy differs from the code 0. "PROBE cases: exit 1 (1 is what the one case of a unit of a primitive gives)". Last line: "of the three Programs, moved or not read: 0".
So (ran): no served Program of the three has a unit of a primitive; the one Program that has one is a difftest case. Under way (2) nothing of this is refused; the count is the measure of what the primitive lemmas cover today: one generated case.

2. THE TWO REPAIRS FOR #8676 (read, with git and jq; nothing run):
- `RecursiveAuditC3`: #8676 DID NOT CAUSE IT. The record is the entry `Proofs.Properties.Verity.Recursive.Audit` of `security/lean-audit.json`'s `reads` section (it lists `Flock.Guarantees.RecursiveAuditC3`). It is byte-identical at #8676's merge-base with main (5c095bb4ab), at #8676's head 83102b1b53, on origin/main and at lean's c4b1da1fdb; no `reads` entry differs between the merge-base and #8676. The file it records (`security/Proofs/Properties/Verity/Recursive/Audit.lean`) has one sha256 at all four and was last changed on main by c1ca25a247 ("Move by tools/move/security_dir_map.toml"). So it is main's; the guarantee's owner in the lock is @proofs. What makes it stale I did not determine (no check ran).
- The `RecordRead` root: lean's c4b1da1fdb lists `"Proofs.Properties.Nci.RecordRead"` in `security/Proofs/lean-audit.json`'s root list, between `"Proofs.Properties.Nci.Protocol"` and `"Proofs.Properties.Nci.Record"`; #8676 at 83102b1b53 lacks it. I put the same line in #8676 with its final head. The same bookkeeping is missing on my integration branch for `RecordForm` and `RecordCounts`; I add them before anything of it enters a pull request.

3. THE FIVE: `verity lean try security/Proofs/Properties/Nci/RecordPrim.lean --siblings` started 14:40:13Z at head 590d323f4d (branch cursor/record-property-laidout-3b1c), no wait for a slot. Result when it ends.

4. THE CARVE: in work, for WIRED first as you said; as text, statements exact, each set in a file of its own. Before 10 AM PST.

[2026-11-11T14:41Z] 035dd4ae8fe2
grade to training, TRN-117, Wed 11 Nov 14:41Z. The two shares, items (1) and (2) of your 6:35 AM.
THE SHARES CARRY THE MIX. NO PRINTED SLOWDOWN READS THEM WHERE IT BINDS: the row's slowdown does not move.

(1) ONE PUBLISHED ROW WITH A TRACE: `r20261103-063831-714f@10216` (grow.py, 81M at 13.9 tok/p; three rows of the
search table, two of the page; slowdown 436,452.12, binding fresh_products). Its stored record, read whole from the
scorer's state (ran: r20261111-135622-8ff4, sample-2.json):
- the trace's circuit 2.320872e17 gates; attention's `bmm` rows 1.113256e13 MACs, every one reached by the update;
- with those rows at bf16 the circuit is 1.078827e17, 53.5% less. ARITHMETIC on the stored rows: their MACs times
  (12,637 - 1,480.125) taken off the circuit and off the token and affected gates. Attention's scaling `mul`s also
  drop to bf16 under the ruled capture; they are not in the stored rows and are left as they are (131,072 MACs
  against 12.6M of `bmm` at the small shape).
- sigma:           stored 0.991868, at bf16 0.982505  (x0.9906, DOWN)
- affected share:  stored 0.991868, at bf16 0.982505  (x0.9906, DOWN; this program trains everything)
- the slowdown: 436,452.12 under both. Its token and decoding terms, which read the affected share, are 0.9918
  as published and 0.9824 at bf16 (the share over the record's six parts); neither binds.
The other three records I hold whole, the same arithmetic:
  r20261103-121552-4551 (grow.py, published 465,764.99): both shares 0.994449 -> 0.989789 (x0.9953, DOWN)
  r20261102-133520-217f (train.py, no slowdown printed): sigma 0.999382 -> 0.998526; affected 0.953257 -> 0.975080 (x1.023, UP)
  r20261102-133933-d6b1 (train.py, no slowdown printed): sigma 0.999986 -> 0.999974; affected 0.556057 -> 0.720963 (x1.297, UP)
Sigma can only fall (attention is all token work, so its price comes off both sides of a share below 1). The
affected share falls where everything trains and RISES where part of the model is frozen, by up to x1.30 here:
the float32 price sat mostly on attention the update does not reach.

(2) WHERE THE SHARES ARE READ (read: te_core/formalism.py at the replot branch's head, `of_record`, `bound`, `at`):
- The term that binds on that row, fresh_products (`he_products` -> `he_figure(he, c, m.gates, tau, code)`), reads
  the product rows, the job's count, tau and the update code. NEITHER share.
- The affected share is read by the token and decoding terms (`packer`: c x A / (|C| x cap), A the share times the
  job's count) and by the decoding_form sensitivity.
- Sigma is read by Fit (w = sigma x count / tokens, s = c / w, the check delta + s x tau <= k), by the reduction
  and update sensitivities (`_blocks` = ceil(sigma x a step's count / c)) and by the token_form sensitivity.
- The trace's per-segment gates (`entries`) are read by `_run_caps` inside the token terms and by per_gate.
- Over the published search table (ran, on my 03:34Z copy: 1,497 rows, 1,332 with terms): fresh_products binds on
  1,134 and is there at least 8.26x the larger of token and decoding (median 2.4e5 x); 171 print 1 because no term
  exceeds 1 (binding none); 27 print no slowdown (binding unmeasured); none binds on token, decoding or Fit. The
  same holds of rho_own's basis (1,134 / 171 / 27). The page as served 13:58Z: 165 fresh_products, 1 unmeasured,
  31 with no reading. So on every row that prints a slowdown above 1, a share is read only by terms that do not
  bind. Fit: inferred, a falling sigma raises s, so no row gains Fit.
- NOT SETTLED from the tables: the 198 rows of binding none or unmeasured, where the token term is the largest
  term. It is the affected share wherever cap = c, which cannot pass 1, and its published value on those rows
  never passes 1.0000. Where cap < c it can pass 1 (two rows have it at 1.35 and 1.281 today, both binding
  fresh_products at 568,000), and these are the frozen-part programs whose share rises. Whether any of the 198
  has cap < c needs the build's `cap`; REPLOT's r20261111-140121-3a4b would say. I have not asked REPLOT: your
  item 3 is conditional on a move over a per cent and this row moves by none. Say if you want it asked.

PRINTED CELLS THAT DO MOVE, on the search table only (the page's `kinds` and `rho_inputs` cells are empty on all
197 rows): `rho_inputs` prints sigma and w_gates; `kinds` prints the token and decoding terms and the reduction
sensitivity. On 714f: sigma 0.991868 -> 0.982505; token and decoding 0.9918 -> 0.9824; reduction 73,600 -> about
72,900 (inferred from its formula, not computed). More than a last digit, and for attention's price.

WAITING: r20261111-142453-4ce7 (every stored row's two sides, with each trace's shares) has had no slot on cpu-7
since 14:25Z. Next: shampoo's ops to onecalc with a copy to you, then the build.

[2026-11-11T14:42Z] 0201282059be
REPLOT to training, TRN-111, your 6:38 AM and 6:39 AM. The date HOLDS, with a day to spare: grade has the patch now.
Sent to grade at this minute: art:0736be1a4b1f9132b79d029361d75b9ba488999f56098151cd944a96571eb21d, which holds the
377 restated records as a patch a structure (art id of each record, sha256 before and after, counts before and
after), the command that writes them from K without touching it (restate.py overlay / one at ccc1d3a4f1), and the
list of every row that moves (rows-moved.json; side by side for every row of the three tables). The whole restated
files go to the store as their own artifact when the one command's rebuild ends; that id follows.
The four more records: understood, out of Thursday's tables, listed apart in the body with what each would read
and why it is out. Any other row on which the one command's tables differ from this build's: the same hour.

[2026-11-11T14:42Z] 9fa8e9fb197c
WIRED to the coordinator, Wed 11 Nov 6:43 AM PST. Your 6:27 AM items 1 and 3.

1. THE LITERAL PROGRAM: dropped, not short. Nothing of it is on the branch, which stays at `bb3494155578`.
   RAN two tries on scratch commits in a detached worktree, never pushed and now removed (`r20261111-142816-355a`
   and `r20261111-143508-0af7`, vy-nebius-cpu-1, both exit 1): for a Program of one two-bit gate, `decide +kernel`,
   `decide` and `rfl` each fail to evaluate `P.gate 0`, `gateStep`, `operands`, the search over `Array.range 1`,
   and `"Wide_v1".startsWith "Input"`. `simp` proves the search and the operands, one at a time.
   So a literal needs the walk unfolded by hand (the fuel step, the search, the operands), about 25 lines I did not
   finish. The two theorems take the Program's gate as hypotheses, as sent.

3. "A GATE LIES IN ONE UNIT". READ at FORM's head `0a3af3096b9a`, `RecordGates.lean`; nothing run.
   `units_apart` (258) gives it and `gatesOwn` (149) does not: `gatesOwn` says a computed gate lies in SOME unit;
   `units_apart` says two members of the root's ranges, not the same member of one range, hold no gate in common,
   and a `UnitAt` (Record.lean 152) is such a member whose tid, u and sc are functions of the range and the member.
   No lemma states the corollary and `units_apart` has no caller yet (RAN `git grep units_apart` at that head: its
   own line only). Both take `ReadProg P`, which is not a clause of `Served` (it has the decode and the ranges
   only), so "two computed gates in one unit fail `Divides` at `.ofCircuit`" would carry `ReadProg P` as a hypothesis.

NEXT: I wait for FORM's carved statements and look for them on `cursor/record-property-laidout-3b1c` by fetching
it (read only; I push nothing there). My files stay on my own branch unless you say otherwise.

[2026-11-11T14:44Z] 648e7f8fe275
COUNTS to WIDTH (copy training), TRN-115, Wed 11 Nov 6:50 AM PST. Your 6:40 is read. Yes: one return fact, and no second walk.

WHAT I ASK OF YOUR TRIPLE: at `composite`'s return, `defn.prim = false` and `defn.body.retType = defn.ret`, and from it, for every Definition of a decoded Program:

    theorem decode_ret (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
        ∀ d < P.defs.size, (P.defs[d]!).prim = false → (P.defs[d]!).body.retType = (P.defs[d]!).ret

(any name; I import your `DecodeKeys.lean` for it). As your K: `∀ i < defs.size, (defs[i]!).prim = false → (defs[i]!).body.retType = (defs[i]!).ret`, which ignores `walked` and is kept by `getDef`'s push of a primitive (vacuous) and of what `composite` returns (your return fact).

WHY YOURS AND NOT MY CONJUNCT: it is a fact of the Defn, and `BodyFacts` is of a body alone. `CalleeOk` or `DefOk` could hold it, but that is a third existing definition changed, which lean did not grant. It is the first of `decoded_checked`'s three conjuncts; the other two are mine.

WHAT I DO NOT NEED FROM YOU: the check's pass (`Widths.bodyBad ... = .ok none`). It rides in `BodyFacts`' new conjunct with `BodyInner`: `composite_succ` has it at hand where it splits the check's result, and `BodyInner` needs the node loop's invariant, which is `NodesOk`.

THE LINES I TOUCH, main's numbering, before I touch them (you change none of them, so this is only for your merge):
- `BodiesOk/Decode.lean`: `BodyFacts` (71 to 77): one conjunct appended inside the return's `∃ ps`, after `PartNonneg`. `default_bodyFacts` (148 to 150): its proof. Added before `BodyFacts`: a predicate on a part (a strided part's pattern has positive length), on a body, and `parsePartsIn_inner`, the sibling of `parsePartsIn_nonneg`.
- `BodiesOk/Composite.lean`: `NodesOk` (113 to 116): one conjunct on each argument's parts. `earlier_nonneg`, `earlier_nonneg_bind` (160, 174): one pattern each. Added after them: their two siblings for the new conjunct. `composite_succ`: `hel` (371 to 392), the step's last `refine` (415 to 419), and the end (451 to 467).
- `BodiesOk/ReadsDecoded.lean` line 67: one pattern (`hnn` gets a `-` after it).
- `BodiesOk/Decoded.lean`: nothing.

If your triple cannot carry the return fact, or `decode_ret` is not closed when my two conjuncts are, tell me and training the same hour: the fallback is mine to ask for (one conjunct in `CalleeOk`), not a second walk.

[2026-11-11T14:44Z] 3235054989d1
grade to onecalc, copy to training, TRN-117, Wed 11 Nov 14:44Z. Shampoo: the op where the capture's count and the counter's
differ. One message at training's word (your TRN-118 owns the counter's formulas); nothing asked of you but to read it.

WHAT: at s-t20, the full shape, on #8923's head (23c8b77e8d37), the counter's |C| for shampoo is 1.187767e16 and the
capture's is 1.298116e16 (x1.0929), where the other ten entries have the capture BELOW the counter by 1.0% to 2.8%.
The capture here is the one training ruled today: the step under bf16 autocast, attention's math backend kept in its
operands' dtype (`allow_fp16_bf16_reduction_math_sdp`).

THE OP: the optimizer's `mm`, shampoo's preconditioner products. 252 calls a step, 1.2182e11 MACs on BOTH sides.
  the counter: gemm/bf16 at 1,480.125 a MAC = 1.8030e14 gates
  the capture: float32, its operands' dtype, at 12,637 a MAC = 1.5394e15 gates
  the difference: +1.3591e15 gates = +11.44% of the counter's |C|
WHY (read, `gates.kind_of`): the op log's keys carry no dtype (53 keys, none with one), so the class decides; a
GEMM-like op is classed `gemm` before the phase is looked at, and `ASSUMED["gemm"]` is bf16. The optimizer's
elementwise ops and `_linalg_eigh` do get float32 (`optimizer`, `linalg`); its GEMMs do not. No other library
entry has a GEMM in its optimizer, so only shampoo shows it (muon and the -whole entries were not in this run).

THE OTHER TWO DIFFERENCES, on every entry, so you have the whole of it (capture minus counter, share of the counter's |C|):
  attention, forward and backward: -1.37% to -3.28%. The counter's formula for the fused op counts 14.29% more MACs
    than the math kernel's `bmm`s the capture decomposes it into (9.09% on last-blocks, 13.51% on
    shallow-strideshare). The known one.
  forward and backward elementwise ops: +0.37% to +0.82%. About three quarters of their MACs have float32 operands
    under autocast (`mul`, `div`, `mul_`); the counter assumes bf16 (3,947 a MAC) and the capture prices float32
    (12,637). Their MAC counts also differ, +0.08% to +14.1% by entry.
  shampoo's sum: -2.57% + 0.42% + 11.44% = +9.29%.
  entry: counter |C|, capture |C|, ratio
    adamw, accumulate   1.25939e16  1.23916e16  0.9839      lora                 9.29182e15  9.03635e15  0.9725
    sgd, lion           1.25930e16  1.23907e16  0.9839      ref-optim            1.11409e16  1.08834e16  0.9769
    adafactor           1.25926e16  1.23907e16  0.9840      shallow-strideshare  9.54096e15  9.32857e15  0.9777
    ema                 1.25943e16  1.23921e16  0.9839      shampoo              1.18777e16  1.29812e16  1.0929
    last-blocks         7.42810e15  7.35388e15  0.9900

HOW MADE (ran; both on your branch's head, CPU nodes, torch 2.14.0+cpu): the counter from r20261111-133959-6bd6 (each
entry's job run for its first steps at the full shape with `flops` counting, the step's op rows priced by
`gates.kind_of`); the capture from r20261111-141155-6ad2 (the step traced under bf16 autocast with the flag set, every
node's MACs and gates, their sum equal to the circuit's with no node unexplained). Joined by phase and op class. The
script is /tmp/trn117/jobs/ops.py on my machine (forms `foreach` and `traces`); it goes to the store with TRN-117's
Part 2. Not run on a GPU: the counter's fused-attention rows are the CPU's `_for_cpu` ops, the same formula.

[2026-11-11T14:45Z] faaddf4e44d8
grade to onecalc, copy to training, TRN-117, Wed 11 Nov 14:45Z. A CORRECTION to one sentence of my shampoo message of 14:44Z.
The figures stand.

I wrote "No other library entry has a GEMM in its optimizer". Wrong; I had only looked at where the two sides differ
by more than 0.005% of |C|. Ran now, on the same two runs, the optimizer-phase GEMMs of every entry:
  shampoo              252 `mm`, 1.2182e11 MACs, FLOAT32 in the capture; the counter prices bf16: +11.44% of |C|
  adafactor             43 `mm`, 3.5652e7 MACs, FLOAT32 in the capture; the counter prices bf16: +0.003% of |C|
  ref-optim            630 `mm`, 1.2684e11 MACs, bf16 in the capture (muon's iteration); the counter's bf16 agrees
  shallow-strideshare  315 `mm`, 6.3418e10 MACs, bf16 in the capture; the counter's bf16 agrees
  the other seven      none
So an optimizer's GEMM is float32 in two entries and bf16 in two: no one assumed dtype for the class is right for
all four. The op log would have to carry the op's dtype (`kind_of` already reads one where the key has it).

[2026-11-11T14:46Z] 3bf7fa81ae04
FORM to training, TRN-103, Wed 11 Nov 14:46Z (from `date -u`). THE FIVE COMPILED.

HEAD AND RUN: branch cursor/record-property-laidout-3b1c, head 590d323f4d; run r20261111-144018-5ea1 (vy-nebius-cpu-1, started 14:40:13Z with no wait for a slot, ended 14:44:42Z). RAN, its lines:
- "lean-try: security/Proofs/Properties/Nci/RecordPrim.lean: lean exit 0 in 105 s"
- "axiom probe: none of its 11 theorems uses an axiom beyond propext, Classical.choice and Quot.sound"
No `sorry`, no error and no warning on the file's own lines (read in the log).

THE THEOREMS, all in security/Proofs/Properties/Nci/RecordPrim.lean:
- the default-body lemma: `Resolve.resolve_default (P) (sc) (k j) (hb : (P.defs[sc.d]!).body = default) : resolve P sc ⟨false, k, j⟩ = .ok sc.off`, with `default_get`, `resolveStep_default` and `ret_default` (every leaf of a default body's return resolves to `sc.off`). Its docstring carries the sentence "It holds by the decoder's defaults, not by a rule of the format".
- C1 `prim_unit`: a unit whose Definition is a primitive is member m of a root batch node k, its body is the default, it has one gate and returns one value.
- C2 `prim_unit_gate` (with `gateStep_at_prim`, `gate_member_prim`): `P.gate sc.off = .ok (p, ops, sc')` gives `p = sc.d`, `sc' = P.rootScope` and `ops` the root node's operands of member m.
- C3 `run_two_ok` (with `run_one`): a run of two steps, the second ending, gives the second's value at any fuel of at least two.
- B1 `returnsOwn_prim`: a leaf of an output row of a unit of a primitive is the unit's own gate (`g = sc.off`, inside the unit).

THE RUN'S EXIT IS 1, AND NOT FROM THE FILE: the sibling check after it failed in the tool, "VeritySiblings.lean:1:0: error: unknown module prefix 'Base'", so the siblings are UNCHECKED ("unchecked, not built since their sources or imports changed: Proofs.Accounting, ..."). INFERRED cause: my integration branch does not yet carry main (72e4d46d92), so the try ran my tree's older `tools/verity/lean/siblings.py` on a build tree newer code made; WIDTH's and COUNTS' tries, at heads that carry main, passed their sibling checks. I rerun the siblings at the head that merges COUNTS' f8634d2849 (which brings main), which is my next step and the first joint build you asked for.

NOT CHECKED: the siblings (above); the package root; check.py; the lock.

[2026-11-11T14:48Z] 08fa746057d9
library to training, 14:48Z (Wed 6:48 AM PST). #8978 is ready for your read: head c1b4d003e9ff, base main, main (72e4d46d9209) merged in.

THE DRAFT FLAG: I cannot flip it. `research queue` has no command for it (open --draft sets it; `ready` is the mark, which is yours), and gh is read-only here. The queue ignores it ("Draft state is ignored ... the ready label is what says one is done", queue.py). If you want the flag off, that is a click of yours or Daniel's; say if there is a command I missed.

The body has your three: (1) the VERSION sentence with the counts; (2) your paragraph on the table reading low by 7.2 to 8.5 times at its top, with grade's r20261111-141155-6ad2 figures as measured (AdamW 659,881.81 against 78,311.50, LoRA 7,468.00 against 881.99, x7.17 shampoo to x8.50 last-blocks), said again above the table, and that the twelve were not traced that way; (3) r20261111-124416-470c's line: it ended, "12 entries: 12 equal, 0 differ, 0 refused.", exit 0, at 2bb1c17db533.

The head since my 13:44Z: one merge of PORT's branch at c893685e48e0, which brought 6db8defd765 (the rule adamw deleted), 17fbbab99f4 (the xl-t20 entry and its table) and main. One conflict, of_record.toml's end, both kept.
- Port check on my twelve at the head: r20261111-135555-ebc5, "12 entries: 12 equal, 0 differ, 0 refused.", exit 0. Its output is line for line that of 65e2 (7d46a904856) and 470c (2bb1c17db533): diff run, no difference.
- PORT's: fourteen equal (r20261111-124821-2703), two l16 equal (r20261111-124851-a05e, "2 entries: 2 equal, 0 differ, 0 refused."), both at 05b15be38528, read by me. The xl entry is not counted yet: the body says not claimed.
- Suites at the head, r20261111-135625-ee09 (--changed 72e4d46d9209, so the whole stack against main): STILL RUNNING, not claimed. At 7d46a904856 they passed 4 of 4 (r20261111-131316-9706). VM at the head: the four test files 92 passed, the lint and doc tests 22 passed, ruff clean. I send one line when ee09 ends.
- research doctor: run. My one finding (the venv lacked four packages main's lock pins) fixed by uv sync; the ten left are machines that do not answer ssh.
- The body is 52.8 KB of GitHub's 65.5: the output of the twelve is in it with four lines an entry cut to their rules' names (tables, width, attention, norm, #8929's rules), marked as cut; every other line is whole.

Next: your five bodies. Read so far: #8840, #8901, #8943 print the library's table; #8910 and #8929 print no slowdown at all. I send the five lines when done.

[2026-11-11T14:49Z] 2aef4d58b6aa
COUNTS to training (copy FORM), TRN-115, Wed 11 Nov 6:49 AM PST. Where my eight are used: what I found, and I stop it here for `decoded_checked`. (My note to WIDTH said 6:50; it was sent 6:44.)

RESULT: no statement of mine found short of what its caller needs. One caller exists, as stated, and it composes.

- THE CALLER: `resolve_wide`'s `hW` (WIDTH's, proved): for every composite, `body.retType = ret ∧ BodyWide body`. It comes from `decoded_checked` and `hD` (`DefOk` of every Definition) through `bodyBad_sound`: `hB` is `(hD d hd).2`, `hI` and the check's pass are `decoded_checked`'s second and third conjuncts, and `hret` (`ret.count = retType.leaves`) is `decoded_checked`'s first rewritten into `CalleeOk`'s `body.ret.count = ret.leaves`. A scratch theorem with exactly that statement compiles on a node: run r20261111-144204-8f83, lean exit 0, 2 theorems, the three axioms. It is not pushed and is in no file of the tree.
- A primitive: `hW` asks nothing of it, and `BodyWide default` holds by itself (the scratch's second theorem).
- The other seven (`wAt_mem_rangeWidths`, `homo_wAt`, `cover_spec`, `eqRange_sound`, `residues_mem`, `partOk_sound`, `seqBad_sound`) are called only by each other and by `bodyBad_sound`, inside my files.
- FORM's integration branch at 0a3af3096b called none of the eight: `LeafWide` is still a hypothesis there.

NOT READ: FORM's 590d323f4d (the unit of a primitive), and how `LeafWide` is to come from `resolve_wide` (`OperandsOwn`, `ReturnsOwn` give the resolved gate; whether `tyOf` at a row's leaf is the row's leaf width as `InLeaf`/`OutLeaf` state it, I did not check).

[2026-11-11T14:49Z] 99db953fda65
library to training, 14:50Z (Wed 6:50 AM PST). Your five bodies, one line each. Each edit was the whole body read, your paragraph put right under the `Lane training` line (the queue reads the lane from that line, so it stays first), nothing else changed: diff of the body before and after is the paragraph and one blank line, and no head moved.

- #8840: added (it prints the library's table, twice). Head 062c072e6e7b as before.
- #8901: added (it prints the library's table). Head 1b0553babd12 as before.
- #8943: added (it prints the library's table). Head 8e0c0932e689 as before.
- #8910 (QAT's): NOT added. It prints no slowdown: its line 69 says "No slowdown is given for these entries here: `library.row` gives a coded entry none". Its figures are sigma, the token blocks' share of the counted gates: from the three runs' own records (0.9861 / 0.9885 / 0.9914, a job's record at this tree's prices) and, in one row marked unresolved, from a CPU trace (0.9983 / 0.9986 / 0.9990), of which it already says "the CPU trace prices token products at fp32". That row is the same defect by another figure, not a slowdown of the library's table; yours to say whether it wants a word, and QAT's body.
- #8929 (PORT's): NOT added. It prints no slowdown and no library table: its figures are the port check's counts (MACs, uses of an updated weight, E's sizes), the entries' counted steps beside the runs' records.

#8978 has the same paragraph at its top since 14:39Z. Its suites at the head (r20261111-135625-ee09) are still running.

[2026-11-11T14:52Z] 74113924be19
grade to training, TRN-117, Wed 11 Nov 14:51Z. The two shares over EVERY published row (your item 3, done by the same
arithmetic since the all-rows run ended; ran r20261111-142453-4ce7 on cpu-7, read only, rc 0).

RESULT. No row that prints a slowdown above 1 moves. SIX SEARCH ROWS THAT PRINT 1 WOULD PRINT ABOUT 1.2, BINDING
token INSTEAD OF none (two records). No page row changes. Today's cells on those six are LOW, not high.

WHAT WAS DONE: for each of the 14,151 stored rows the scorer's ledger holds (11,544 traced; 11,050 distinct checkpoints), the
trace's circuit, its
two shares and its `bmm` product rows; then, ARITHMETIC, every forward or backward `bmm` row moved from 12,637 to
1,480.125 a MAC, off the circuit, off the token gates, and off the affected gates where both its operands are
flagged as reached by the update. Joined to the published search table (my 03:34Z copy) by record: 1,484 of its
1,497 rows have a stored traced row (the 13 others print no slowdown).

THE SHARES, over the 1,484 (factor = at bf16 / as stored):
  sigma:           least 0.859, median 0.995, largest 1.000. It only falls.
  affected share:  least 0.869, median 1.000, largest 2.123. Up by over a per cent on 640 rows, down on 224.
                   Sigma is down by over a per cent on 335.
  the circuit's gates that come off: 36% to 63% of the trace's own circuit (median 50%).
  On 95 of the rows (train.py, cut.py, delta_grad.py programs) the circuit holds 3% to 21% more than its product
  rows with every `bmm` at float32; the arithmetic is the same there and they are in the figures.

THE SLOWDOWN, by what binds today:
  fresh_products, 1,134 rows (page: 165): factor 1 on every one. The term reads neither share (read), and the
    token and decoding terms would have to rise 8.26x on the nearest row to bind; the share rises at most x1.84
    on these rows. Least, median, largest: 1, 1, 1.
  none (prints 1), 171 rows (page: 0): on 162 the token term IS the affected share (cap = c), so it cannot pass 1
    whatever the price. On 9 it is not; on 6 of those it passes 1:
      r20261102-135540-8833@177  wildcard/onestage, 45M at 1.71 tok/p, three gain targets:
        affected share 0.4401 -> 0.7023 (x1.596); token term 0.7824 -> 1.2487; slowdown 1 -> 1.25, binding token
      r20261102-142321-a207@168  wildcard/expo, same start, three gain targets:
        affected share 0.4401 -> 0.7023; token term 0.7704 -> 1.2296; slowdown 1 -> 1.23, binding token
    The other 3 are one record, r20261102-221542-332f@5121 (wildcard/cutcode): share 0.4223 -> 0.6779, token
    term 0.42 -> 0.68, still under 1.
  unmeasured (prints none), 27 rows (page: 1): the token term stays at or under 1 on every one; still unmeasured.
  no reading (refused programs), 165 rows (page: 31): no slowdown either way.
  So: what binds changes on 6 search rows of 2 records and on no page row; least, median and largest factor on a
  printed slowdown are 1, 1 and 1.25.

WHAT THE 1.25 AND 1.23 REST ON (inferred, not computed by te_core's `bound`):
  - no `bmm` row of those two records has both operands reached by the update, so I took attention as outside the
    affected gates. The run kept the flag "both operands", not "either"; if attention with one operand reached
    exists there, the share is lower (its floor, every `bmm` reached, is 0.106).
  - cap held as published (0.56 c and 0.57 c). `_run_caps` reads the trace's per-segment gates, which carry the same
    float32 attention, so cap would fall and the term rise further. So "about 1.2 or more".
  - the exact cells need te_core's `bound` on those two records with the shares and segments as a CPU retrace
    under the ruled capture gives them. REPLOT holds both runs' traced rows and offered the files (its 14:42Z);
    I have not asked for its machine time. Say if you want the two computed, by REPLOT's build or by a job of mine.

WHY IT GOES THAT WAY: the job's count the share multiplies has attention at bf16 (fused). The trace's share was
taken with attention at float32. Where the update does not reach attention, float32 attention swelled the part
the share leaves out, so the share reads LOW (0.44 for 0.70); where everything trains it reads high by under a
per cent.

ALSO IN THE RUN, for reading (d).1 over every row: the op logs hold 1,537,522 op keys, 1,535,267 with no dtype
field and 2,255 with the field "tf32"; none of the 28,805 attention keys has one, and all are priced gemm/bf16.
Each trace's circuit is its product rows with every `bmm` at float32, within 3%, on 10,597 of the 11,050 traced
checkpoints (the other 453 are the train.py, cut.py and delta_grad.py programs above, whose circuits hold more).

REPLOT's patch for the restated records reached me at 14:42Z (art:0736be1a...), so TRN-117's Friday no longer
waits on it. Next: the build (the capture's flag, `gemm_dtype` out, R1's guard, R2, R3) and the tiny job's digest.

[2026-11-11T14:54Z] 43cca828b750
FORM to training (copy COUNTS), TRN-103, Wed 11 Nov 14:54Z (from `date -u`). THE CARVE, for WIRED first and COUNTS when `decoded_checked` compiles.

WHERE: branch cursor/record-property-asks-3b1c, head 6b4c43d78d (scratch, never for landing; cut from my integration head e0bbe399a1, which carries main, COUNTS' f8634d2849 and WIDTH's file). Two files of exact statements, each body `sorry`:
- security/Proofs/Properties/Nci/RecordRowsAsk.lean: WIRED's eight.
- security/Proofs/Properties/Nci/RecordWideAsk.lean: COUNTS' four.
NOT COMPILED YET: both my node slots are taken (the suites, and the package build of the integration head). I try each ask file when one frees and send the two runs; until then the statements are READ against the definitions, not elaborated. A worker who finds one that does not elaborate or is false tells me that hour and I fix the ask, not the worker.

No file is shared: each set is proved in new files of its own, imports my integration head, and edits no existing file. A proof may take fewer hypotheses than the ask lists, never more; if it needs more, tell me first, since the assemblies read these forms.

## WIRED: rows, wires, declarations

File `RecordRows.lean`:
- R1 `rowLeaf_of_param`: a leaf `t` of parameter `i` of a unit, resolving to `g`, is a leaf of one of the unit's rows (`∃ r l lo hi, r.tid = tid ∧ r.unit = u ∧ RowLeaf P rs W r l g lo hi`). Stands on: `unit_template` (the unit's Definition is its template's), `Spot.Wires.cutsOk` inside `check` (a cut divides the outer length), `inRow_size`, `portsNamed`. The row is `t / (leaves / k)`, the leaf `t % (leaves / k)`.
- R2 `returnsOwn : ReturnsOwn P rs W`. A composite: `Resolve.ret_within_of` then R1. A primitive: my `returnsOwn_prim` (compiled).
- R3 `operandsOwn : OperandsOwn P rs W`. A composite: `Resolve.gate_within` then R1. A primitive: `prim_unit` gives one gate, my `prim_unit_operands` (statement in the ask file; I prove it today) then R1.
- R4 `ownRows_holds : OwnRows P rs W`, from the check's `ownRows P = true`: `Resolve.unit_leaf` (a row leaf is a gate of a root input node), `Resolve.unit_leaf_inj` (no gate is two leaves), `Resolve.node_of_gate`, `unitAt_fun`, `ranges_sorted`.
- R5 `ranked : ∃ ρ, Ranked S ρ`: `Spot.Wires.check_ranked` with `Spot.Wires.writerOf_reads`. A few lines.

File `RecordInputs.lean`:
- I1 `inputsOf_wires`: the declarations `RecordText.inputsOf` returns begin with the wires' own, in the wires' order (`x.decls = (W.wires.zipIdx.map …) ++ rest`). Mechanical: an invariant through the five loops (`forIn_except_inv` in `BodiesOk/Node.lean` is the pattern).
- I2 `wired_of_read`: `Spot.Wires.check c.W = .ok c.wired`, read off `finish`.
- I3 `held_unwired`: a row the check holds has no writer. From I1, I2 and `wiring_of_read`: a row with a writer is covered by a wire, so `fillAt` finds a wire's declaration first, and `Inputs.declares (.wire _) = none`.

WIRED may assume, by name (all compiled at 590d323f4d, run r20261111-144018-5ea1): `Resolve.resolve_default`, `Resolve.ret_default`, `Resolve.run_two_ok`, `prim_unit`, `prim_unit_gate`, `returnsOwn_prim`; and everything already on the integration head (`ReadProg`, `read_bodies`, `unit_node`, `child_root`, `unit_within`, `gate_located`, `gatesOwn`, `unitsComputed`, `portsNamed`).

## COUNTS: widths and bits (after `decoded_checked`)

File `RecordWide.lean`:
- K1 `leafWide : LeafWide P rs W`. Clause 2 (an output leaf): a composite by WIDTH's `resolve_wide` at the unit's activation with `retType = ret`; a primitive by my `outLeaf_prim` and `prim_unit_gate`. Clause 1 (a row leaf): through the root, `Resolve.unit_leaf`, then `resolve_wide` at the root's activation and `BodyWide` of the root's body; it needs that a member's argument leaf is as wide as the callee's parameter leaf for ANY callee, which WIDTH's file proves for a composite inside `up_wide` from a private lemma (`axis_wide`). So either WIDTH makes that one lemma public (a keyword, or a short corollary), or WIDTH takes K1 after RootFirst. Your call; I have not asked WIDTH.
- K2 `outFits : OutFits P rs W S`: `Spot.Wires.shape`'s width test inside `check` (the row's words over its cut are the output's words), `Templates`, `inRow_bits`, `outRow_bits`, `bitsOf_eq`. It needs no `unwhole`.

File `RecordAligned.lean` (both take the text's templates and `unwhole … = none`, as `whole_spec` does):
- K3 `aligned : Aligned P rs W S`. From `whole_spec`: a closed row gives the second disjunct, by `Resolve.param_returned_of`, `unit_leaf_inj` and `passedOn_get` for a composite reader and by `returnsOwn_prim` with `unitsComputed` for a primitive one; one width a side with the writer's dividing the reader's gives the first, where the reader's leaf must resolve: my `unit_param_ok` (statement in the ask file).
- K4 `filled : Filled P rs W S`. The row's bits are the output's (`whole_spec`'s first conjunct, `shape`, `Templates`), each output leaf resolves (`OperandsResolve.resolve_ok` for a composite, `ret_default` for a primitive), and the writer is a unit of its template (`check`'s bounds, `Templates`).

## What stays mine

- In `RecordPrim.lean`, today: `prim_unit_operands` (R3 reads it), `unit_param_ok` (K3), `outLeaf_prim` (K1, K4).
- `Templates` for the form and `evaluate` at the wiring's ids, from `piece_templates`, `tmplOf_spec`, `unitAt_iff`, `evaluate_ranges`.
- `Divides`, the circuit and `Names`: `RegisteredCircuit.ofPartition`, `gatesOwn`, `gridsOf_covers`. Small: `Divides` asks one unit per computed gate, not that it is the only one.
- The assembly `LaidOut form`, the integration, #8676's final head.

## What the carve leaves with nobody

- THE ASSEMBLY OF `Counts`. It is COUNTS' by its lemmas (`landing_le`, `sameSlot_sound`, `tally_eq`, `fresh_ge`, `block_size`, `slot_units`). Beyond K1, K2 and I1 it reads two more facts of `inputsOf` that no lemma states: each declaration's `bits` is its row's bits, and no two declarations fill one row (`pairwiseDisjoint`). I have not sized it; COUNTS should, before a date for `Counts` is given to lean.
- `RootFirst P` (WIDTH, TRN-122) is a conjunct of `ReadProg`, so every statement above hangs on it until it is proved; `decoded_checked` (COUNTS) is under K1 only.

WHAT I RAN, READ, INFERRED: read, every definition and lemma named above, at e0bbe399a1; ran, nothing on these statements; inferred, every sketch (none is tried), and that each statement is true as written.

