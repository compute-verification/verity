---
name: smell-check
description: >
  How to run a smell check in Verity, and the catalogue of smells found so far: work that reads like what Daniel asked for
  but isn't, such as credit for a side computation beside the served one, a "served" or "end to end" number that came from
  a model or a harness, a rule-following path that is opt-in, numbers that don't compare, one value committed twice, a
  check that passes for the wrong reason, and a claim relayed from another lane without reading what backs it. Each entry
  quotes the words that hid it, says what it bent and how it was fixed. Use when Daniel or a lead asks for a smell check,
  before you call something served, end to end, proved, exact, measured or credited, when you relay another lane's claim,
  when a result looks too good, when you review a brief, or when you found a smell and want to record it.
---

# Smell check

Daniel sees the work only through what agents tell him, so he can't ask about a shortcut he doesn't know exists. On 19 Oct
the served Dither window turned out to credit receipts computed in a torch pass beside vLLM's own int8 GEMM. Its lane
called it "served", kernels relayed that to him, and he found out only by asking a different question: "the smelliest thing
I've heard all day". A smell check is how agents find these first. Daniel's ruling (18 Oct, 10:31 PM PDT): he asks every
agent for a smell check from time to time, and each one reports what it finds and adds it to the catalogue below. Run one
too whenever a claim you're about to rely on or pass on carries one of the words below.

## Run one

1. **List the claims.** Write down what you were told (your brief, other lanes' posts, PR titles and bodies, labels) and
   what you told others (reports, cards, notes, labels), and mark every claim that uses one of these words: *served, end to
   end, measured, proved, exact, bit-exact, credited, committed, zero-knowledge, hiding, live, landed, pinned*.
2. **Read what backs each one, not its summary.** For code, the path that actually runs on the served or proved path
   (follow the call from the entry point; a docstring says what was meant). For a number, the run's own outputs and
   labels. For a theorem, its statement and the definitions it reads. Ask each question in "What to ask" below.
3. **Ask the one question that matters:** would Daniel, reading the code and the run rather than the summary, recognize
   it as what he asked for? If you'd have to explain a "but", it's a smell.
4. **Report every smell you find,** yours or another lane's, in the format below. A smell in another lane's work goes to
   that lane's owner too. Report "none found" with the list of what you read, so the asker can see what wasn't.
5. **Act on it.** Fix your own smells without asking (the friction skill's 6 Oct ruling). Send another owner's to that
   owner. One that changes what a claim says or rests on goes to Daniel (`.agents/skills/asking-daniel/SKILL.md`). Relabel
   what's already published (`research data label`), and correct anything you told Daniel in the same report.
6. **Add it to the catalogue** below: one entry, newest first, in the PR that fixes it or in a one-line PR of its own.

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

- **Is what's credited or proved the thing that was served?** A second kernel beside the served one, a side computation,
  a stand-in model, or a reference pass in place of the real one means the claim is about something else.
- **Does "served", "end to end" or "measured" mean the real path?** Pinned build, compiled, the record's settings, the real
  workload, interference included; not a harness, a model, an unpinned build or a fixed batch quoted as max throughput.
- **Does the record path follow the design rules, or is the rule-following path opt-in?** One hash, hiding leaves, live
  coins, everything committed at serve time, everything zero-knowledge.
- **Do the numbers compare?** Same baseline, metric, model, card, build and scheme; the scheme named in the label; a model's
  number labeled as one.
- **Is anything done or kept twice?** Two GEMMs for one linear, two trees over one value with nothing tying them, two
  implementations of one format.
- **Does each check fail when it should?** A negative test refused for an unrelated reason, a theorem made vacuous by a
  definition, a gate that can't fail, a test that reads only its own fixture.
- **Is anything deferred that the claim needs now?** Commitments, proofs or checks promised for later, while the claim is
  stated as if they had happened.
- **Did you relay it without reading it?** A claim from another lane's summary is a quote, not a fact, until you've read
  what backs it.
- **Is there code on the soundness path that nothing runs?**

## Catalogue

Newest first. Add yours.

- **Dither credited a side computation, not the served GEMM** (19 Oct; found by compute-accounting's W2 from the code,
  CMP-50, after kernels' question J11).
  - What it was: SERVE-H1's window ([#2639](https://github.com/compute-verification/verity/pull/2639), draft) serves
    through vLLM's int8 GEMM (`torch._int_mm`), runs a second `_int_mm` for z beside it, and computes the H1 receipts in a
    torch pass that took 457 s a window. Credit counted that pass, so the "useful work" wasn't the served computation, and
    serving ran two GEMMs.
  - Quote: its receipts.py docstring, "as a side computation on a served int8 call ... z, the plain A Bᵀ cell, is the
    served y's integer product and no part of this computation"; the lane, "Dither (H1 + fx1) *is* served now"; kernels to
    Daniel, "Dither is already served in vLLM".
  - Bends: PoUW credits the served work.
  - Fix: one fused H1 + fx1 kernel produces y and z and commits z (kernels-dither-rtx); compute-accounting's credit gate
    refuses a unit whose linear ran a second GEMM; the runs are relabeled "H1 receipts beside the served GEMM; not
    useful-work credit".
- **Negative tests refused for the wrong reason** (19 Oct; compute-accounting's joint note, its open item 13).
  - What it was: all three of W2's negatives (a flipped z cell, a wrong receipt word) were refused with "S2/R7: the
    session parameters are not the verifier's", so none reached the check it was meant to test.
  - Fix: rerun with the verifier's session parameters; both are now refused inside the proof (r20261019-045751-1c6b,
    [#2675](https://github.com/compute-verification/verity/pull/2675)).
- **An overhead card resting on units nobody had agreed to** (18 Oct; Daniel).
  - What it was: kernels' recommendation card v1 priced the commit at per-layer units, with a model that left out
    interference with serving. Proofs' optimizer cuts proof units per kernel call.
  - Quote: Daniel, "What do you mean per model layer instead of per kernel call -- I would talk to Proofs about this."
  - Bends: overhead is measured on the served path.
  - Fix: kernels' only overhead number is the pinned served_commit loop, end to end, one change an iteration, plotted from
    the store; the red team then voided card v2's served-overhead cells (note:hash/20261018T2024Z-finding-redteam-numbers).
- **Committing less at serve time than the claim needs** (18 Oct; Daniel).
  - What it was: a plan to commit only the replay blocks' inputs and outputs at serve time and the proof units' later,
    though PoUW's work blocks must be committed while serving.
  - Quote: Daniel, "the kind of like, Lazy commitment scheme where, like, at serving time, we don't commit the IO of the
    proof blocks, only the IO of the replay blocks. This is like basically not going to work."
  - Fix: everything is committed at serve time (ASK-25), and the optimizer's cost counts every serve-time commitment.
- **The record path wasn't the rule; the rule-following path was opt-in** (18 Oct; kernels' baseline).
  - What it was: the GLM-5.3 Commit of record used vllm-v1, SHA-256 position leaves with no hiding salt; hm96-sha512 rows
    were an opt-in arm (rows-sha512). Every overhead number on the record path understated the real one about 4x (262 vs
    1,090 ms a Qwen3-8B run).
  - Bends: one hash, everything zero-knowledge.
  - Fix: circuits moves the record to rows-sha512, and kernels measures only that.
- **Quoted numbers from a retired scheme** (18 Oct; kernels).
  - What it was: two Pearl-C windows quoted tonight (3.097x and 1.630x) were the retired BLAKE3 -h2 scheme, and the
    lanes' overhead metrics disagreed: "% of compiled throughput", "% of model-kernel GPU time", "x graphed stock FP8".
  - Fix: the scheme goes in every run's label; the loop reports TPOT and throughput against pinned compiled stock.
- **One value committed twice, by four implementations of one format** (18 Oct; kernels).
  - What it was: a linear's input was committed by the Commit (BF16 rows) and by Pearl-C (FP32 row segments, its own
    tree) with nothing tying the two, and four lanes each kept a SHA-512 kernel for the same hm96-sha512 format.
  - Fix: kernels' shared library for the format; compute-accounting's A-BIND opens Pearl-C's A from the Commit's table.
- **Dead legacy kernels on the soundness path** (18 Oct; kernels).
  - What it is: `verity/ml/kernels/hash_gpu` holds Poseidon2, BLAKE3 and SHA-256 Merkle kernels from archived B-Ligero,
    imported by nothing live but inside a unit core may import.
  - Fix: proofs to archive them (not done on 19 Oct); prover-side kernels live outside `verity/ml/kernels`.
- **A theorem made vacuous by a definition** (from the process-design skill).
  - What it was: #118 redefined `MerkleScheme.Collision` and made `merkle_binding` vacuous; the axiom audit passed it.
  - Fix: guarantees' records in the lock, and a person reading the changed record.
