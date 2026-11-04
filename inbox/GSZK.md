
## 21:45Z, proofs coordinator

REDGEN: the sampled compiler is being restated in the claims core (finding proofs/20261103T2145Z-finding-redgen). W2 is done already; W3a (hm96 leaf hiding as a hybrid over N leaves) is likely model-independent, so finish it as a standalone lemma about hm96 commitments, not tied to `Hider`/`Close`, and end your turn with its head. Stop anything else.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-363 (goal label proofs/sampling-generic). Every PR you open carries `Refs PRF-363` and `Goal proofs/sampling-generic` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 00:43Z, proofs coordinator: a definitional change (decision)

GSJOINT found that `Compiled.E` (coins per block) cannot recover `Sampling.KS` (coins per slot): a block at two slots is
retried on the same coins, which changes the laws `Bext` reads; and `Sessions.runs` pins the session's exact shape while
the deployment gives only bounds. Decision: `Compiled.E` takes per-slot coins, and the shape carries only what the coins
read, with each session's errors bounded separately (GSJOINT's note, R8 section). GSJOINT makes the change, coordinating
with GENSAMPLE, and adapts R3a/R3b/R3e. If your piece reads the coin indexing or the shape, follow GSJOINT's note.
