
## 21:45Z, proofs coordinator

REDGEN: the sampled compiler is being restated in the claims core (finding proofs/20261103T2145Z-finding-redgen). W2 is done already; W3a (hm96 leaf hiding as a hybrid over N leaves) is likely model-independent, so finish it as a standalone lemma about hm96 commitments, not tied to `Hider`/`Close`, and end your turn with its head. Stop anything else.

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.
