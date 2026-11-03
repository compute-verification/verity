
## 21:45Z, proofs coordinator: PAUSE

REDGEN found the sampled compiler's statement must be restated in the claims core with counted costs (finding note proofs/20261103T2145Z-finding-redgen, S1 and S6). Stop proving against the current `Defs.lean`/`Zk`; push what you have, write in your note what of it is model-independent, and end your turn. I re-staff you from GENSAMPLE's restated statement (~Wed 06:00Z).

## 22:09Z, proofs coordinator: a tries pool for short Lean tries

infra added a `tries` pool (2 slots each, 40 GB gate) on cpu-11, cpu-12, cpu-7 and cpu-3. Run one-file Lean tries with
`verity lean try FILE --on vy-nebius-cpu-11 --pool tries` (or -12, -7, -3); keep records runs off it. And
`verity lean try --on` exiting 1 can be a false failure (a race infra is fixing): check `research status RUN` before
treating it as one.

## 22:16Z, proofs coordinator: your issue

Your Linear issue is PRF-363 (goal label proofs/sampling-generic). Every PR you open carries `Refs PRF-363` and `Goal proofs/sampling-generic` in its body: from about Wed 1 PM PST ci gates each mark on the issue's goal label.

## 23:08Z, proofs coordinator: re-staffed (replaces the 21:45Z PAUSE and W5)

Stop W5 (it targets the superseded Game-model `Defs`). Push what you have and note what's model-independent. Then take,
on the restated claims-core compiler (GENSAMPLE's note, section 'The restated statement (claims core), for re-staffing
(22:45Z)'; `Security/Definitions/Claims/SamplingGeneric.lean` on `cursor/proofs-gensample-0ab3`):
- R3e, the acceptance-probability composition: Pr[acc and P_a(F) <= delta] <= c^2 delta + ..., via `Sessions.draws`
  and the skeleton of your old `Compose` (L);
- R7, `ZKOutLeak`: the sum over outcomes (S).
Branch `cursor/proofs-gsr3e-<4 hex>` from GENSAMPLE's current head. Tries on `--pool tries`. No sorry. Done by Thu
06:00Z. Refs PRF-363, Goal proofs/sampling-generic.
