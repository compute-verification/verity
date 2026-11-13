circuits/PARTSRETIRE, Thu 12 Nov 5:49 PM PST. **On the lead's word, `dsa-q`'s replacement in #9569 is now the census's own Definition, `census:DsaIndexerQ_v1{H=32,D=128,ROPE=64}`, named first, with `dsa-q-row` (its Boolean version) second.** `stage.py`'s refusal reads "stage census:DsaIndexerQ_v1{H=32,D=128,ROPE=64} or dsa-q-row in its place (the census's whole Definition of the indexer query, or its Boolean version)". It uses the same two-choice form as `router` and `moe-w13`.

- **The head** is `fe800897e74ddcb476bb939a50f6564bc6f32740` on `cursor/circuits-cir259-parts-retire-6f04`. The PR is unmarked.
- **Suites:** `suites.py --changed 5f0c008aa040` at that head passed 9 of 9, none cached, 0 failed (run `r20261113-011713-d278`).
- **Evidence:** the Lean verdicts are run `r20261112-183158-1477`'s. Lean took about 147 GB there at one block, and no replacement has a Lean accept at a full statement's count.
