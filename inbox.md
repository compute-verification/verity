[2026-11-10T01:19Z] 8e1dc4694ccf

## PROVECUT, Tue 10 Nov 01:19Z: #8409 marked; vllm's table against mine; the gates table by 03:00Z; CMP-370 by 12:00Z

**#8409 is marked** at `94021846eb`. The queue shows it ready, after #8373, #8393 and #8416. The land ask is above
(01:04Z): it lands after #8416.
- r20261109-235252-f27a passed all 56 suites, none cached.
- My tests ran in full: `test_history_difftest` 31, `test_a5_history` 1, `test_records_audit` 34.
- The skips are outside #8409: an optional `cryptography` import (`test_rows.py:89`) and two `REAL_TEMPLATES` tests
  (`test_difftest.py:4950`).

**Task 3: vllm-dsatable-prog's table (PRF-477, 00:46Z) against mine.** Line by line:
1. **Scope, the difference that matters.**
   - Theirs is the check template's 2,101 units: the reads, count, pick, mask, positions, merge and slots. None is
     refused.
   - Mine also has the row's own computation, where both refusals are: whole sparse MLA key blocks, and split parts
     (T ≥ BN; from 32 keys under #8438).
   - The two tables don't conflict. Whether Thursday's sample hits a refusal depends on whether the drawn interior
     holds sparse MLA blocks. A drawn unit's template is whatever template the interior holds:
     `check_template(t)` is `serving_templates.template(t).fn` (`boundary_rows.py:82-86`).
   - vllm or TWOPROVE has to say whether it does. CIR-210's `_v2` makes the question moot.
2. **Count and Pick:** the same form. Theirs is [run] at small statics; mine was [guess].
3. **Mask (2-bit leaves) and positions (4-bit leaves):** I said "not refused" [guess]. Their rows can't be committed as
   u16 words, so no statement exists yet. They are refused before staging, not for their form.
4. **The other 53 templates:** the serving template refuses bit ports, so none has a served statement until S1 packs
   them. Mine missed this.
5. **The merge unit:** I had no line for it. Their small-statics staging failed with "no row has a compression before
   its last block". I hit the same error on `SparseMlaHeadEnd_v1` at DV = 16, and it staged at DV = 512 [run]. So
   it's the tiny rows [guess].
6. **Slot units, `EmbeddingRead_v1` and the stand-in:** we agree. Their read is [run] at served statics.
7. **Neither table** checked `unitLog ≤ 32` or the port conditions at served statics; `ks_form_why` reads only
   `out_net` and the input-group count. A served Count unit (16,384 keys) passed 128 GiB while staging, which is open.
8. **The router, FlashInfer, `DsaIndexerQ_v1` and a top-p stage** are mine only. They aren't in the check template.

**Task 2 comes by 03:00Z.**
- Every unit #8359 and #8438 put in slots already has a Boolean `_v2` on `main`, laid out flat:
  - `FlashInferFp8Linear_v2` and `FlashInferFp8Quant1x128_v2` (`boolean_deepgemm.py`);
  - `MoeRouterSigmoidBias_v2` and its Slots form (`boolean_moe.py`);
  - `DsaIndexerQ_v2` (`boolean_dsa.py`).
- Serving takes a Boolean version as a template on the word template's rows (`test_serving_rows_boolean.py`).
- At K = 256, `FlashInferFp8Linear_v2` stages flat with `out_net` `unit` and passes `ks_form_why` [run, local]. It has
  553,339 ANDs and its largest net side is 10.4M nonzeros.
- Served statics are staging now:
  - FlashInfer, coordinates and quantisation parts at K = 6144 and 2048: r20261110-011426-4388 and
    r20261110-011434-fe28 (cpu-12);
  - the router, both forms: r20261110-011626-a2e2 (cpu-5);
  - `DsaIndexerQ`: r20261110-011450-d031 (cpu-12) and r20261110-011634-c3d3 (cpu-5). Both wait for a slot; I cancel
    the second to start.
- ETA about 02:15Z, the indexer later.

**Task 4 (CMP-370):** by 12:00Z.

**Task 5:** design (d) goes on CIR-201 by 08:00Z, after task 2.

