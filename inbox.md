[2026-11-13T08:01Z] bbf7d5a1ddec
DGRAD, Fri 13 Nov 12:05 AM PST (TRN-156: q4muon-full's rows, the carried state, the branch)

**q4muon-full matches its record on every use and on all but one kind of product, and that one is an `of_record` artifact, not the route: `per_row` divides the record's four rows a forward out of the momentum code's groups because there are 4,096 of them, so it needs your ruling before the entry goes in; the carried state is fixed in the route, with no change to the capture.**

1. **q4muon-full** (RAN r20261113-065657-3f56, prototype route; 50:47 wall, 18.9 GB RSS; my 8 GB ask was too small):
- It matches 880/880 uses and 53/61 products. `norm` fires as expected. E is identity plus scaledquant, 18,696,800 B after 2 steps, beside the record's quant at 18,829,664 B after 2979.
- One line is the printed loss, `fwd/theta div [[]] x16`, which takes `record_only` as the other four entries do.
- The other 12 lines are six pairs, each one shape on the record's side against another on the config's, for example:
  - `other/param mm [[1024,64], [64,64]] x72` (the record's alone) and `other/param mm [[4096,64], [64,64]] x72` (the config's alone);
  - likewise the `div` and `mul` rows at `[1024,64]` against `[4096,64]`, and `mul [[1024]] x48` against `[[4096]] x48`.
- Why (READ `of_record.py` `per_row.of_tokens`): it replaces every dimension equal to rows × ctx (4 × 1024 = 4096) with ctx, in every row of the record, the optimizer's (`other/param`) included. The 24 attention matrices are 512 × 512 = 4,096 groups of 64, so the record's momentum code at `[4096,64]` is read as `[1024,64]`, while the config, counted at one row a forward, keeps `[4096,64]`. The MLP matrices (11,264 groups) compare equal. The 72 = 24 matrices × (one pack + two unpacks) on both sides (READ the route).
- The fix I'd propose is narrow: `per_row` divides rows out only of the forward's and the backward's rows (`fwd`/`bwd`), never `other`'s, which reads no token. That is a change to `of_record.py`, which I don't make without your word. Until then q4muon-full is not in `of_record.toml`; its config, reference and test are in the PR.

2. **The carried state (your 5).**
- READ `capture/circuit.py` `measure`: `state_bits` sums the carried values (`_update`'s carried, from an earlier step and depending on its tokens) that are not `weightlike`, i.e. not reached walking back from the update values the step's token ops read.
- The prototype held each factor as a parameter and rewrote it through a slice (`a[:r].copy_`). Mid-run, the slice of its old value was carried and read by nothing the token ops reach, so it was priced at 32 bits an entry.
- Each step now builds the factors as new tensors from the store, so nothing carries them. RAN locally on the tiny start, with a trace taken mid-run (budget 50): `state_bits` 4,608 → 0 for rank-1 MLP adapters (3 × 48 entries × 32 bits before). The store is priced in `update_cut_bits` = 1,296 (8 bits × 144 + 9 scales × 16).
- The dense path was already right: 16,896 = the 8-bit momentum's planes and scales plus the gains' two AdamW moments, which a step does carry. A matrix trained whole is written over its weight whole, as `qat` does.
- `test_routes` holds both widths and the mid-run `state_bits == 0`. No capture change.

3. **Grow's cost** (INFERRED): counting to step 164, the second of the last phase, means 165 steps of 8 forwards of one row on m-t20, about 20 s a step on a CPU slot, so about an hour, and about 4 GB (RAN r20261113-071311-8669: 2 steps of the last phase plus the trace, 3:57 wall, 3.7 GB).

4. **ia3 recount** (RAN r20261113-073027-5ffe at d9fdf5b8a3): `_fused_adamw_` now equal. Left: the norm rows and the attention's 30/32 and 14/16, as you ruled. It goes in its own PR after dgrad's.

5. **The branch** `cursor/dgrad-route-8002`, head c6a62058e1, based on main 9182900fb9 (no merge since your rule). Commits:
- `routes/_codes` with tests;
- `LowRank(bits)` with the rule written out in a test (RAN locally: `s-t20-1.1x-lora`, the one `lora` entry, keeps digest 4267292a… and E sha256 09f4828a… on main and on the branch; route `lora` packs with `Identity`, so no entry on main reads `LowRank`);
- `ScaledQuant.encode_trained`, plus a fix to one existing test that listed it among the codecs that pack no trained parts;
- `E_KINDS` (`lora2: lowrank`; `E_BESIDE = {"quant": ("identity",)}`, which needs one of quant's own codecs beside it, tested) and `E_PACKING` for `lowrank`;
- the route with six configs, the plain-torch references (all 7 pass locally, grow and q4muon-full included), the stopped-and-continued test (`OUTPUTS`, growing at step 1, before the stop), and the four LoRA entries in `of_record.toml` with the loss `record_only`.
- The ratchet gate passes at c6a62058e1.

In flight:
- r20261113-075603-c287: `of_record` over every held entry at b6b00e0fa0 (the head but for a docstring);
- r20261113-075657-b2a5: the same at main 9182900fb9, for the before and after you asked about;
- r20261113-075801-a29c: `suites.py --changed 9182900fb9 --quick`.

The draft PR opens when they're read.

