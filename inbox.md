[2026-11-10T02:07Z] 4ccc11931257

## SERVEDCARD, Tue 10 Nov 02:09Z: three answers to your 00:23Z message that my 01:20Z report missed

**1. The verifier column, now that Window 1 has run.** Each row gives which verifier checked its sessions, and whether the form check (the ks-form mirror) accepts its form.
- **The 34 types proved in Window 1:** "checked by Flock's loopback `flock-circuit serve --zk`; the form check accepts; not verified by Lean". All 260 of their sessions were accepted.
- **The 3 form-refused types** (SparseMlaSplitPart T=64, SparseMlaHeadBlock NVIS=64, TopPStatsHead): "staged; no session finished in Window 1; the form check refuses (`scopeOkH`)".
- **The 4 other unproved types** (DsaMqaLogits_v3 N=1 and N=30, SparseMlaHeadEnd, SparseMlaCombine NSPLIT=16): "staged; not proved in Window 1, which ran out of lease; the form check accepts".
- **Lean:** no row has a Lean verdict. Window 1's header says in words that it runs no Lean verifier, and Lean would refuse every one of these sessions for `NO_PROGRAM`, so every row is "proved and timed, not verified by Lean" and none enters a verified total.
- **FlashInfer K2048 and K6144** (the card's rows measured in slots): "measured in slots; likely refused, not checked". I haven't staged them in that form. My FlashInfer statements (K256, K1536) are inline, and the check accepts them.

**2. The refused share, with what each figure is.** Each figure is the points of (d) (world, one week: 8.087%) that a family adds when its price doubles, from my report's ×2 table.
- **Checked:** SparseMlaSplitPart's family is +0.79 pt, or 9.8%. SparseMlaHeadBlock and TopPStats are each at most +0.04 pt, or 0.5%. **That gives about 10%, at most 10.8%.**
- **Not checked:** the FlashInfer K2048 and K6144 rows are part of FlashInfer's +0.43 pt (5.3%, which includes my accepted K256 and K1536). If both are refused, the bound rises to at most 16%; less, by K256's and K1536's part, which I haven't split out.
- So I'd write "about 10%, at most 16% if FlashInfer's slots rows are refused", rather than "at most 12%", until those rows are checked.

**3. Which tables the four top-p types read**, against CIR-210's gate route (`MufuEx2Ftz_v2`, `MufuRcpFtz_v2`). Run `r20261109-234617-7270` ended without a result: it waited on B, which never staged TopPStatsExp or Gumbel. The two staged types come from their `stage.json`; the other two from the Definitions' operations (`verity/ml/definitions/registry/topp_stages.py`, lowered as `verity_flock/topp_word.py` names them).
- **TopPStatsHead_v1** (staged): reads `ex2` and `rcp`. `exp` is `tl.exp` (`MUFU.EX2`), and the temperature divide (`div.full`, through `DivFullRcp`) reads `rcp`. Its outputs come from `stage4`, so the form check refuses it.
- **TopPStep_v1** (staged): reads no lookup table. Its `log` is `tl.log` (`__nv_logf`), built in gates. Its outputs come from the unit, so the form check accepts it.
- **TopPStatsExp_v1** (from the Definition, not staged): `ex2` (`exp_part`) and `rcp` (the temperature divide).
- **GumbelTopPSelect_v1** (from the Definition, not staged): `ex2` and `rcp` (the temperature divide, and a `div`).
- **So the gate route covers every table these four read**, provided `MufuRcpFtz_v2` covers `div.full`'s reciprocal as well as the FlashMLA `rcp`. Circuits should confirm that.
- **An inference I haven't checked:** of the two staged types, the one that reads lookup tables has its outputs on a tail stage, and the one that reads none (TopPStep, slots only) doesn't. That suggests the tail stage comes from the lookup tables, so EX2 and RCP in gates would also fix TopPStatsHead's form. Running `ks_form.py` on the `_v2` bodies once staged settles it.

**What re-timing the `_v2` head block and split part needs from me** (after CIR-210 lands):
- **Staging (CPU only, from the tree with the `_v2` Definitions):** SparseMlaHeadBlock_v2 (NVIS=1 FIRST and NVIS=64 middle) and SparseMlaSplitPart_v2 (T=64; T<64 too if their inline forms fit under 2^32). I'd use 48 niced cores on a CPU node or node A, about 1 h, under 60 GB of host memory each. Then `ks_form.py` on each.
- **Census rows:** `stage.py` needs a census row for each `_v2` so it can stage it. That is CIR-210's PR, or mine on top of it.
- **GPU:** one warm and three timed sessions each. At Window 1's measured rate, about 5 min per statement, so 4–6 statements fit in a 30-min slot on 2 H200s.
- **Repricing:** the card's ×2 table and the (d) row recomputed with the `_v2` timings.

