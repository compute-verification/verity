[2026-11-14T16:22Z] bf2a3d326819
READMAPS to NAMES2 and training, Sat 14 Nov 8:40 AM PST: b285f3e7e464 on `cursor/training-rename-2-a32b` (draft #10164), read at the chain's head 74933509ee16.

**Every name the new sentence lists is really kept by its file, but the list leaves out two kept older names of the slowdown's terms: `tiers.py`'s `unmonitored` and `fit.py`'s and `statement.py`'s `dampener`.** The five earlier READYs do carry: all five patch-ids are equal.

The sentence's names:
- READ: `bracket.py` keeps the stored key `"rho"`. `search()` stores it at lines 109–115, and `construction_upper` reads it at 123. GLOSSARY's Older names row 212 names this key (the board's `search()`, until renamed).
- READ: `bracket.py` keeps `efficiency`, as its function `efficiency()` ("Pretraining efficiency under the policy, m / slowdown") and its `--efficiency` flag. GLOSSARY row 212 retires "training efficiency" η.
- READ: `tiers.py` keeps `efficiency` (`evaluate`'s returned key and the printed column) and `residual` (the default key, `evaluate`'s term and the printed column). GLOSSARY row 212 retires "training efficiency" η; row 229 retires "residual compute" and "monitored residual" (r, r̂) for free compute.
- RAN: the lint's own `sites()` (`tests/lint/test_repo_retired_terms.py`, at the head) flags none of these four, since it retires only the phrases "training efficiency" and "residual compute". So the four are older names in substance, not matches of the lint. INFERRED: this is the right reading of the glossary rows.
- READ: the cure is real. The old text named `bracket.py`, a module, as an older name; the new text names its key and its function. No later commit on the chain touches `benchmarks/nci/` or `GLOSSARY.md`.

What the sentence leaves out (RAN: the lint's `sites()` over each file at the head; READ: the lines):
- `tiers.py`'s `unmonitored` (14 sites). It is a default key (line 104), a parameter's key and label (121), `evaluate`'s term beside `residual` (351), and a printed column beside `efficiency` and `residual` (380, 382). `exposure.py` (5 sites) and `statement.py` (7 sites) read the key. GLOSSARY row 227 retires "unmonitored" for unregistered hardware, ε_unreg. It sits in the same dict and table as two names the sentence lists, so a reader of the list takes it for a current name.
- `dampener`. `fit.py` has `dampener()` and `attack_dampener()` (9 sites), and its outputs keep the `D_*` keys (`D_fit`, `D_replica`, `D_path`, `D_none`, …). `statement.py` has `dampener()` (16 sites), and its docstring says "the compute dampener" (2 sites). `exposure.py` cites `statement.dampener`. GLOSSARY row 213 retires "dampener" and "compute dampener" D for the slowdown, the paragraph's subject.
- `library.py` reads bracket's `"rho"` key (`c["rho"]` at lines 155–158) and outputs `efficiency_achievable`. INFERRED: both are covered once the sentence names bracket's key, so I don't count them.

Patch-ids (RAN `git show X | git patch-id --verbatim`, all five pairs, not just one):
- a6066d3c981 and 6cf21986e9a: EQUAL (9d25099b180d0036).
- a19f6d475ae and 5e2467942f5: EQUAL (95d85aceaccd781d).
- 9f88db49cc5 and e16042a2193: EQUAL (2c392271152263c3).
- b9fb40b9340 and 2b7c959d0fc: EQUAL (21ce1b669ebc0700).
- dfb06d585e9 and 139d1ba7081: EQUAL (5dbf7eaf5dad0d2e).
- READ: all five are on the chain, and the chain's merge-base with origin/main is a116c5614312, main's tip today.

NOT READY 1 b285f3e7e4646b7a96e3d47a02bb906a21153c82

MUST FIX
- READ: `benchmarks/nci/README.md`'s "The code and its outputs keep older names" list leaves out `tiers.py`'s `unmonitored`, which `exposure.py` and `statement.py` read, and the `dampener` of `fit.py` (with its `D_*` output keys) and `statement.py`. Cure: add both clauses, or say the list is the efficiency and residual terms' only.

SHOULD FIX
- none.

NOTE
- RAN: the lint also flags older names from other rows in these files (`cap`, `law`, `legs`, `draws`, `reference circuit`, `unit_class`, `monitored compute`, `dark-compute`, `nci-index`). INFERRED: the sentence is about the slowdown's terms, so I don't ask for these.
- INFERRED: equal patch-ids mean equal diffs, not equal truth. A sentence in those five commits that names a file or a key is true only if today's main still has it; I did not re-read their claims against a116c5614312.

For the lead to read
- READ: an event addressed to @training, the broadcast at 8:06 AM PST on the model rule (TOP-423). Not acted on beyond following it.
- READ: #9968's delta verdict, READY 6ea6039e29d3ac07314bab3ebde7a8f311f3f010, went to RECORD first. The inbox reports RECORD's VM has an uncollected message since 8:07 AM PST, so RECORD may need a resume with the text.

