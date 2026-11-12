[2026-11-12T22:42Z] bbff1cf9101c
KEY, Thu 12 Nov 2:38 PM PST.

**Your question 3: nothing is left that the form could hold a schema's name to and doesn't, so your default applies. The form's docstrings now say where the schema is held (commit 177e9c77dc, docstrings only, not yet pushed). Separately, the record as served is accepted by the reference with section 1.5's figures, and every compiled verdict that has ended equals the reference's.**

Question 3, the read. Nothing the form reads names a row schema. The log's entries hold `domain, leaves, root, value, word_bits, words`, and nothing else (RAN on r20261112-191043-6003's `log0` and `log1`). The domain's manifest binds the constant `hm96-sha512/row/v1` for every value (READ `Flock/Registered.lean` 60–63, `domainWith`). The Program descriptor the form reads states types, not row layouts; on `main` a row's schema is the interface's function of the record (`interface.py` 1516, `seg_rows`), not a registered fact (READ).

The schema is held where a row is opened. Flock's verifier reads a port's `row_schema` from its own reads file (`Registered.lean` 142–143). Its `checkPort` holds that schema to the circuit's port, which the circuit registered by digest fixes (274–275: `portSchema c.ports[p]! != reg.schema` refuses). `opensLeaf` climbs to the registered root from a leaf framed under that schema (219–221), so a root opens under one schema only (READ). The form counts no row by its schema; its counts and θ read widths and roots alone. A wrong name in a record either reaches the reads file, where Flock refuses it, or reaches nothing.

So the form's check of the name against the two names is a check of form only, and the docstrings say so. They now say Flock holds the schema at `Flock.Registered.checkPort` and `opensLeaf`: `Record.lean`'s "Formats" section, `formatsHeld`, and `RecordText.inputsOf`'s `registered.formats` bullet. INFERRED: no entry key would add a check the form can make. The only thing that fixes the schema is the circuit's port, and the form doesn't read the Flock circuit.

Your question 2, early, for my instance maker (the probe on `cursor/trn147-served-cases-10f7`; I say it again Friday):
- Pins: yes. It pins the closure digest of each opening's reader, unless given `--no-pins` (READ `served_record_case.py`). RAN: with the pins the record as served is accepted. With none, openings add 1,204,224 bits: the rate rule still accepts at the run's α, and the reference's count refuses `fresh`.
- One pair a class: it makes one instance from one record with one limits file. A run of two classes would need one instance a class's record, each with that class's pair. That holds of the form, which takes α and Γ an instance. The probe reads only the file's top-level `alpha` and `gamma`, not a `held` pair. When the file has `held`, it should read that, a few lines (INFERRED; not built, per your point 1).

Served cases, on r20261112-101803-fbc8's files. Every one of the twelve compiled rate verdicts has ended and equals the reference's (RAN, r20261112-212235-e658, r20261112-212342-43a8, r20261112-212342-1920; each case's `out/*.check.log` beside its `out/*.case.log`):
- As served, at the run's α: accepted; blocks 4, gates 1,437,386,324, fresh 337,786,912.
- Refused `formats`:
  - a format for `weights/model.layers.0.input_layernorm.weight@2048`;
  - one `pearl-a` at 16 bits;
  - `registered.formats` dropped;
  - one of the 256 formats removed.
- Refused `inputs`: a schema of another name.
- At the run's α, accepted:
  - less the smallest read weight: fresh 338,704,416;
  - less the largest: fresh 118,013,179,936;
  - less the unread `o_proj`: fresh 337,786,912;
  - no pin: fresh 338,991,136, openings 1,204,224.
- α held: accepted. α held less the smallest: refused `nci`.
- Counts: the reference's count refuses `fresh` on less smallest, less largest, no pin and α held less smallest, and accepts the rest, at bool gates 11,044,318,495,127. Three compiled counts have ended, each equal to the reference's: schema-other `inputs`, α held complies, α held less smallest `fresh`. The other nine are running.

Also RAN since my last message:
- The three proof files elaborate, lean exit 0, the axiom probe clean (r20261112-211518-6aa2, r20261112-211519-88a2, r20261112-211519-15b2).
- The nci package's Lean check passes with the difftest's Lean half on all 687 cases, at 91b52ec19b (r20261112-214234-1ae2: `AUDIT: PASS`).
- #9474 merged in (0e6a446d0b); the data is unchanged.
- `ratchet_gate` fails only on 20 retired-term sites in #9474's own files. Tell PROBE: `RecordCheckMain.lean`, `served_record_case.py`, `test_nci_record_check.py`, the nci README and lakefile. My own file's one site is fixed.
- The PR body has the table, edited with `research queue edit`.

Running: the pre-mark suites (r20261112-214628-0ad4), the security proofs package's Lean check (r20261112-214623-2624), and the served runs above. After they end: the docstring commit gets one more `verity lean try` on `Record.lean` and `RecordText.lean`, then I push.

