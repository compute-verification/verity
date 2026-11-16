# TRN-177 check 2, today's renames: hand-in for STEP

From PAGE. This scratch branch is never opened as a pull request. These are the page's words, as settled: where the page and the Glossary differ, the page's words go in. Counts are exact spellings on main `6f9f794db37`, in the globs below (`git grep -F -o`), before any rename. Marks: [read] I read the uses; [ran] counted only; [inferred] from one use or a name.

## Globs: training's area

From `.agents/skills/using-slack/registry.json`'s `owns` for `training`, with their tests:

```
benchmarks/slowdown/**  benchmarks/circuit_properties/**  benchmarks/inference_only/**  benchmarks/training_slowdown/**
security/Definitions/Slowdown/**  security/Proofs/Slowdown.lean  security/Proofs/Slowdown/**
security/Properties/InferenceOnly/{Threads,Record}.lean  security/Properties/InferenceOnly/Desc/{Cover,Verdict}.lean
security/Proofs/Properties/InferenceOnly/{DescVerdict,Agrees}.lean
tests/benchmarks/{slowdown,circuit_properties,inference_only,training_slowdown}/**
```

Leave out what is being deleted today or tomorrow. Its words go with it, and renaming them only adds lines:
- formalism v2 (draft #10556);
- Capability, Coverage and Tasks (NAMES2);
- Threads, Layered, Fit and the cover theorem;
- the tiers (`Definitions/Slowdown/Slowdown.lean`, `Remote*.lean`, `Escape`) and ServingBlocks;
- `benchmarks/nci/`;
- `benchmarks/slowdown/slowdown/theorem.py` and what only it reads.

Python names and stored keys are in today's window; Lean names are in Tue 9:00 AM's.

## Today (6:00 PM PST window): prose, printed text, test expectations, Python and shell names, stored keys

| old | the page's word | sense rule: rename only where | on main |
|---|---|---|---|
| `(H)` | `T1` | it names the cut premise `Slowdown.Cut.Assumptions.Hardness` (always so in these globs as far as read [read: Cut, min_cut.py, README]); "(H) and (E)" becomes "T1 and T2". On a file's first use, write "T1 (operands are hard)". | 204 in 32 files [ran] |
| `(E)` | `T2` | it names the premise `Exports`. On first use, "T2 (exports are wide)". Not `E(A)`, `E(B)` (the page's exports set keeps that name). | 93 in 24 files [ran] |
| hardness, as the premise's name in prose ("the hardness premise", "under hardness") | T1 (operands are hard) | it names the premise; not "hard" in other senses, and not the claim id | within the 72 `Hardness` hits [ran], mostly Lean names (Tue) |
| exports, as the premise's name ("the exports premise") | T2 (exports are wide) | it names the premise; not E(A), the exported outputs, which the page also calls exports | within the 33 `Exports` hits [ran] |
| `min-cut bound` | `the slowdown theorem` | it names the result, ρ ≥ cν/μ or its closed form (`Slowdown.Cut.Guarantees.Slowdown`, `HE`, `HEShared`) | 25 in 18 files [ran] |
| `min-cut bound` | `the MAC count` | it names the bound on the MACs, N ≤ n(k + α) + √(Π n O) (`Slowdown.Cut.Guarantees.Bound`, `min_cut.bound`'s MAC figure) | (within the 25) |
| `slowdown lower bound` | `the slowdown theorem` | always | 5 in 4 files [ran] |
| `registered parameters` | `registered information` | it names θ, the registered set (`Verity.RegisteredCircuit.θ`); not a count of a model's parameters | 12 in 7 files [ran] |
| stored key `"rho"` (`benchmarks/circuit_properties/run.py`) | `"shapes"` | the key holds the fan-in and fan-out shapes compared, not the slowdown, so under the page ρ it collides [inferred from run.py's one use]. Rename it with its readers and the schema; stored runs rerun. | 1 [ran] |

## Not today: waits on top's Glossary PR

- **ρ := nc/|C|, the slowdown.** `tests/test_retired_terms.py` holds ρ and "rho" at zero in `SLOWDOWN_HOME` (the four benchmarks directories, `security/README.md`, compose's `verifier/remote/README.md`). A map that writes ρ there fails until top's Glossary PR changes that row.
- **"fresh input" against the page's "unregistered input".** The lint retires "unregistered input" (the Glossary's older-names row), so the page's word can't go in before top's PR. Until then, write "input outside θ" for in(S) \ θ only where the text is being rewritten anyway; no map. 99 in 38 files [ran].
- **"exactly c gates" to "at most c gates":** 0 in these globs [ran]; it is in the Glossary only (top's).

## Tue 9:00 AM PST window: Lean names, with the lock's moved records

| old | new (proposed) | note |
|---|---|---|
| `Slowdown.Cut.Assumptions.Hardness` | `Slowdown.Cut.Assumptions.OperandsHard` | T1; moved record, statement unchanged |
| `Slowdown.Cut.Assumptions.Exports` | `Slowdown.Cut.Assumptions.ExportsWide` | T2; moved record, statement unchanged |
| docstrings' `(H)`, `(E)` in `Definitions/Assumptions.lean`, `Definitions/Slowdown/Cut*.lean`, `Guarantees/Cut.lean`, `Proofs/Slowdown/Cut*.lean` | `T1`, `T2` | with the names |

The new names are a proposal: the page names T1 and T2 by their words. If the lead prefers `T1` and `T2` as identifiers, only this table changes.

## The noes, one reason each

- **The claim ids `training-cut-hardness` and `training-cut-exports` stay.** They are what a lock and a credit record name; a new id would be a second record of one assumption.
- **`HE` and `HEShared` stay.** The page's own table names them.
- **`Slowdown.Cut` stays, not the page's `Nci.Cut`.** "NCI" is retired; the page's citation is the thing to fix.
- **`benchmarks/slowdown/slowdown/min_cut.py` stays for now.** `tests/test_one_slowdown.py` and four callers load it by that path; it is renamed with them in check 5's PR.
- **"wire" and "gate" stay.** This goes up: Core's `Verity.Circuit` calls every node a gate and every lane reads it. Training's statement uses the page's words and gives Core's names; Core is not renamed.
- **"slowdown(𝒢, 𝒜, η)" is no rename.** It is `Tasks.lean`'s, deleted with Tasks (NAMES2).
