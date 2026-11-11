[2026-11-11T17:58Z] 634b42551db3
COUNTS to training, copy FORM, TRN-115, Wed 11 Nov 9:57 AM PST. Jobs 2 and 3 of your 9:40 AM order. No difference found. Every line is marked RAN (a command or a node try gave it) or READ (I read the text; nothing ran).

"The note": you did not name it. I took lean's property note, `lanes/lean/20261110T1630Z-report-lean238-property.md`, the only note that states the property and `complies_of`. It states the definitions in prose; their Lean text is lean's property file. If you meant another note, name it and item (i) is to redo against it; (ii) and (iii) do not depend on the note.

JOB 2, retired terms, at 7f39106d60 (RAN: my two files' blobs there are the ones at my head 311914697b, 39528bcb6c and cf361a7099).
- RAN `tests/test_retired_terms.py`: `offenders` and `doc_offenders` of `RecordBlockFresh.lean` give `([], set())` each; of `RecordFormCounts.lean` the same. Control, RAN: a planted "proof unit" in a docstring beside them is flagged `(…, 1, 'proof unit')`.
- RAN `tests/lint/test_repo_retired_terms.py`: `sites` counts 24 in `RecordBlockFresh.lean`, the allowlist allows 24; 8 in `RecordFormCounts.lean`, the allowlist allows 8. Where they are, RAN: 18 import lines, 5 namespace or open lines and the one identifier `Nci.novelReads` in `block_fresh`'s statement; 5 import lines and 3 namespace or open lines. None is in prose.
- NOT RUN: the root suite itself (`suites.py`); I called the two lints' own functions on the two files.

JOB 3, the five statements at 7f39106d60. I wrote one of the five (`form_counts`).

Whose text. RAN (`git rev-parse`): `security/Properties/Nci/Record.lean` is blob e58d49bb79 at 7f39106d60, at b5a083b55e and at lean's head c4b1da1fdb; `security/Proofs/Properties/Nci/Record.lean` (`complies_of`) is 7eac12a383 at all three; the check's `NciCheck/Record.lean` is 6fcf37b7f1 at 7f39106d60 and at b5a083b55e. So the property and `complies_of` are lean's bytes, and the check's Lean is the integration head's.

(i) Each statement beside the note.
- `form_owns : Properties.Record.Owns form` (`RecordForm.lean`). Note: `Owns`, a premise of `complies_of`. READ: no hypothesis, no binder. RAN: its type is the term `NciCheck.Properties.Record.Owns NciCheck.Record.SecurityProofs.form` and nothing else.
- `form_rates : Properties.Record.Rates form` (`RecordForm.lean`). READ: no hypothesis. RAN: type is `…Record.Rates …form` and nothing else. The note names `Rates` and gives it no prose; the only text is the property file's.
- `form_laidOut : Properties.Record.LaidOut form` (`RecordServed.lean`). Note: "`LaidOut` only says one exists". READ: the definition is `∀ j tr c, F.read j tr = .ok c → ∃ d, Served F c d`; no hypothesis on the statement. RAN: type is `…Record.LaidOut …form` and nothing else.
- `form_counts : Counts form` (`RecordFormCounts.lean`, mine). Note: "`Counts` asks its count only of circuits whose wiring is the text's", "allows a block with no gates". READ: the definition has both (`∀ d, Served F c d → …`, `∨ d.𝓘.gates u = ∅`); no hypothesis. RAN: type is `…Record.Counts …form` and nothing else.
- `form_complies : Complies form := complies_of form_owns form_laidOut form_counts form_rates` (`RecordComplies.lean`). Note: "`Complies F`: accepting gives `Carries`, some `d` with `Served`, and `Meets` per block and `Nci.CompliesRate` for every such `d`, at the instance's `Γ` and `α = num / den`", and "replaced by `Complies` of training's form once `Owns`, `LaidOut`, `Counts` and `Rates` are theorems". READ: the definition says that, clause for clause. RAN: type is `…Record.Complies …form` and nothing else.
- How RAN: run r20261111-175127-7508 on vy-nebius-cpu-3, the detached 7f39106d60 plus one untracked scratch file importing `RecordComplies` (removed after, never committed): lean exit 0 in 212 s. For each of the five it compared the theorem's type with the application of the property's constant, by full name, to `NciCheck.Record.SecurityProofs.form`: "is a theorem: true; universe parameters: 0; … and nothing else: true", five times. `#print axioms` of the five and of `complies_of`: `[propext, Classical.choice, Quot.sound]`, six times.
- `form` itself, RAN `#print`: `read := NciCheck.Record.read`, `slot := Record.Case.slot`, `held := Record.Case.heldAt`, `wiring := c.W`, `inst` the six fields of `c.f.i`, `summarize` the check's summary's `gates` and `fresh`, `rate` `(NciCheck.failure Γ num den #[(s.gates, s.fresh)]).map (·.2)`, `layout` `Record.layoutOf`'s five fields. READ: the check's own `Record.rate` has that same body, and `form_rate` (`rfl`) says so on a record's summary.

(ii) Redefined or shadowed.
- RAN, same run: the environment of a file importing `RecordComplies` holds 55 constants whose last name is `Owns`, `Rates`, `LaidOut`, `Counts`, `Complies`, `Served`, `Form`, `form`, `Accepts` or `Carries`. `Owns` and `Rates`: one each, the property's. `LaidOut`: the property's and `NciCheck.Properties.SecurityProofs.Walk.LaidOut` (`Proofs.Properties.Nci.Walk`, of a Program and a number). `Counts`: the property's and `Spot.Select.Counts`. `Complies`: the property's and `Nci.Complies` (`Definitions.Nci`, the other policy). `Served`: the property's, `Pouw.Window.Served`, `WeightSecurity.Served`. The property's nine (those six, `Form`, `Accepts`, `Carries`) are declared in module `Properties.Nci.Record`; `NciCheck.Record.SecurityProofs.form` in `Proofs.Properties.Nci.RecordForm`, and every other `form` is a field or a definition of another namespace. None of the others is the head of any of the five types (the comparison above is by full name).
- RAN (`git grep` over every `.lean` at 7f39106d60): no declaration named `Owns`, `Rates`, `LaidOut`, `Counts` or `Complies` in namespace `NciCheck.Properties.Record` outside the property file; no `notation`, `macro`, `syntax`, `alias`, `export`, `renaming` or `hiding` line names one of them.
- READ: the five theorems' files and `complies_of`'s open `RecordText` whole (RAN, the list above: it holds none of these names) and otherwise only named lists (`Lean (Json)`, and names of `NciCheck.Record`, `NciCheck.Record.SecurityProofs`, `Flock.Program`, `Flock.TemplateQuery`, `NciCheck.DefinitionDigest`). None has a `variable` or a `universe` line.

(iii) `complies_of`.
- RAN, same run, its binders in order: implicit `Form`; `Owns F`; `LaidOut F`; `Counts F`; `Rates F`. "binders in all: 5", conclusion `NciCheck.Properties.Record.Complies F`. Four hypotheses, these four, in the note's order; no fifth, no instance argument.
- READ: `theorem complies_of {F : Form} (hown : Owns F) (hlaid : LaidOut F) (hcount : Counts F) (hrate : Rates F) : Complies F`, no `variable` line above it.

Not a difference, and not in your three items, so I did not stop on them; both are what the five statements do not say.
- RAN (grep): no theorem at this head names `Record.check` or `Record.evaluate`. `Accepts form j tr c` is `Record.read j tr = .ok c` and the form's rate rule returning none. READ: `evaluate` sets `complies` to exactly that (`read`, `summarize`, `rate c.f.i.gamma c.f.i.num c.f.i.den`, `isNone`), so the published `{"complies": true}` is `∃ c, Accepts form j tr c` by reading, by no theorem.
- RAN (grep): no theorem says some `j`, `tr`, `c` is accepted; the lock at this head still lists `NciCheck.SecurityProofs.ServedComplies` and has no entry for `form_complies`.

Not checked: the package's check (`check.py`, the kernel replay record), a full `--siblings` (the run of job 1 imported 640 modules of 3921), the joint build you said is owed at 7f39106d60. I end my turn here, as ordered.

