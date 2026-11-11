[2026-11-11T03:32Z] 4d3964ad9cc5
FORM (TRN-76) to training's coordinator: (g), (e), (f) for TRN-96, each ready to post. #8676's branch is at `cb869f5842`, pushed, a draft, unmarked.

No finding on (g): `read` never accepts a text that declares one row twice.

---- to post ----

(g) `read` refuses that text, so it holds no row of it. Two wires into one row: `wires` (`Spot.Wires.check`'s overlap, in `finish`). Every other pair of a wire, a registered read, a table read, an opening's port and ingress: `inputs` (`inputsOf` ends on `pairwiseDisjoint` over all the declarations, the wires first). [read by me at `6ee66219c8`] Two random difftest ways reached it only by chance, so it is now pinned: cases 650 to 663, one per pair the served shape has a row for. The compiled check and the reference agree on all 664 cases. [run by me: `r20261111-030425-3347`]

(e) Bits: one definition, neither of the two. `tmplOf` reads `Flock.Sigma.bits`, Flock's structural one; mine is gone; `bitsOf_eq : Spot.Bits.bitsOf t = Flock.Sigma.bits t` is beside my clauses. [`8070485d29`, `cb869f5842`; elaborated by me] Not `Spot.Bits.bitsOf`: it lists every leaf, `Array<n, Value<16>>` costing n, and its import adds 61 modules to the check. [read; counted by me] Fields: they differ only on a Tuple or Record of no field. Refusing it is meant: the interface that writes the record refuses that return [run by me], so mine stays, with a docstring sentence. 0 of 664 verdicts changed. [run by me: `r20261111-031750-98e8`]

(f) Accepted as you rule it: the numbering is lean's lemma, stated by PROPERTY; the rows and outputs, `tmplOf` against `TemplateOf`, are inside my proof of `LaidOut`. I read the code no other way. [read by me at `cb869f5842`] That proof then also holds two joins: `bitsOf_eq`, written, since `TemplateOf` reads `Spot.Bits.bitsOf` and `tmplOf` reads `Flock.Sigma.bits`; and that the templates `Spot.Wires.Wiring.ofJson` gives back from the JSON the check builds (`piece`) are `tmplOf`'s. It needs `TemplateOf` to keep reading `NciCheck.RecordText.fieldsOf`.

---- for you, not for posting ----

(e) is a departure from your default on both halves. Rule on it if you read it otherwise; the change back is small.

Bits.
- Where the two differ: in value on no type. `Flock.Sigma.widths_sum`, on `main`, proves the leaves' widths sum to `Flock.Sigma.bits`, and `bitsOf_eq` joins that to `Spot.Bits.bitsOf` [elaborated by me, axioms the three]. In work, on every Array. Smallest example: `Array<2, Value<16>>`. Structural: 2 times 16. `Spot.Bits.bitsOf`: `Ty.widths` builds `#[16, 16]`, one entry a leaf, then sums. The decoder takes any length up to 2^53 - 1, so a type of about 40 bytes can have 2^50 leaves [read: `Flock/Program.lean`, `Ty.ofJson`, `MAX_SAFE_INT`].
- What the check did: my own structural `bits`, a fourth definition in the tree. Now `Flock.Sigma.bits`: the same value by the same work, and #8676 defines none.
- Why not `Spot.Bits.bitsOf`, the two reasons named: (1) AGENTS.md, Verifier cost: the form reads a type only by its description today (no `Ty.widths` in `Record.lean` or `RecordText.lean` [read]). (2) The list `layers` -> `NciCheck.RecordText` in the package's `lean-audit.json` (`Flock.TemplateQuery`, `Spot.Wires`) would gain `Spot.Bits`, which brings 61 modules, 22,142 lines, of Flock's circuit and proof modules into a check whose closure is 16 modules, 8,157 lines; `Flock.Sigma` adds one module, 871 lines, that imports the decoder alone [counted by me from the import lines at `6ee66219c8`]. No list forbids the import.
- Left on `main`, not mine: three definitions of a type's bits (`Flock.Sigma.bits`; `Spot.Bits.bitsOf` and `NetworkCheck.Period.bitsOf`, one body). The one home is `Flock.Program`, as `Ty.bits` beside `Ty.leaves` and `Ty.widths`: proofs' file. Worth asking proofs with `strOf?` and `strAt`.

Fields.
- Where the two differ: only on a Tuple or a Record of no field. Smallest example: `Tuple[]`. Mine: one field, the type itself, of no bit. `Spot.Bits.fieldsOf`: no field.
- What the check does there: `tmplOf` refuses the Definition, `templates`, since its one output has no bit [read]. The reference refuses it `templates`, and `verity.ml.windows.interface.layout` refuses it too ("o0: its leaves are [] bits, 0 in all") [both run by me on a Definition returning `Tuple[]`, at `6ee66219c8`]. No difftest case has such a Definition [read], so nothing pins it yet.
- Meant, from what the check needs: an output with no bit has no word, and no wire could read it (a wire's two rows have the same words, a parameter's row at least one); the interface that writes records lays out no unit of such a Definition, so refusing costs no honest record, and the check takes no form ahead of its proof. Mine is the interface's `_fields`, line for line.
- Which way: mine stays, one sentence in its docstring (in `8070485d29`). One `fieldsOf` would mean reading `Spot.Bits.fieldsOf` (the 61 modules again) and a refusal line added to keep the verdict. The one home is again `Flock.Program`.
- Verdicts: 0 of 664 changed, the first 650 among them: the compiled check against the same pinned reference results, 664 cases, 0 differences [run by me: `r20261111-031750-98e8`, AUDIT PASS, at `8070485d29`].

(g), the numbers: cases 650 to 663 are the served 2 x 3 shape's record with one declaration more. 650: a wire into a wired row, `wires`. 651 to 663, `inputs`: a registered read into a wired row, into one row of a wired cut, into a row a registered read fills, into a row a table read fills, into an opened row; a table read into a row a table read fills; among an opening's ports a wired row, a row a table read fills, a row twice; ingress into a wired row, a registered read's, a table read's, an opened row. The new test also takes each second declaration alone, with the first dropped, and the reference passes it: the refusal is the pair's.

The clauses' module is written, tried and pushed: `Security/Proofs/Properties/Nci/RecordRead.lean` (`cb869f5842`), a root of the proofs package. `slot_of_read`, `declared_of_heldAt`, `wiring_of_read`, `inst_of_read`, `bitsOf_eq`. Tried in its own package on a node, `verity lean try --siblings --on auto`, `r20261111-033013-4bfe`: lean exit 0 against its built imports (83 jobs, its imports only, no build of the proofs tree), none of its 13 theorems uses an axiom beyond `propext`, `Classical.choice`, `Quot.sound`, and no built sibling declares a name it does (870 imported; the modules of `Proofs.Accounting` and others not built on that slot were not checked) [run by me]. Not run: the proofs package's Lean check, so nothing is replayed through the kernel and no record is written.

(b): the `valueAt` change is 17 uses and the definition, all in `RecordText.lean`, a file #8676 adds, so no other pull request can hold it before #8676 lands. I put it in #8676 as its own commit on the crossed head, once lean's draft no longer reads `valueAt` (it does at `628efb1799b8`, in `ofJson_wiresOf`).

Runs so far: `r20261111-030425-3347` Lean check at `6ee66219c8`, AUDIT PASS, 664 cases, 0 differences. `r20261111-031211-1212` suites at `6ee66219c8`, 2 passed, 0 failed. `r20261111-031750-98e8` Lean check at `8070485d29`, AUDIT PASS, 664 cases, 0 differences. `r20261111-033013-4bfe` the module's try at `cb869f5842`, exit 0. In flight: `r20261111-032946-93c7`, suites at `cb869f5842` (check: `research fetch r20261111-032946-93c7`).

