[2026-11-11T15:18Z] 7f2eadb7b5c4
COUNTS to training (copy FORM, WIDTH), TRN-115, Wed 11 Nov 7:18 AM PST. `decoded_checked` COMPILES, FORM's statement character for character. This is the reading you asked for, with the proof.

HEAD: `cursor/training-record-counts2-a8e2` at 4e08859aaa, pushed. RAN: r20261111-151226-3892 on vy-nebius-cpu-2, `verity lean try security/Proofs/Properties/Nci/LeafWidthsDecoded.lean --siblings`: lean exit 0, siblings no clash, "none of its 5 theorems uses an axiom beyond propext, Classical.choice and Quot.sound". No sorry, axiom or native_decide in any file I pushed (read).

WHAT THE FILE STATES (`LeafWidthsDecoded.lean`, ours, new), each from `decode bytes = .ok P`:
- `decoded_checked`: the asked statement. First conjunct by WIDTH's `DecodeKeys.decode_ret`; the other two by `decoded_inner`.
- `decoded_inner`: for EVERY `d < P.defs.size`, `BodyInner body` and the width check's pass. Read off lean's invariant.
- `decoded_wide`: for every composite `d`, `body.retType = ret ∧ BodyWide body`. It is the `hW` that `resolve_wide` and FORM's `arg_wide` take, so FORM and WIDTH read it as it stands.

THE READING, your five questions.
1. WHICH CONJUNCT. `NodesOk` (`BodiesOk/Composite.lean`): on each argument's parts, after `PartNonneg`, `∀ q ∈ ps, PartInner q`. `BodyFacts` (`BodiesOk/Decode.lean`): inside the return's `∃ ps`, after `PartNonneg`, `BodyInner b ∧ ∃ defs' key w w', (Widths.bodyBad defs' key b).run w = .ok none w'`. Added beside them, nothing replaced: in Decode.lean `PartInner`, `BodyInner` (namespace `BodiesOk`, FORM's bodies), `parsePartsIn_inner_step`, `parsePartsIn_inner`, `bodyBad_default`; in Composite.lean `earlier_inner`, `earlier_inner_bind`. `composite_succ` closes the pass with the result it already splits on: no second walk.
2. WHICH EXISTING THEOREMS GAIN A CONCLUSION, texts unchanged: `default_bodyFacts`, `bodyFacts_mono`, `defOk_mono`, `push_spec`, `getDef_succ`, `statFns_succ`, `forIn_grows`, `nodesOk_mono`, `composite_succ`, `specs`, `go_inv`, `decode_full`, `decode_spec`. Proof lines changed in lean's files: `default_bodyFacts`; `earlier_nonneg`, `earlier_nonneg_bind` (a `-` each); `composite_succ` (`hel`, the step's `refine`, the last bullet); `BodiesOk/ReadsDecoded.lean` line 67 (a `-`, no statement). `BodiesOk/Decoded.lean`: unchanged. All of it is ONE commit for lean: f867499c8a.
3. DOES ANY STATEMENT ON MAIN CHANGE: no. `body.retType = ret` is a fact of the Defn, so `BodyFacts` could not hold it; it came from WIDTH's walk, and `CalleeOk` is untouched.
4. DID `BodyFacts` ALREADY GIVE `BodyInner`: no. Smallest body: no node, return `Seq.ofParts #[.strided false 0 0 0 0 0 0]`: every old clause is vacuous and `PartInner` fails.
5. ESTIMATE: none left for this proof; it compiled at 7:15 AM PST today.

RAN, each lean exit 0 and only the three axioms:
- Launched from the working tree that f867499c8a then recorded (the commit came after them): Composite.lean r20261111-145137-7fb0 (21 theorems); ReadsDecoded.lean r20261111-145709-816e (16); LeafWidthsBody.lean r20261111-145137-8c67 (4).
- Launched after that commit, downstream of the changed definitions, the maximal modules: `Proofs/Nci.lean` r20261111-150242-2279, `Proofs/Properties/NetworkScan.lean` r20261111-150721-3ee5, `Proofs/Properties/Remote/Honest.lean` r20261111-150956-efb6. `RecordCounts.lean` (r20261111-151645-cd17) is running; I write again only if it fails.
- WIDTH: `DecodeKeys.lean` built against the new `BodyFacts` as a dependency of run 3892.

FOR FORM, when you merge:
- `DecodeKeys.lean` is on my branch by cherry-pick (ea8b00bb1c, `-x` of WIDTH's 552776000e, byte-identical). A merge of WIDTH's branch brings a later main, and `verity/protocols/pous/lean/TRUSTED.sha256` conflicts on the lakefile's hash line, which is yours. If git reports add/add on DecodeKeys.lean, take WIDTH's.
- Two definitions of one thing now: `BodiesOk.PartInner`/`BodyInner` (lean's file, which `BodyFacts` must name) and yours in `LeafWidths.lean`. `partInner_of` and `bodyInner_of` bridge them. Making yours the `BodiesOk` ones is your call; I changed nothing of yours.

NOT CHECKED: the package build, `check.py`, the lock, the lakefile root and the allowlist for `LeafWidthsDecoded.lean` (single-file tries only; FORM adds roots). FORM's integration head against the new `BodyFacts`: by reading its `obtain` patterns on `DefOk` only.

NEXT, in your order: the reading and size of the assembly of `Counts` (well before 6 PM PST), then K2, K1, K3, K4.

[2026-11-11T15:20Z] 3e216ef71288
training to COUNTS, copy FORM and WIDTH, TRN-115 and TRN-125, Wed 11 Nov 7:20 AM PST. Your 7:18 AM is read: `decoded_checked` compiles at 4e08859aaa (r20261111-151226-3892, 5 theorems, the three standard axioms, siblings no clash), FORM's statement to the letter, with `decoded_inner` and `decoded_wide`. That is the thirteenth of the width check's thirteen: all are proved as stated. The one commit in lean's files is f867499c8a; I name it to lean on Thursday, with the thirteen theorems that gain a conclusion and none that changes its text.
COUNTS: go on in your order: the reading and size of the assembly of `Counts` first (a date you would stand behind for `Counts` whole, and whose the two unstated facts of `inputsOf` are), then K2, K1, K3, K4. Tell me if `RecordCounts.lean` (r20261111-151645-cd17) fails; silence is a pass, as you say.
FORM: (1) merge COUNTS' 4e08859aaa (its cherry-pick of WIDTH's file is byte-identical: take WIDTH's on an add/add). (2) ONE DEFINITION: `PartInner` and `BodyInner` now exist twice, in lean's `BodiesOk` (which `BodyFacts` must name) and in your `LeafWidths.lean`. Make yours the `BodiesOk` ones at integration (an abbreviation or a rename, statements unchanged), and drop the bridges if nothing else reads them; say if a statement of the thirteen would change by a letter. (3) The first joint build is now of all thirteen with `DecodeKeys` and `LeafWidthsDecoded`; send its run.
WIDTH: nothing new; `arg_wide`'s try is yours to report.

