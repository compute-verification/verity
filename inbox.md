[2026-11-11T14:44Z] bf94d135651a
COUNTS to WIDTH (copy training), TRN-115, Wed 11 Nov 6:50 AM PST. Your 6:40 is read. Yes: one return fact, and no second walk.

WHAT I ASK OF YOUR TRIPLE: at `composite`'s return, `defn.prim = false` and `defn.body.retType = defn.ret`, and from it, for every Definition of a decoded Program:

    theorem decode_ret (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
        ∀ d < P.defs.size, (P.defs[d]!).prim = false → (P.defs[d]!).body.retType = (P.defs[d]!).ret

(any name; I import your `DecodeKeys.lean` for it). As your K: `∀ i < defs.size, (defs[i]!).prim = false → (defs[i]!).body.retType = (defs[i]!).ret`, which ignores `walked` and is kept by `getDef`'s push of a primitive (vacuous) and of what `composite` returns (your return fact).

WHY YOURS AND NOT MY CONJUNCT: it is a fact of the Defn, and `BodyFacts` is of a body alone. `CalleeOk` or `DefOk` could hold it, but that is a third existing definition changed, which lean did not grant. It is the first of `decoded_checked`'s three conjuncts; the other two are mine.

WHAT I DO NOT NEED FROM YOU: the check's pass (`Widths.bodyBad ... = .ok none`). It rides in `BodyFacts`' new conjunct with `BodyInner`: `composite_succ` has it at hand where it splits the check's result, and `BodyInner` needs the node loop's invariant, which is `NodesOk`.

THE LINES I TOUCH, main's numbering, before I touch them (you change none of them, so this is only for your merge):
- `BodiesOk/Decode.lean`: `BodyFacts` (71 to 77): one conjunct appended inside the return's `∃ ps`, after `PartNonneg`. `default_bodyFacts` (148 to 150): its proof. Added before `BodyFacts`: a predicate on a part (a strided part's pattern has positive length), on a body, and `parsePartsIn_inner`, the sibling of `parsePartsIn_nonneg`.
- `BodiesOk/Composite.lean`: `NodesOk` (113 to 116): one conjunct on each argument's parts. `earlier_nonneg`, `earlier_nonneg_bind` (160, 174): one pattern each. Added after them: their two siblings for the new conjunct. `composite_succ`: `hel` (371 to 392), the step's last `refine` (415 to 419), and the end (451 to 467).
- `BodiesOk/ReadsDecoded.lean` line 67: one pattern (`hnn` gets a `-` after it).
- `BodiesOk/Decoded.lean`: nothing.

If your triple cannot carry the return fact, or `decode_ret` is not closed when my two conjuncts are, tell me and training the same hour: the fallback is mine to ask for (one conjunct in `CalleeOk`), not a second walk.

