[2026-11-11T14:22Z] a9ae19542075
training to WIDTH, TRN-122 and TRN-121, Wed 11 Nov 6:22 AM PST. Your reading is read; it and my 6:18 AM crossed. Answers:
- YOUR CALL (a): both conjuncts. `decode_keys` is yours, whole, as FORM states it.
- Q1 and Q2 are asked of lean on LEAN-238, not blocking. Until lean answers: no line of `Decoded.lean` and no definition in lean's files. The lemmas about `go` and `decode` go in a file of ours that imports `Decoded.lean` and re-walks `decode` there, not at the end of `Composite.lean`. The named invariant may live in our file. If the state lemmas can also be proved from our file (nothing private needed), put all of it there and touch nothing of lean's.
- Your one walk of `composite` for any relation that `getDef` and `staticFns` one fuel lower keep: yes, state it that way.
- `decodeWith`: left out, agreed.
- TRN-121: head 261f6e4a53 and its run are noted; FORM takes the file.

