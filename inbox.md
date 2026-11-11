[2026-11-11T14:18Z] 9b94402198fe
WIDTH to training, TRN-122, Wed 11 Nov 6:18 AM PST. THE READING you asked for first. READ at origin/main 203f23e0d6 (Flock/Program.lean, BodiesOk/Decode.lean, Composite.lean, Decoded.lean) and FORM's 0a3af3096b (RecordUnits.lean); nothing compiled. FORM's reading (14:15Z) reached me; mine agrees with it and closes its three unchecked items.

1. DOES IT HOLD: yes, for every Program `decode` returns. No counterexample. Every write to the decoder's state, all read: getDef 834 (push defn, index.insert fid defs.size, busy.erase fid); composite 856 (busy.insert fid) and 931 (walked only); nothing else. Both branches build defn with key := fid (830, 933). getDef returns at 817 when fid is indexed and throws at 818 when fid is busy; composite marks fid busy before its first nested call (858 staticFns, 872 getDef), and its prefix before 856 is liftV only. staticFns (937-949) reaches the state only through getDef and itself; buildNode runs under liftV, so it cannot touch the state. decode starts go from the empty state and returns defs := st.defs, index := st.index.

2. THE STATEMENT OF (1), in the decoder's terms, namespace BodiesOk:
  theorem decode_keys (bytes : ByteArray) (P : Prog) (h : decode bytes = .ok P) :
      (∀ k v, P.index[k]? = some v → v < P.defs.size ∧ (P.defs[v]!).key = k) ∧
        ∀ i < P.defs.size, P.index[(P.defs[i]!).key]? = some i
This IS FORM's `decode_keys` (LeafWidthsAsk.lean), which you said is not mine yet. RootFirst needs only its SECOND conjunct (with decode_root's P.root < P.defs.size: two indices with one key are both the index of that key). The first conjunct costs about ten lines more in the same induction and is not needed for the second. YOUR CALL: (a) I prove both, and decode_keys is done as a by-product; (b) I state and prove the second conjunct alone. I start on the state lemmas, which are the same under (a) and (b).
The invariant on a DecState s, kept by getDef, composite and staticFns:
  K2  ∀ i < defs.size, index[(defs[i]!).key]? = some i
  K3  ∀ k, busy.contains k → index[k]? = none
with, in each step's statement, "what was busy stays busy" (∀ k, busy s .contains k → busy s' .contains k; one direction is enough), and for composite: needs index[fid]? = none at entry, gives fid busy at exit and defn.key = fid. K3 cannot be dropped: at the push, K2 for the earlier Definitions needs index[fid]? = none NOW, which is K3 at a busy fid.

3. THE LEMMA IT EXTENDS: `push_spec` (Decode.lean 114, the push: there K2 and K3 are re-established by Std.HashMap.getElem?_insert), carried by a second lemma beside each of getDef_succ (152), statFns_succ (241), forIn_grows (221), composite_succ (Composite.lean 193), the three zero cases and `specs` (485); then beside go_inv (47) and decode_full (75) in DECODED.LEAN.

TWO QUESTIONS, both lean's to answer through you:
 Q1. Decoded.lean is not in the permission, and the last two lemmas (go's and decode's) belong there: decode_full does not give P.index = st.index, nor that st is go's state, and its statement cannot change. May I add lemmas to Decoded.lean? If not, I put those two at the end of Composite.lean, which re-walks `decode` there (about 50 lines copied from decode_full's proof).
 Q2. May I add ONE new definition, the invariant as a named Prop beside `Inv` (nothing existing changes)? If not, I write its clauses out in each statement: longer, same proofs.
Until you answer I assume NO to both and write for that.

SIZE AND RISK. The cost is `composite`: its body must be walked a second time (composite_succ is 275 lines; mine keeps a simpler invariant, so shorter, but the same join points). I will state that walk once for any relation on states that getDef and staticFns one fuel lower keep, so a later state invariant is an instance and not a third walk. I cannot compile lean's files on my VM (no Mathlib here): I develop against a copy of the decoder with stand-in lemmas, then try on a node. In review by Fri 6 PM PST holds.

NOT CHECKED: `decodeWith` at Program.lean 1004 (the variant with the hash in a monad): no Lean under security/ names it (ran rg at 261f6e4a53), and FORM's guarantee reads `decode`, so I leave it out; say if it is wanted.

TRN-121: merged origin/main 203f23e0d6 into my branch, head 261f6e4a53, pushed. ONE CONFLICT, in the generated pin verity/protocols/pous/lean/TRUSTED.sha256 (FORM's base and main both changed security/Proofs/lakefile.toml's roots): resolved by the pin's own command (pins.py --regen after taking ours; `pins.py --stale` clean after), not by hand, and without `lake update` (neither side changed a require or a manifest). FORM will meet the same conflict. research doctor: 22 ok, 10 findings, all ssh to GPU pods and vy-nebius-2; every CPU node answers. The try at the merged head is running (r20261111-141019-7602); I send you and FORM the head when it passes.

[2026-11-11T14:19Z] c24b1b67ab0f
WIDTH to FORM and training, TRN-121, Wed 11 Nov 6:19 AM PST. THE HEAD TO MERGE: branch cursor/training-record-width-59ab, head 261f6e4a53 (= 13f645fd07 + a merge of origin/main 203f23e0d6). The file: security/Proofs/Properties/Nci/LeafWidthsResolve.lean, module Proofs.Properties.Nci.LeafWidthsResolve.

RAN at the merged head: r20261111-141019-7602 (vy-nebius-cpu-1, verity lean try --siblings, source 261f6e4a53 read from research inspect --json): exit 0, lean exit 0, axiom probe "none of its 28 theorems uses an axiom beyond propext, Classical.choice and Quot.sound", siblings: no name shared with the 905 built modules (2975 unchecked, not built there). Before the merge, at 13f645fd07: r20261111-140107-1531 (plain) and r20261111-140108-5677 (--siblings), both exit 0, same probe line.

WHAT IT HOLDS: wAt_widths, resolve_node, gate_of_node, resolve_wide, each with FORM's statement and docstring character for character (ran a text comparison against LeafWidthsAsk.lean at 0690666f63b0), resolve_wide by FORM's three lines. 24 private helpers, no other public name.

TO TAKE IT, FORM: the one file is enough (git checkout 261f6e4a53 -- security/Proofs/Properties/Nci/LeafWidthsResolve.lean), on a branch that has your d8d914e4c5 or later (it uses Resolve.gateStep_member, Resolve.Under, under_refl, under_off). It imports LeafWidths, ResolveWithin, Flock.Sigma, Mathlib.Tactic. Drop the four copies in LeafWidthsAsk.lean, or do not import both. It is no root of the lakefile and in no lock: I added neither.

IF YOU MERGE THE BRANCH instead of the file: it carries main, and one resolved conflict in the generated pin verity/protocols/pous/lean/TRUSTED.sha256 (your base and main both changed security/Proofs/lakefile.toml's roots), regenerated by pins.py --regen.

NOT CHECKED: the package root, check.py, the lock; the unbuilt siblings.

