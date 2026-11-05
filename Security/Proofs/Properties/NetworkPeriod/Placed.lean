import Properties.NetworkPeriod.Placed
import Proofs.Properties.NetworkScan

/-!
# An accepted scan's proof blocks each commit at most `max(b_ing, b_eg)` bits

`ScanWindowsPlaced` reads a window's statement only through `StmtBits`: its gates are some members' joint program
(`jointOf`), each member's circuit file one `fs` lists at a template, and each proof block a member's unit's
(`Bits.blockOf`). The bound is per block, from its member's file alone, then summed over an incorrect set:

* `committed_blockOf` (Lemma A, the file): a unit's block commits at most `16 · (paramWords + retWords)` bits of the
  Definition its file is checked against (`Bits.check`), reading no input padding (`Bits.readsWhy` at no parameter):
  its committed gates are the source gates it owns (`insOf`, at most 16 a port word: a port's bits past its first word
  are the zero, `insOf_length`) and its returned copy rows (16 a word of the return, `units_committed`); no gate
  outside it reads one of its rows (`Lowered.reads_jointUnit`).
* `words_le_terms` (Lemma A, the scan): each template's `16 · (paramWords + retWords)` is at most
  `max sc.ingress sc.egress`, `scanProg`'s terms (`vuBits`): bucket 0's and the later ingress buckets' at `ingress`
  (the later template returns bucket 0's return), the egress template's at `egress`.
* `committed_le_terms` (Lemma A): the two, at a member's file at a template index.
* `faultyBits_le` (Lemma B): an incorrect set of at most `b` blocks, each committing at most `β`, commits at most
  `b · β`.
-/

namespace Properties.NetworkPeriod.Placed

open SampledProofs.Bits
open Properties.NetworkScan
open NetworkCheck.Period (Scan Schedule scanProg templateNodes paramWords vuBits)
open NetworkCheck.Circuit (templateDef)

/-- **Lemma A, the file alone**: where member `m`'s circuit file passes Bits' check as the template `tmpl`, whose
Definition is `d`, and reads no input padding, its unit `u`'s proof block (`blockOf`, at the member's constants and
first position) commits at most `16 · (paramWords d.params + retWords d.ret)` bits of the members' joint program. -/
theorem committed_blockOf {P : Flock.Program.Prog} {tmpl : String} {d : Flock.Program.Defn} {ms : Members} {m u : ℕ}
    (hm : m < ms.length) (hu : u < (ms.getD m default).2.size)
    (hc : check (ms.getD m default).1 P tmpl = none) (hd : P.defs.find? (·.key == tmpl) = some d)
    (hr : readsWhy (ms.getD m default).1 d (unitReads (ms.getD m default).1) [] = none) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      16 * (paramWords d.params + retWords d.ret) := by
  sorry

/-- **Lemma A, the scan's terms**: at an accepted scan, each template's `16 · (paramWords + retWords)` is at most
`max sc.ingress sc.egress`. -/
theorem words_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    (h : scanProg P program s = .ok sc) {i : ℕ} (hi : i < (templateNodes P).length) :
    16 * (paramWords (templateDef P ((templateNodes P).getD i 0)).params +
      retWords (templateDef P ((templateNodes P).getD i 0)).ret) ≤ max sc.ingress sc.egress := by
  sorry

/-- **Lemma A**: at an accepted scan whose templates' files `fs` Bits lays out (`FilesOf`) reading no input padding
(`PaddingUnread`), a unit's proof block of a member whose file is the one `fs` lists at a template index commits at
most `max sc.ingress sc.egress` bits of the members' joint program, whatever the other members are. -/
theorem committed_le_terms {P : Flock.Program.Prog} {program : String} {s : Schedule} {sc : Scan}
    {fs : List Flock.Circuit} (hscan : scanProg P program s = .ok sc) (hfs : FilesOf P fs) (hpad : PaddingUnread P fs)
    {ms : Members} {m u : ℕ} (hm : m < ms.length) (hu : u < (ms.getD m default).2.size) {i : ℕ}
    (hi : i < (templateNodes P).length) (hfile : (ms.getD m default).1 = fs.getD i default) :
    Claims.Sampling.committed (jointOf ms)
        ((blockOf (keyProg (ms.getD m default).1 (ms.getD m default).2.size)
          (consts (ms.getD m default).1 (ms.getD m default).2.size) u).map (base ms m + ·)) ≤
      max sc.ingress sc.egress := by
  sorry

/-- **Lemma B**: a set `F` of the partition's blocks, at most `b` of them, each committing at most `β` bits, commits at
most `b · β` (`faultyBits`). -/
theorem faultyBits_le {C : Claims.Sampling.Prog} {Q : Claims.Sampling.Partition} {F : Finset ℕ} {β b : ℕ}
    (hF : F ⊆ Finset.range Q.blocks.length)
    (hβ : ∀ v < Q.blocks.length, Claims.Sampling.committed C (Q.blocks.getD v []) ≤ β) (hb : F.card ≤ b) :
    NetTiming.faultyBits C Q F ≤ b * β := by
  sorry

end Properties.NetworkPeriod.Placed
