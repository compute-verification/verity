import Proofs.Properties.Nci.BodiesOk.Decoded
import Proofs.Properties.Nci.BodiesOk.Order
import Proofs.Properties.Nci.Resolve

/-!
# `Resolve.BodiesOk`, proved

`bodiesOk`: every body the walk reaches is `Resolve.BodyOk`. A Program `decode` returns holds Definitions that are
`DefOk` (`decode_spec`): each node calls an earlier Definition as `NodeOk` says, which gives `UpOk`, `DownOk` and `OpOk`
(`up_ok`, `down_ok`, `op_ok`), and every reference names a leaf (`RefIn`). `Prog.orderViolation` passing gives a set of
Definitions holding the root and closed under calls whose nodes name only parameters and earlier nodes
(`order_closed`); every activation `Good` reaches has its Definition in that set (`good_closed`), so its references are
`RefOk` (`body_ok`).
-/

namespace NciCheck.Properties.SecurityProofs.BodiesOk
open Flock.Program

/-- Every activation `Good` reaches runs a composite of a call-closed set holding the root. -/
theorem good_closed (P : Prog) (S : ℕ → Prop) (hS0 : S P.root) (hr : (P.defs[P.root]!).prim = false)
    (hS : ∀ d, S d → (P.defs[d]!).prim = false → ∀ k < (P.defs[d]!).body.nodes.size, S ((P.defs[d]!).body.nodes[k]!).fn) :
    ∀ sc, Resolve.Good P sc → S sc.d ∧ (P.defs[sc.d]!).prim = false
  | .top d off st, hg => by
    obtain ⟨rfl, -, -⟩ := hg
    exact ⟨hS0, hr⟩
  | .child parent d off k m, hg => by
    obtain ⟨hgp, hk, hfn, hp, -, -⟩ := hg
    obtain ⟨h1, h2⟩ := good_closed P S hS0 hr hS parent hgp
    refine ⟨?_, hp⟩
    show S d
    rw [← hfn]
    exact hS _ h1 h2 k hk

/-- A reference of a node's argument lies in one of the argument's parts, among the names `orderViolation` checks. -/
theorem names_of (s : Seq) (ps : Array Part) (hps : s = Seq.ofParts ps) (t : ℕ) (ht : t < s.count) :
    ∃ q ∈ s.parts, ((s.get t).p, (s.get t).i) ∈ partNames q := by
  subst hps
  obtain ⟨i, hi, h1, h2, hget⟩ := Walk.seq_get ps ht
  refine ⟨ps[i]!, ?_, ?_⟩
  · rw [(Walk.ofParts_spec ps).1, getElem!_pos ps i hi]
    exact Array.getElem_mem hi
  · rw [hget]
    exact get_mem_names _ _ (by omega)

/-- A checked composite of a decoded Program is `BodyOk`. -/
theorem body_ok (P : Prog) (hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]!) (d : ℕ) (hchk : Checked P d) :
    Resolve.BodyOk P d := by
  by_cases hd : d < P.defs.size
  swap
  · have e : P.defs[d]! = default := getElem!_neg P.defs d hd
    refine ⟨fun k hk => ?_, fun t ht => ?_⟩
    · rw [e] at hk; exact absurd hk (Nat.not_lt_zero _)
    · rw [e] at ht; exact absurd ht (Nat.not_lt_zero _)
  obtain ⟨-, hnodes, hret, -⟩ := hok d hd
  refine ⟨fun k hk => ?_, fun t ht => ?_⟩
  · obtain ⟨hfn, hnd, hargs⟩ := hnodes k hk
    have hfs : ((P.defs[d]!).body.nodes[k]!).fn < P.defs.size := by omega
    have hc : CalleeOk P.defs[((P.defs[d]!).body.nodes[k]!).fn]! := (hok _ hfs).1
    refine ⟨hfs, fun i t ht => ?_, up_ok _ _ hc hnd, down_ok _ _ hc hnd, fun _ => op_ok _ _ hnd⟩
    by_cases hi : i < ((P.defs[d]!).body.nodes[k]!).args.size
    swap
    · rw [getElem!_neg _ i hi] at ht; exact absurd ht (Nat.not_lt_zero _)
    obtain ⟨⟨ps, hps, -⟩, hrefs⟩ := hargs i hi
    obtain ⟨hp1, hp2⟩ := hrefs t ht
    obtain ⟨q, hq, hnm⟩ := names_of _ ps hps t ht
    have hord := (hchk k hk i hi q hq _ hnm).2
    exact ⟨hp1, fun h => ⟨hord h, (hp2 h).2⟩⟩
  · exact hret t ht

end NciCheck.Properties.SecurityProofs.BodiesOk

namespace NciCheck.Properties.SecurityProofs.Resolve
open Flock.Program BodiesOk

/-- **Every body resolution reaches holds its references in range.** -/
theorem bodiesOk : BodiesOk := by
  intro bytes P hdec hov _ _ _ _ sc hg
  obtain ⟨st, hP, hinv, hroot⟩ := decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := order_closed P hov
  have hok : ∀ i < P.defs.size, DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  obtain ⟨hSd, hpd⟩ := good_closed P S hS0 hroot (fun d hd hp => (hS d hd hp).2) sc hg
  exact body_ok P hok sc.d (hS sc.d hSd hpd).1

end NciCheck.Properties.SecurityProofs.Resolve
