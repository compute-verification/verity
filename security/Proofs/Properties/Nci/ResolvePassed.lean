module

public import Proofs.Properties.Nci.ResolveWithin
public import Proofs.Properties.Nci.RecordPassed

/-!
# A leaf an activation returns through a parameter is a leaf of a parameter its Definition passes on

`Resolve.ret_within` says a leaf an activation `U` returns is one of `U`'s gates or the gate a leaf of one of `U`'s
parameters resolves to. This module names the parameter (`ret_passed_on`): it is one `Record.passedOn` lists for
`U`'s Definition. So a parameter `passedOn` does not list has no leaf among the leaves `U` returns that are not
`U`'s own gates.

The walk of resolution from a leaf of `U`'s return keeps `Record.SecurityProofs.Passing` (`step_passing`): a step
down enters a member at a reference of its callee's return, a step up reaches a reference of the argument that fills
the parameter it left through, and a scan's step to the member before reaches a reference of the callee's return
again. It rests on two facts of a decoded Program: a body's return and each argument of each node are the sequences
of their parts (`Seq.ofParts`). `ret_passed_on` takes them as hypotheses, and `ret_passed_on_of` and
`param_returned_of` have them from `decode` (`decoded_walk`).
-/

@[expose] public section

namespace NciCheck.Properties.SecurityProofs.Resolve

open Flock.Program
open NciCheck.Record.SecurityProofs (Passing passOf passing_start passing_down passing_up passing_back passing_exit
  arg_lt)

/-- A reference of a sequence is a reference of one of its parts. -/
theorem seq_part (s : Seq) (hs : s = Seq.ofParts s.parts) (t : ℕ) (ht : t < s.count) :
    ∃ q ∈ s.parts, ∃ t' < q.count, s.get t = q.get t' := by
  have ht' : t < (Seq.ofParts s.parts).count := by
    rw [← hs]
    exact ht
  obtain ⟨i, hi, h1, h2, hget⟩ := Walk.seq_get s.parts ht'
  refine ⟨s.parts[i]!, ?_, t - (Seq.ofParts s.parts).starts[i]!, by omega, ?_⟩
  · rw [getElem!_pos s.parts i hi]
    exact Array.getElem_mem hi
  · rw [← hget, ← hs]

section
variable (P : Prog) (hroot : P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false)
  (hB : ∀ sc, Good P sc → BodyOk P sc.d) (hL : ∀ d < P.defs.size, Walk.LaidOut P d)
  (hret : ∀ d < P.defs.size, (P.defs[d]!).prim = false →
    (P.defs[d]!).body.ret = Seq.ofParts (P.defs[d]!).body.ret.parts)
  (harg : ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size,
    ∀ i < ((P.defs[d]!).body.nodes[k]!).args.size,
      (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
        Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts)
include hroot hB hL hret harg

omit hL in
/-- **One step of resolution keeps the walk passing.** From a state that is not a parameter's reference at `U`
itself, a step to another state keeps `Passing`. -/
theorem step_passing (U : Scope) (B : ℕ) (s s' : Scope × Ref) (hs : Inv P B s)
    (hx : resolveStep P s = .ok (.inr s')) (hne : ¬ (s.1 = U ∧ s.2.p = true)) (hpass : Passing P U s) :
    Passing P U s' := by
  obtain ⟨sc, ⟨p, i, j⟩⟩ := s
  obtain ⟨hg, hr, -, -⟩ := hs
  dsimp only at hg hr hne
  cases p
  · have hi : i < (P.defs[sc.d]!).body.nodes.size := (hr.2 rfl).1
    have hj := (hr.2 rfl).2
    dsimp only at hi hj
    obtain ⟨hfnlt, -, -, hdown, -⟩ := (hB sc hg).1 i hi
    have hdn := hdown j hj
    rcases hf : ((P.defs[sc.d]!).body.nodes[i]!).form
    · rw [rs_prim P sc i j hf] at hx
      cases hx
    · rw [rs_call P sc i j hf] at hx
      cases hx
      simp only [hf] at hdn
      obtain ⟨hcp, hcount⟩ := hdn
      obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret _ hfnlt hcp) j hcount
      exact passing_down P U sc i j _ _ hpass hfnlt hcp hq ht' hget
    · rw [rs_batch P sc i j hf] at hx
      by_cases hcp : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = true
      · simp only [hcp, ite_true] at hx
        cases hx
      · simp only [hcp, Bool.false_eq_true, ite_false] at hx
        cases hx
        simp only [hf] at hdn
        have hcp' : (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).prim = false := by simpa using hcp
        obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret _ hfnlt hcp') _ (hdn.2.2 hcp')
        exact passing_down P U sc i j _ _ hpass hfnlt hcp' hq ht' hget
    · simp only [hf] at hdn
      obtain ⟨hcp, hdn⟩ := hdn
      by_cases hn : ((P.defs[sc.d]!).body.nodes[i]!).n = 0
      · exact absurd hx (rs_scan0 P sc i j _ hf hn)
      obtain ⟨carry, y, hty, -, hcount⟩ := hdn hn
      rw [rs_scan P sc i j carry y hf hn hty] at hx
      dsimp only at hx
      cases hx
      have hcount' : (if j < carry.leaves then (((P.defs[sc.d]!).body.nodes[i]!).n - 1, j)
          else ((j - carry.leaves) / y.leaves, carry.leaves + (j - carry.leaves) % y.leaves)).2 <
          (P.defs[((P.defs[sc.d]!).body.nodes[i]!).fn]!).body.ret.count := by
        split
        · rename_i hjc
          simpa [hjc] using hcount
        · rename_i hjc
          simpa [hjc] using hcount
      obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret _ hfnlt hcp) _ hcount'
      exact passing_down P U sc i j _ _ hpass hfnlt hcp hq ht' hget
  · rcases sc with ⟨d, off, st⟩ | ⟨parent, d, off, k, m⟩
    · unfold Good at hg
      obtain ⟨-, -, rfl⟩ := hg
      exact absurd hx (rs_top P d off i j _)
    · have hg0 := hg
      obtain ⟨hgp, hk, hfn, hcp, hm, -⟩ := hg
      have hU : U ≠ .child parent d off k m := fun e => hne ⟨e.symm, rfl⟩
      obtain ⟨hpd, -⟩ := good_d P hroot hB parent hgp
      obtain ⟨hd, -⟩ := good_d P hroot hB _ hg0
      have hd' : d < P.defs.size := hd
      obtain ⟨-, hargs, hup, -, -⟩ := (hB parent hgp).1 k hk
      rw [hfn] at hup
      have hip := (hr.1 rfl).1
      have hjp := (hr.1 rfl).2
      dsimp only at hip hjp
      have hidx := hup i hip j hjp m hm
      rcases hf : ((P.defs[parent.d]!).body.nodes[k]!).form
      · rw [rs_up P parent d off k m i j (Or.inl hf)] at hx
        cases hx
        simp only [hf] at hidx
        have hia := arg_lt _ i j hidx
        obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (harg _ hpd k hk i hia) j hidx
        exact passing_up P U parent d off k m i j hpass hU hpd hk hfn hia hq ht' hget
          (fun h => ((hargs i j hidx).2 h).1)
      · rw [rs_up P parent d off k m i j (Or.inr hf)] at hx
        cases hx
        simp only [hf] at hidx
        have hia := arg_lt _ i j hidx
        obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (harg _ hpd k hk i hia) j hidx
        exact passing_up P U parent d off k m i j hpass hU hpd hk hfn hia hq ht' hget
          (fun h => ((hargs i j hidx).2 h).1)
      · rw [rs_up_batch P parent d off k m i j hf] at hx
        cases hx
        simp only [hf] at hidx
        have hia := arg_lt _ i _ hidx
        obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (harg _ hpd k hk i hia) _ hidx
        have hget' := (memberRef_eq (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j).trans hget
        have key : ∀ r : Ref, r = (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
            (memberIdx (((P.defs[parent.d]!).body.nodes[k]!).args[i]!) m j) → r.p = false → r.i < k :=
          fun r hr' hp => by subst hr'; exact ((hargs i _ hidx).2 hp).1
        exact passing_up P U parent d off k m i j hpass hU hpd hk hfn hia hq ht' hget'
          (fun h => key _ (memberRef_eq _ m j) h)
      · by_cases hi0 : i = 0
        · subst hi0
          rw [rs_up_scan0 P parent d off k m j hf] at hx
          by_cases hm0 : m = 0
          · simp only [hm0, ite_true] at hx
            cases hx
            have hidx' : j < (((P.defs[parent.d]!).body.nodes[k]!).args[0]!).seq.count := by
              simpa [hf, hm0] using hidx
            have hia := arg_lt _ 0 j hidx'
            obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (harg _ hpd k hk 0 hia) j hidx'
            exact passing_up P U parent d off k m 0 j hpass hU hpd hk hfn hia hq ht' hget
              (fun h => ((hargs 0 j hidx').2 h).1)
          · simp only [hm0, ite_false] at hx
            cases hx
            have hidx' : j < (P.defs[d]!).body.ret.count := by simpa [hf, hm0] using hidx
            obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret d hd' hcp) j hidx'
            exact passing_back P U parent d off k m _ _ _ hpass hU hd' hcp hq ht' hget
        · rw [rs_up_scan P parent d off k m i j hi0 hf] at hx
          cases hx
          have hidx' : (if i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs then
              m * ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).dims.map (·.2)).getD 0 + j else j) <
              (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.count := by
            simpa [hf, hi0] using hidx
          have hia := arg_lt _ i _ hidx'
          obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (harg _ hpd k hk i hia) _ hidx'
          have hre : (if i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs then
              (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
                (m * ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).dims.map (·.2)).getD 0 + j)
              else (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get j) =
              (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
                (if i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs then
                  m * ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).dims.map (·.2)).getD 0 + j else j) := by
            split <;> rfl
          have key : ∀ r : Ref, r = (((P.defs[parent.d]!).body.nodes[k]!).args[i]!).seq.get
              (if i ≤ ((P.defs[parent.d]!).body.nodes[k]!).nxs then
                m * ((((P.defs[parent.d]!).body.nodes[k]!).args[i]!).dims.map (·.2)).getD 0 + j else j) →
              r.p = false → r.i < k :=
            fun r hr' hp => by subst hr'; exact ((hargs i _ hidx').2 hp).1
          exact passing_up P U parent d off k m i j hpass hU hpd hk hfn hia hq ht' (hre.trans hget)
            (fun h => key _ hre h)

/-- **A walk of resolution that starts passing under `U` ends in `U` or leaves through a parameter `U`'s Definition
passes on.** -/
theorem run_passing (U : Scope) (B : ℕ) :
    ∀ (n : ℕ), n ≤ Flock.Steps.FUEL → ∀ (s : Scope × Ref) (b : ℕ), Inv P B s → Under U s.1 → Passing P U s →
      Flock.Steps.run "scope resolution" (resolveStep P) n s = .ok b →
      (U.off ≤ b ∧ b < B) ∨ ∃ i j, i ∈ passOf P U.d ∧ resolve P U ⟨true, i, j⟩ = .ok b
  | 0, _, _, _, _, _, _, h => by simp [Flock.Steps.run, throw, throwThe, MonadExceptOf.throw] at h
  | n + 1, hn, s, b, hs, hu, hpass, h => by
    have h' := h
    rw [Flock.Steps.run.eq_2] at h'
    rcases hx : resolveStep P s with e | (a | s')
    · simp [hx, bind, Except.bind] at h'
    · simp only [hx, bind, Except.bind, pure, Except.pure, Except.ok.injEq] at h'
      rw [← h']
      exact Or.inl ⟨Nat.le_trans (under_off P U s.1 hs.1 hu) ((step_where P hB B s _ hs hx).1 a rfl).2,
        (step_ok P hroot hB hL B s _ hs hx).1 a rfl⟩
    · simp only [hx, bind, Except.bind] at h'
      have hs' := (step_ok P hroot hB hL B s _ hs hx).2 s' rfl
      rcases (step_where P hB B s _ hs hx).2 s' rfl with
        ⟨hp0, d, off, k, m, hc⟩ | ⟨hp, parent, d, off, k, m, hsc, hc⟩
      · have hpass' := step_passing P hroot hB hret harg U B s s' hs hx
          (fun hc' => by have h2 := hc'.2; rw [hp0] at h2; cases h2) hpass
        refine run_passing U B n (by omega) s' b hs' ?_ hpass' h'
        rw [hc]
        exact Or.inr hu
      · by_cases hU : s.1 = U
        · obtain ⟨sc, ⟨p, i, j⟩⟩ := s
          have hp' : p = true := hp
          have hU' : sc = U := hU
          rw [hp', hU'] at h hpass
          exact Or.inr ⟨i, j, passing_exit P U i j hpass, resolve_of_run P hn h⟩
        · have hpass' := step_passing P hroot hB hret harg U B s s' hs hx (fun hc' => hU hc'.1) hpass
          have hup : Under U parent := by
            rw [hsc] at hu hU
            have hu' : U = .child parent d off k m ∨ Under U parent := hu
            rcases hu' with hu' | hu'
            · exact absurd hu'.symm hU
            · exact hu'
          rcases hc with hc | ⟨off', m', hc⟩
          · refine run_passing U B n (by omega) s' b hs' ?_ hpass' h'
            rw [hc]
            exact hup
          · refine run_passing U B n (by omega) s' b hs' ?_ hpass' h'
            rw [hc]
            exact Or.inr hup

/-- **A leaf an activation returns is one of its gates, or a leaf of a parameter its Definition passes on.** -/
theorem ret_passed_on (U : Scope) (hU : Good P U) (t : ℕ) (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨
      ∃ i j, i ∈ passOf P U.d ∧ resolve P U ⟨true, i, j⟩ = .ok b := by
  obtain ⟨hd, hpc⟩ := good_d P hroot hB U hU
  obtain ⟨q, hq, t', ht', hget⟩ := seq_part _ (hret U.d hd hpc) t ht
  exact run_passing P hroot hB hL hret harg U _ Flock.Steps.FUEL (Nat.le_refl _) _ b
    (down_ok P hroot hB hL _ U hU (Nat.le_refl _) t ht) (under_refl U) (passing_start P U hd hpc hq ht' hget) h

end

/-- What resolution takes of a Program read (one that decodes, with no order violation and its bodies laid out):
its root is a composite, every activation's body is `BodyOk` and laid out, a composite's parameters are its body's,
and a return and each argument of each node are the sequences of their parts. -/
theorem decoded_walk (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) :
    (P.root < P.defs.size ∧ (P.defs[P.root]!).prim = false) ∧ (∀ sc, Good P sc → BodyOk P sc.d) ∧
      (∀ d < P.defs.size, Walk.LaidOut P d) ∧
      (∀ d < P.defs.size, (P.defs[d]!).prim = false → (P.defs[d]!).body.params = (P.defs[d]!).params) ∧
      (∀ d < P.defs.size, (P.defs[d]!).prim = false →
        (P.defs[d]!).body.ret = Seq.ofParts (P.defs[d]!).body.ret.parts) ∧
      ∀ d < P.defs.size, ∀ k < (P.defs[d]!).body.nodes.size, ∀ i < ((P.defs[d]!).body.nodes[k]!).args.size,
        (((P.defs[d]!).body.nodes[k]!).args[i]!).seq =
          Seq.ofParts (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts := by
  obtain ⟨hr, hrp, -⟩ := BodiesOk.decode_root bytes P hdec
  obtain ⟨-, hL, -, -⟩ := Walk.bodiesLaidOut_spec P hlay
  obtain ⟨st, hP, hinv, -⟩ := BodiesOk.decode_spec bytes P hdec
  obtain ⟨S, hS0, hS⟩ := BodiesOk.order_closed P hov
  have hok : ∀ i < P.defs.size, BodiesOk.DefOk P.defs i P.defs[i]! := by rw [hP]; exact hinv.1
  refine ⟨⟨hr, hrp⟩, fun sc hg => ?_, hL, fun d hd hp => ((hok d hd).1.2 hp).1, fun d hd _ => ?_,
    fun d hd k hk i hi => ?_⟩
  · obtain ⟨hSd, hpd⟩ := BodiesOk.good_closed P S hS0 hrp (fun d hd hp => (hS d hd hp).2) sc hg
    exact BodiesOk.body_ok P hok sc.d (hS sc.d hSd hpd).1
  · obtain ⟨-, -, -, ps, hps, -⟩ := hok d hd
    have hparts : (P.defs[d]!).body.ret.parts = ps := by rw [hps]; exact (Walk.ofParts_spec ps).1
    rw [hparts]
    exact hps
  · obtain ⟨-, hnodes, -⟩ := hok d hd
    obtain ⟨-, -, hargs⟩ := hnodes k hk
    obtain ⟨⟨ps, hps, -⟩, -⟩ := hargs i hi
    have hparts : (((P.defs[d]!).body.nodes[k]!).args[i]!).seq.parts = ps := by
      rw [hps]; exact (Walk.ofParts_spec ps).1
    rw [hparts]
    exact hps

/-- **`ret_passed_on`, on a Program read**: one that decodes, with no order violation and its bodies laid out
(`Desc.bodiesLaidOut`, which `Desc.walkable` holds). -/
theorem ret_passed_on_of (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (U : Scope) (hU : Good P U) (t : ℕ)
    (ht : t < (P.defs[U.d]!).body.ret.count) {b : ℕ}
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    (U.off ≤ b ∧ b < U.off + (P.defs[U.d]!).gates) ∨
      ∃ i j, i ∈ passOf P U.d ∧ resolve P U ⟨true, i, j⟩ = .ok b := by
  obtain ⟨hroot, hB, hL, -, hret, harg⟩ := decoded_walk bytes P hdec hov hlay
  exact ret_passed_on P hroot hB hL hret harg U hU t ht h

/-- **A gate an activation reads through a parameter and returns comes through a parameter `Record.passedOn`
lists.** A parameter's leaf resolves to a gate before the activation (`resolve_lt`), so the gate is none of the
activation's own, and `ret_passed_on` names a listed parameter one of whose leaves it is. So where each gate is the
leaf of one parameter at most, a parameter `passedOn` does not list has no leaf the activation returns. -/
theorem param_returned_of (bytes : ByteArray) (P : Prog) (hdec : Flock.Program.decode bytes = .ok P)
    (hov : P.orderViolation = none) (hlay : Desc.bodiesLaidOut P = true) (U : Scope) (hU : Good P U) (i j : ℕ)
    (hi : i < (P.defs[U.d]!).params.size) (hj : j < ((P.defs[U.d]!).params[i]!).2.leaves) {b : ℕ}
    (hb : resolve P U ⟨true, i, j⟩ = .ok b) (t : ℕ) (ht : t < (P.defs[U.d]!).body.ret.count)
    (h : resolve P U ((P.defs[U.d]!).body.ret.get t) = .ok b) :
    ∃ i' j', i' ∈ passOf P U.d ∧ resolve P U ⟨true, i', j'⟩ = .ok b := by
  obtain ⟨hroot, hB, hL, hprm, hret, harg⟩ := decoded_walk bytes P hdec hov hlay
  obtain ⟨hd, hpc⟩ := good_d P hroot hB U hU
  have hlt : b < U.off :=
    resolve_lt P hroot hB hL U.off U ⟨true, i, j⟩
      ⟨hU, ⟨fun _ => by rw [hprm U.d hd hpc]; exact ⟨hi, hj⟩, fun hc => by cases hc⟩, fun _ => Nat.le_refl _,
        fun hc => by cases hc⟩ hb
  rcases ret_passed_on P hroot hB hL hret harg U hU t ht h with ⟨hge, -⟩ | hx
  · omega
  · exact hx

end NciCheck.Properties.SecurityProofs.Resolve
