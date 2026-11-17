module

public import Proofs.Flock.Claims.Recursion.ResLowerD
public import Proofs.Flock.Claims.Recursion.ResAccOff
import Mathlib.Algebra.Order.Floor.Extended
import Mathlib.Algebra.Order.Interval.Basic
import Mathlib.Analysis.Complex.UpperHalfPlane.Basic
import Mathlib.Analysis.SpecialFunctions.Bernstein
import Mathlib.Analysis.SpecialFunctions.Gamma.Basic
import Mathlib.Analysis.SpecialFunctions.Trigonometric.DerivHyp
import Mathlib.Combinatorics.Enumerative.DyckWord
import Mathlib.Combinatorics.SimpleGraph.Triangle.Removal
import Mathlib.Data.NNRat.Floor
import Mathlib.Data.Nat.Choose.Multinomial
import Mathlib.Geometry.Euclidean.Altitude
import Mathlib.NumberTheory.Chebyshev
import Mathlib.NumberTheory.Height.NumberField
import Mathlib.NumberTheory.Height.Projectivization
import Mathlib.NumberTheory.LucasLehmer
import Mathlib.NumberTheory.SelbergSieve
import Mathlib.RingTheory.Radical.NatInt
import Mathlib.Tactic.Echelon.Zsqrtd
import Mathlib.Tactic.NormNum.Irrational
import Mathlib.Tactic.NormNum.IsCoprime
import Mathlib.Tactic.NormNum.IsSquare
import Mathlib.Tactic.NormNum.LegendreSymbol
import Mathlib.Tactic.NormNum.ModEq
import Mathlib.Tactic.NormNum.NatFib
import Mathlib.Tactic.NormNum.NatLog
import Mathlib.Tactic.NormNum.NatSqrt
import Mathlib.Tactic.NormNum.Ordinal
import Mathlib.Tactic.NormNum.Parity
import Mathlib.Tactic.NormNum.Prime
import Mathlib.Tactic.NormNum.RealSqrt
import Mathlib.Topology.Sheaves.Init
import Proofs.Flock.C3.Reps.AtList
import Proofs.Flock.Claims.Recursion.DecodeAt
import Proofs.Flock.Claims.Recursion.VerdictLoop
import Proofs.Flock.Claims.Recursion.VerdictRowDTie
import Proofs.Flock.Claims.Recursion.VerdictRowG

/-! The steps of `resLowered_D`: which part owns a piece, check 5's tie, and the residuals. -/

@[expose] public section

namespace RecDeploy.DecompBR

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)

theorem inner_named {i : RecVStar.Inner} {ss : Array VStar.Stmt} (h : RecVStar.statements i = .ok ss) {s j : ℕ}
    {nm : String} {pt : VStar.Part} (hs : ss[s]? = some (.inner j nm pt)) :
    nm = RecResiduals.unitName i.m i.kLog i.claims j := by
  obtain ⟨sc, ps, C, Ks, -, hC, hKs, hss⟩ := FlockC3.Reps.getElem?_list h
  rw [hss s] at hs
  split at hs
  · cases hs
  split at hs
  · simp only [Option.some.injEq, VStar.Stmt.inner.injEq] at hs
    obtain ⟨rfl, rfl, -⟩ := hs
    rfl
  · rcases Array.mem_append.1 (Array.mem_of_getElem? hs) with hm | hm
    · rcases FlockC3.Reps.candidate3_kinds hC _ hm with ⟨_, _, he⟩ | ⟨_, _, _, _, he⟩ |
        ⟨_, _, _, _, _, _, he⟩ <;> cases he
    · rcases hKs _ hm with ⟨_, _, he⟩ | ⟨_, _, _, he⟩ <;> cases he

theorem ownRD_some {x : Msg} {q : ResPiece} {s : ℕ} (ho : ownRD x q = some s) :
    ∃ (r : XR) (ss : Array VStar.Stmt) (nm : String) (pt : VStar.Part), readX x = some r ∧
      RecVStar.statements r.shape = .ok ss ∧ ss[s]? = some (.inner q.key.1 nm pt) := by
  unfold ownRD at ho
  cases hr : readX x with
  | none => simp [hr] at ho
  | some r =>
    rw [hr, Option.bind_some] at ho
    cases hss : RecVStar.statements r.shape with
    | error e => rw [hss] at ho; simp [Except.toOption] at ho
    | ok ss =>
      rw [hss] at ho
      simp only [Except.toOption, Option.bind_some] at ho
      obtain ⟨hsz, hp, -⟩ := Array.findIdx?_eq_some_iff_getElem.1 ho
      have key : ∀ t : VStar.Stmt, ss[s]? = some t → isResOf q.key.1 t = true →
          ∃ nm pt, ss[s]? = some (.inner q.key.1 nm pt) := by
        intro t ht hp
        cases t with
        | inner j nm pt =>
          have hj : j = q.key.1 := by simpa [isResOf] using hp
          subst hj
          exact ⟨nm, pt, ht⟩
        | _ => simp [isResOf] at hp
      obtain ⟨nm, pt, hget⟩ := key ss[s] (Array.getElem?_eq_getElem hsz) hp
      exact ⟨r, ss, nm, pt, by first | rfl | exact hr, by first | rfl | exact hss, hget⟩

theorem mem_resPD' {x : Msg} {q : ResPiece} (hq : q ∈ resPD x) :
    ∃ v, setupOf false x = some v ∧ ∃ (p r : ℕ) (S : RecOpen.GfS) (ns : List ℕ) (nv : ℕ), r < 2 ∧
      RecResiduals.part (shI v.2.1 v.2.2) p = .ok (S, ns, nv) ∧ q.key = (p, r, 0, 0) ∧ q.S = S := by
  unfold resPD at hq
  cases hv : setupOf false x with
  | none => rw [hv] at hq; cases hq
  | some v =>
    rw [hv] at hq
    obtain ⟨p, -, hq⟩ := List.mem_flatMap.1 hq
    refine ⟨v, rfl, ?_⟩
    unfold resGroup at hq
    cases hp : RecResiduals.part (shI v.2.1 v.2.2) p with
    | error e => rw [hp] at hq; simp at hq
    | ok t =>
      obtain ⟨S, ns, nv⟩ := t
      rw [hp] at hq
      try dsimp only at hq
      rcases List.mem_cons.1 hq with rfl | hq
      · exact ⟨p, 0, S, ns, nv, by omega, hp, rfl, rfl⟩
      · rw [List.mem_singleton] at hq
        subst hq
        exact ⟨p, 1, S, ns, nv, by omega, hp, rfl, rfl⟩

end RecDeploy.DecompBR

namespace RecDeploy.ResLower

open Flock
open Claims (Msg)
open FlockDeploy (Coin FinalP setupOf)
open Claims.Recursion.VStarB

noncomputable section

theorem part_partS {sh : _root_.Flock.Shape} {j : ℕ} {S : _root_.Flock.RecOpen.GfS} {ns : List ℕ} {nv : ℕ}
    (h : _root_.Flock.RecResiduals.part sh j = .ok (S, ns, nv)) :
    ∃ (S0 : _root_.Flock.RecAlgebra.GfS) (gs : Array (Array ℕ × Array ℕ × Array ℕ)) (R P V : Array ℕ),
      _root_.Flock.RecAlgebra.structureOf sh = .ok S0 ∧
      _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes sh) = .ok gs ∧
      gs[j]? = some (R, P, V) ∧
      _root_.Flock.RecResiduals.partS S0 (_root_.Flock.RecResiduals.portSizes sh) R P V = .ok S := by
  unfold _root_.Flock.RecResiduals.part at h
  obtain ⟨S0, hS0, h⟩ := FlockVBridge.PartS.bind_ok h
  dsimp only at h
  split at h
  · cases h
  · obtain ⟨gs, hgs, h⟩ := FlockVBridge.PartS.bind_ok h
    try dsimp only at h
    cases hj : gs[j]? with
    | none => rw [hj] at h; cases h
    | some t =>
      obtain ⟨R, P, V⟩ := t
      rw [hj] at h
      try dsimp only at h
      obtain ⟨S2, hS2, h⟩ := FlockVBridge.PartS.bind_ok h
      simp only [pure, Except.pure, Except.ok.injEq, Prod.mk.injEq] at h
      obtain ⟨rfl, -, -⟩ := h
      exact ⟨S0, gs, R, P, V, hS0, hgs, hj, hS2⟩

theorem decAt_of {rd : Msg → Option FinalP} (hBs : BasesHold rd) {xo : Msg} {sto : Claims.Recursion.CompiledStatement}
    (hst : parse xo = some sto) (hf : Claims.Recursion.Form.Dense sto.form) {ms : List Msg} {cs : List Coin}
    (hr : FlockDeploy.ReadOkR rd (xIOf xo) ms cs) (hlen : ms.length = cs.length) :
    Nonempty (Rows.DecAt rd (xIOf xo) ms cs) := by
  have hr' := hr
  unfold FlockDeploy.ReadOkR at hr'
  cases h : FlockDeploy.actR rd (xIOf xo) ms cs with
  | rej => rw [h] at hr'; exact False.elim hr'
  | coin _ => rw [h] at hr'; exact False.elim hr'
  | decide s spec sts link c₁ mids f =>
    have hd := FlockDeploy.actR_decide h
    have hs := setupWith_sha512_of (actR_decide_setup h).1
    obtain ⟨sess, hsess⟩ := DecodeAt.decode_at rd (fun _ _ _ => True) (xIOf xo) ms cs
      ⟨⟨xo, sto, hst, hf, rfl⟩, hr, fun b hb => hBs xo sto hst b hb ms cs hlen, trivial⟩ _ hd spec sts hs
    exact ⟨⟨_, hd, spec, sts, hs, sess, hsess⟩⟩

theorem shI_alg {rd : Msg → Option FinalP} {xi : Msg} {ms : List Msg} {cs : List Coin} (D : Rows.DecAt rd xi ms cs)
    {xo : Msg} {sto : Claims.Recursion.CompiledStatement} (hst : parse xo = some sto)
    (hf : Claims.Recursion.Form.Dense sto.form) (hx : xi = xIOf xo)
    {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt} (hv : setupOf false xi = some v) :
    DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2) := by
  obtain ⟨h1, -⟩ := D.tables_of_dense hst hf hx
  have h0 : 0 < D.sts.size := by omega
  have he := RecDeploy.DecompBR.DecAt.algSh_tie D RecDeploy.Top.cls₀ hst hf hx h0 hv
  have hG := D.rowG RecDeploy.Top.cls₀ (fun _ => #[]) hst hf hx h0
  have hsh := D.sh_eq RecDeploy.Top.cls₀ (fun _ => #[]) hG h0
  obtain ⟨hsp, hsts⟩ := D.setup_eq
  obtain ⟨v', c₁, r₀, rss, fl, link, pf, hv', -, -, -, -, -, -, hd⟩ := Rows.decR_last D.dec
  have hvv : v' = v := Option.some.inj (hv'.symm.trans hv)
  have hds : D.d.spec = v.2.1 := by rw [hd, hvv]
  have hdt : D.d.sts = v.2.2 := by rw [hd, hvv]
  rw [← he, hsh, hsp, hsts, hds, hdt]

theorem elemPort_eq {xi : Msg} {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}
    (hv : setupOf false xi = some v)
    (hI : DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2))
    {S0 : _root_.Flock.RecAlgebra.GfS} {gs : Array (Array ℕ × Array ℕ × Array ℕ)} {R P V : Array ℕ}
    (hS : _root_.Flock.RecAlgebra.structureOf (DecompBR.shI v.2.1 v.2.2) = .ok S0)
    (hg : _root_.Flock.RecResiduals.groups S0 (_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) =
      .ok gs)
    {jj : ℕ} (hj : gs[jj]? = some (R, P, V)) (e : ℕ) :
    elemPort xi jj e =
      (FlockVBridge.PartS.wIdx (_root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2)) P)[e]? := by
  have hps : _root_.Flock.RecResiduals.portSizes (Rows.shapeNoCoin v.2.1 v.2.2) =
      _root_.Flock.RecResiduals.portSizes (DecompBR.shI v.2.1 v.2.2) := by
    rw [hI, Rows.portSizes_alg]
  have hS' : _root_.Flock.RecAlgebra.structureOf (Rows.shapeNoCoin v.2.1 v.2.2) = .ok S0 := by
    rw [← Rows.structureOf_alg, ← hI]
    exact hS
  rw [← hps] at hg ⊢
  unfold elemPort Rows.portAt
  simp [hv, hS', hg, hj]

theorem accBase_eq {xi : Msg} {v : FlockDeploy.Ins × _root_.Flock.SessionSpec × Array _root_.Flock.Stmt}
    (hv : setupOf false xi = some v)
    (hI : DecompBR.shI v.2.1 v.2.2 = Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2)) :
    accBase xi = _root_.Flock.RecAlgebra.nR (DecompBR.shI v.2.1 v.2.2) := by
  have hr : _root_.Flock.RecAlgebra.rounds (Rows.algShape (Rows.shapeNoCoin v.2.1 v.2.2)) =
      _root_.Flock.RecAlgebra.rounds (Rows.shapeNoCoin v.2.1 v.2.2) :=
    OpenLower.rounds_congr rfl
  unfold accBase
  rw [hv, hI]
  all_goals exact congrArg (fun a => Array.size (Array.flatten a)) hr.symm

theorem shift_of {E E' : ℕ → _root_.Flock.F128} {wI : Array ℕ} {A : ℕ} {δ : ℕ → _root_.Flock.F128}
    {port : ℕ → Option ℕ} (hp : ∀ e, port e = wI[e]?)
    (hlo : ∀ e j, port e = some j → j < A → E' e = E e)
    (hhi : ∀ e j, port e = some j → A ≤ j → E' e = E e + δ ((j - A) % 2)) :
    ∀ e < wI.size, E' e = E e + FlockVBridge.PartS.offOf A wI δ e := by
  intro e he
  have hje : wI[e]? = some wI[e] := Array.getElem?_eq_getElem he
  have hpe : port e = some wI[e] := (hp e).trans hje
  by_cases hlt : wI[e] < A
  · rw [hlo e _ hpe hlt, FlockVBridge.PartS.offOf_low hje hlt, FlockVBridge.PartS.f128_add_zero']
  · have hle : A ≤ wI[e] := Nat.le_of_not_lt hlt
    rw [hhi e _ hpe hle, FlockVBridge.PartS.offOf_acc (c := wI[e] - A)
      (by rw [Nat.add_sub_of_le hle]; exact hje)]

end

end RecDeploy.ResLower
