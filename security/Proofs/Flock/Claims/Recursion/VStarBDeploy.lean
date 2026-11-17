module

public import Proofs.Flock.Claims.Recursion.BoundDeploy
public import Proofs.Flock.Claims.Recursion.DeployChecked
public import Proofs.Flock.Claims.Recursion.VStarLinks
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
import Proofs.Flock.Claims.Recursion.BoundAccepts
import Proofs.Flock.Claims.Recursion.DeployMargin
import Proofs.Flock.Claims.Recursion.DeployScope
import Proofs.Flock.Claims.Recursion.DeploySessions
import Std.Tactic.BVDecide.Normalize.Prop

/-! V\*'s bound-layout premises at the deployment, where `VstarShowsB`, and `HeldB` at V\*'s registered ports, are false, so the route reads `StagedB` instead and `PartChecksB`, `PartOpensB` and `DecompB` stay premises. -/

@[expose] public section

namespace Claims.Recursion.VStarB

open Claims

section Staged

variable {Coin W : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

def StagedB (H : Msg → Msg) (D : Deploy Coin) (I : Inner Coin W) (L : LayoutB) (shape : Form → Prop)
    (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) : Prop :=
  ∀ (x : Msg) (st : CompiledStatement), D.parse x = some st → shape st.form →
    ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin), (run H P D.V x ω r).out = .acc →
      ∀ s < D.nS x, shows s x ((runVstar H D I P x ω r).getD s []) (coinSeq r) ∧
        ∀ i, i + 1 < L.rounds x →
          cmt x i ((runVstar H D I P x ω r).getD s []) = ((run H P D.V x ω r).msgs.map Prod.fst).getD i []

end Staged

end Claims.Recursion.VStarB

namespace RecDeploy

open Claims (Msg Out)
open FlockDeploy (Coin)

def badRecord : ByteArray := ⟨#[255]⟩

theorem badRecord_validate : ByteArray.validateUTF8 badRecord = false := by
  unfold ByteArray.validateUTF8
  rw [ByteArray.validateUTF8.go]
  decide +kernel

theorem badRecord_utf8 : String.fromUTF8? badRecord = none := by
  have h : ¬ badRecord.IsValidUTF8 := ByteArray.validateUTF8_eq_false_iff.1 badRecord_validate
  unfold String.fromUTF8?
  rw [dif_neg h]

def badStage : Msg := encStage [] badRecord ByteArray.empty

theorem decodeStage_badStage : decodeStage badStage = some ([], badRecord, ByteArray.empty) := by
  rfl

theorem setupO_staged_none {st : Stage} {x : Msg} {ms : List (Msg × Out)} {s : ℕ}
    (hrec : String.fromUTF8? (st.record x ms) = none) : setupO (staged st x ms s) = none := by
  rcases hso : setupO (staged st x ms s) with _ | ⟨i, spec, sts⟩
  · rfl
  exfalso
  have h := (setupO_some hso).1
  have hne : staged st x ms s ≠ [] := fun he => by rw [he, insO_nil] at h; cases h
  obtain ⟨r, p, -, -, hd⟩ := staged_decode hne
  unfold insO at h
  rw [hd] at h
  simp only [Option.bind_eq_bind, Option.bind_some] at h
  split_ifs at h
  · simp at h
  have h' : (partRaw r st x ms s p).ins.toOption = some i := by simpa using h
  obtain ⟨-, hsrc⟩ := RecRaw.ins_inner_src (toOption_eq_some h')
  unfold sourceOf at hsrc
  rw [formOf_partRaw] at hsrc
  dsimp only [partRaw] at hsrc
  cases hq : r.innerRaw.partition with
  | none => rw [hq] at hsrc; simp at hsrc
  | some q =>
    rw [hq] at hsrc
    simp only [bind, Except.bind] at hsrc
    cases hj : FlockDeploy.jsonOf "--rec-inner-partition" q with
    | error e => rw [hj] at hsrc; cases hsrc
    | ok pj =>
      rw [hj] at hsrc
      cases ht : textOf "rec inner session" (st.record x ms) with
      | error e => rw [ht] at hsrc; cases hsrc
      | ok rec => simp [textOf, hrec] at ht

theorem vstarShowsB_deployRec_forces {W : Type} {IV : Claims.Verifier Coin} {ivScope : Msg → Bool} {b : Brings}
    {I : Claims.Recursion.Inner Coin W} {L : Claims.Recursion.VStarB.LayoutB} {shape : Claims.Recursion.Form → Prop}
    {decC : Msg → Msg → Coin} {shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop} {cmt : Msg → ℕ → Msg → Msg}
    (hV : Claims.Recursion.VStarB.VstarShowsB (deployRec IV ivScope b) I L shape decC shows cmt)
    {x : Msg} {st : Claims.Recursion.CompiledStatement} (hst : parse x = some st) (hf : shape st.form) {s : ℕ} (hs : s < nS x)
    (hset : ∃ ms, (setupO (staged b.st x ms s)).isSome) (hk : 0 < I.V.k (xIOf x)) (hr : 1 < L.rounds x) :
    False := by
  let ps : Msg → List (Msg × Out) := fun a =>
    ((a, Out.msg []) :: List.replicate (I.V.k (xIOf x) - 1) (([] : Msg), Out.msg [])) ++ [(badStage, Out.msg [])]
  have hlen : ∀ a, (ps a).length =
      I.V.k (Claims.Recursion.xI (deployRec IV ivScope b) x) + (deployRec IV ivScope b).kS x := by
    intro a
    show _ = I.V.k (xIOf x) + 1
    simp only [ps, List.length_append, List.length_cons, List.length_replicate, List.length_nil]
    omega
  have hmsg : ∀ a, ∀ p ∈ ps a, p.2 = Out.msg [] := by
    intro a p hp
    simp only [ps, List.mem_append, List.mem_cons, List.mem_replicate] at hp
    rcases hp with (rfl | ⟨-, rfl⟩) | (rfl | h)
    · rfl
    · rfl
    · rfl
    · simp at h
  have hc : ∀ a, ∀ i < (ps a).length, ∃ m, ((ps a).getD i ([], .rej)).2 = .msg m ∧ decC x m = decC x [] := by
    intro a i hi
    refine ⟨[], ?_, rfl⟩
    rw [List.getD_eq_getElem?_getD, List.getElem?_eq_getElem hi, Option.getD_some]
    exact hmsg a _ (List.getElem_mem hi)
  have hpick : (setupO (staged b.st x (pick b.st x s) s)).isSome := by
    unfold pick
    rw [dif_pos hset]
    exact Classical.choose_spec hset
  obtain ⟨⟨i, spec, sts⟩, ht⟩ := Option.isSome_iff_exists.1 hpick
  have hv : ∀ a, ((deployRec IV ivScope b).vstar x (ps a)).getD s [] = staged b.st x (pick b.st x s) s := by
    intro a
    show (vstar b.st x (ps a)).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [if_neg]
    intro heq
    have hk' := congrArg Claims.Recursion.Nums.rounds heq
    change VO.k (staged b.st x (ps a) s) = VO.k (staged b.st x (pick b.st x s) s) at hk'
    have hl : stageOf (ps a) = some ([], badRecord, ByteArray.empty) := by
      show decodeStage (((ps a).getLast?.map Prod.fst).getD []) = _
      rw [show (ps a).getLast? = some (badStage, Out.msg []) from List.getLast?_concat]
      exact decodeStage_badStage
    have hbad : setupO (staged b.st x (ps a) s) = none := by
      apply setupO_staged_none
      show String.fromUTF8? (((stageOf (ps a)).map (·.2.1)).getD ByteArray.empty) = none
      rw [hl]
      exact badRecord_utf8
    simp only [VO, hbad, ht] at hk' <;> omega
  have h1 := (hV x st hst hf (ps []) (hlen _) (fun _ => decC x []) (hc []) s hs).2 0 hr
  have h2 := (hV x st hst hf (ps [0]) (hlen _) (fun _ => decC x []) (hc [0]) s hs).2 0 hr
  rw [hv] at h1 h2
  have e : ∀ a, ((ps a).map Prod.fst).getD 0 [] = a := fun a => rfl
  rw [e] at h1 h2
  exact absurd (h1.symm.trans h2) (by simp)

/-- Since an honest run is accepted, `VstarShowsB` is false at the deployment wherever an honest run reaches a statement of the shape. -/
theorem vstarShowsB_deployed_rejects (rel : _root_.Flock.Readable.Parsed → _root_.Flock.Readable.Relation)
    (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (nextE : Msg → List Msg → List Coin → Claims.OC Out)
    (colsE : Claims.Seen → ℕ → List (ℕ × List FlockLevel3.GF128)) (wE : Msg → ℝ) (join : List Msg → Msg)
    (rowIn : Msg → Msg) (final : Msg → Option (List (List (Msg × Salt)) × Msg)) (w : Msg → ℝ)
    (decI : List (ℕ × List FlockLevel3.GF128) → List Claims.Flock.PlainTrace) (shape : Claims.Recursion.Form → Prop)
    (decC : Msg → Msg → Coin) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop) (cmt : Msg → ℕ → Msg → Msg) :
    let Bd := bdDeployed rel nextE colsE wE join rowIn final w
    let D := recBound rel b L Bd.IVStar
    Claims.Recursion.VStarB.HeldB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape →
    Claims.Recursion.VStarB.VstarShowsB D (Claims.Recursion.Inner.ofFlock Bd.deploy decI) L shape decC shows cmt →
    ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), parse x = some st → shape st.form → 0 < nS x →
      ∀ (H : Msg → Msg) (Ω : Type) (P : Claims.Prover Ω) (ω : Ω) (r : Fin (D.V.k x) → Coin),
        (Claims.run H P D.V x ω r).out ≠ .acc := by
  intro Bd D hH hV x st hst hf hn H Ω P ω r hacc
  let I := Claims.Recursion.Inner.ofFlock Bd.deploy decI
  let bL : Brings := { b with rd := L.readers }
  obtain ⟨ms, i, spec, sts, hxO, hso, -⟩ := stagedPart_run H Bd.IVStar bL.st x P ω r hacc (s := 0) hn
  have hset : ∃ ms, (setupO (staged bL.st x ms 0)).isSome := ⟨ms, by rw [← hxO, hso]; rfl⟩
  have hk2 : 2 ≤ (FlockDeploy.V false).k (xIOf x) := by
    obtain ⟨s', spec', sts', hs', -⟩ := parse_innerOk hst
    simp only [FlockDeploy.V, hs'] <;> omega
  obtain ⟨hK, -, -⟩ := hH x st hst hf
  have hK' : (FlockDeploy.V false).k (xIOf x) = L.rounds x := hK
  have hk : 0 < I.V.k (xIOf x) := by
    show 0 < (FlockDeploy.V false).k (xIOf x)
    omega
  exact vstarShowsB_deployRec_forces (IV := Bd.IVStar) (ivScope := (FlockDeploy.deployPlain rel).scopeCheck)
    (b := bL) (I := I) hV hst hf hn hset hk (by omega)

def layoutBOf : Claims.Recursion.VStarB.LayoutB := Claims.Recursion.VStarB.LayoutB.ofLayout layoutOf

theorem heldB_layoutBOf_iff {W : Type} (D : Claims.Recursion.Deploy Coin) (I : Claims.Recursion.Inner Coin W)
    (shape : Claims.Recursion.Form → Prop) :
    Claims.Recursion.VStarB.HeldB D I layoutBOf shape ↔
      ∀ (x : Msg) (st : Claims.Recursion.CompiledStatement), D.parse x = some st → ¬ shape st.form := by
  constructor
  · intro h x st hst hf
    obtain ⟨hK, hcl, -⟩ := h x st hst hf
    have hpos : 0 < I.V.k (Claims.Recursion.xI D x) := by
      rw [hK]
      exact layoutOf_rounds_pos x
    rcases (hcl 0 hpos).1 with h0 | ⟨s, -, h0⟩ | ⟨h0, -⟩
    · have h' : layoutBOf.fixedAt x 0 = none := rfl
      rw [h'] at h0
      simp at h0
    · have h' : layoutBOf.msgAt x s 0 = none := rfl
      rw [h'] at h0
      simp at h0
    · exact h0 rfl
  · intro h x st hst hf
    exact absurd hf (h x st hst)

theorem vstarAccepts_bound_staged {W Col T : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W)
    (ivScope : Msg → Bool) (b : Brings) (L : Claims.Recursion.VStarB.LayoutB) (DO : Claims.Flock.Deploy Coin Col T)
    (shape : Claims.Recursion.Form → Prop) (decC : Msg → Msg → Coin)
    (chk : ℕ → Msg → (ℕ → Coin) → List Msg → Prop) (shows : ℕ → Msg → Msg → (ℕ → Coin) → Prop)
    (cmt : Msg → ℕ → Msg → Msg)
    (hplay : ∀ x ms (c : Fin (I.V.k (xIOf x)) → Coin),
      (Claims.run H (Claims.replay ms) I.V (xIOf x) () c).msgs.length = I.V.k (xIOf x))
    (hRP : Claims.Recursion.VStar.RunPairs H (deployRec I.V ivScope { b with rd := L.readers }) I decC)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope { b with rd := L.readers }) I L shape)
    (hSt : Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope { b with rd := L.readers }) I L shape shows cmt)
    (hChk : ∀ s, Claims.Recursion.VStarB.PartChecksB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt chk s)
    (hOpen : ∀ s, Claims.Recursion.VStarB.PartOpensB H (deployRec I.V ivScope { b with rd := L.readers }) DO L shape
      shows cmt s)
    (hDec : Claims.Recursion.VStarB.DecompB H (deployRec I.V ivScope { b with rd := L.readers }) I.V L shape chk)
    (hRows : L.ReadsRows H)
    (hRRP : ReadersRunPlayed H I.V { b with rd := L.readers } DO) :
    Claims.Recursion.VStarAccepts H (deployRec I.V ivScope { b with rd := L.readers }) I DO shape := by
  intro x st hst hf Ω P ω r hacc ws ho htrue
  obtain ⟨hlenR, -⟩ := hRP x Ω P ω r hacc
  obtain ⟨hK, hcl, hlast⟩ := hHeld x st hst hf
  change I.V.k (xIOf x) + kS x ≤ _ at hlenR
  change I.V.k (xIOf x) = L.rounds x at hK
  have hsh : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      shows s x (((deployRec I.V ivScope { b with rd := L.readers }).vstar x
        ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
          (I.V.k (xIOf x) + kS x))).getD s []) (Claims.coinSeq r) ∧
      ∀ i, i + 1 < L.rounds x →
        cmt x i (((deployRec I.V ivScope { b with rd := L.readers }).vstar x
          ((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
            (I.V.k (xIOf x) + kS x))).getD s []) =
        (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
          (I.V.k (xIOf x) + kS x)).map Prod.fst).getD i [] := by
    intro s hs
    obtain ⟨h1, h2⟩ := hSt x st hst hf Ω P ω r hacc s hs
    refine ⟨h1, fun i hi => (h2 i hi).trans ?_⟩
    have hin : i < I.V.k (xIOf x) + kS x := by omega
    rw [List.getD_eq_getElem?_getD, List.getD_eq_getElem?_getD, List.map_take, List.getElem?_take_of_lt hin]
  have hlA := runAs_length H I.V { b with rd := L.readers } x P ω r hacc
  have hcA : (runCs (I.V.k (xIOf x) + kS x) r).length = _ :=
    (runCs_length I.V { b with rd := L.readers } x r).trans hlA.symm
  have hoA := ho
  rw [runVstar_staged H I ivScope { b with rd := L.readers } P ω r hacc] at hoA
  rcases hRRP x _ _ ws hlA hcA (stagedAll_run H I.V { b with rd := L.readers } x P ω r hacc) hoA htrue with
    hag | hcoll
  swap
  · exact Or.inr hcoll
  have hrd : ∀ j, j + 1 < I.V.k (xIOf x) → (L.msgs x ws).getD j [] =
      (((Claims.run H P (deployRec I.V ivScope { b with rd := L.readers }).V x ω r).msgs.take
        (I.V.k (xIOf x) + kS x)).map Prod.fst).getD j [] := by
    intro j hj
    have hj' : j < (Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.length := by
      rw [hplay]
      omega
    have h1 : ((Claims.run H (Claims.replay (L.msgs x ws)) I.V (xIOf x) ()
        (fun i => (runCs (I.V.k (xIOf x) + kS x) r).getD i.val default)).msgs.map Prod.fst).getD j [] = _ :=
      hag j hj' hj
    rw [Claims.Recursion.VStar.run_replay_fst H I.V (xIOf x) (L.msgs x ws) _ hj'] at h1
    exact h1
  by_cases hpc : ∃ s, ∃ hs : s < ws.length, (Claims.findColl (L.calls x s ws[s])).isSome
  · obtain ⟨s, hs, h⟩ := hpc
    exact Or.inr (Claims.Recursion.VStar.coll_of_part (calls := L.calls) hs htrue h)
  have hpc' : ∀ s (hs : s < ws.length), ¬ (Claims.findColl (L.calls x s ws[s])).isSome :=
    fun s hs h => hpc ⟨s, hs, h⟩
  have hchk : ∀ s < (deployRec I.V ivScope { b with rd := L.readers }).nS x,
      chk s x (Claims.coinSeq r) (L.msgs x ws) := by
    intro s hs
    have hsw : s < ws.length := ho.1 ▸ hs
    have hts : ∀ e ∈ L.calls x s ws[s], H e.1 = e.2 :=
      fun e he => htrue e (Claims.Recursion.VStar.mem_calls_of_part hsw he)
    have hshs := hsh s hs
    rcases hOpen s x st hst hf _ ws[s] (ho.2 s hsw) hts with hop | hop
    swap
    · exact absurd hop (hpc' s hsw)
    rcases hChk s x st hst hf _ ws[s] (ho.2 s hsw) hts with hck | hck
    swap
    · exact absurd hck (hpc' s hsw)
    refine hck _ hshs.1 _ (fun i hi m hm => ?_) (fun i hi o m hm => ?_) ?_ ?_
    · have hi1 : i + 1 < L.rounds x := by
        by_contra hne
        have hil : i = I.V.k (xIOf x) - 1 := by omega
        have hn := hlast s
        change L.msgAt x s (I.V.k (xIOf x) - 1) = none at hn
        rw [← hil] at hn
        simp [Claims.Recursion.VStarB.LayoutB.held, hn] at hm
      exact (hrd i (by omega)).trans ((hshs.2 i hi1).symm.trans ((hop _ hshs.1 i hi1).1 m hm).symm)
    · obtain ⟨pre, p, post, hl, hp, he⟩ := Claims.Recursion.VStarB.LayoutB.mem_piecesOf hm
      have hne : L.pieceAt x i ≠ [] := by
        rw [hl]
        simp
      have hfx : L.fixedAt x i = none := by
        rcases hfi : L.fixedAt x i with _ | m'
        · rfl
        · exact absurd ((hcl i (by change i < I.V.k (xIOf x); omega)).2 (by simp [hfi])) hne
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj he
      rw [Claims.Recursion.VStarB.LayoutB.msgs_getD L x ws hi,
        Claims.Recursion.VStarB.LayoutB.msg_of_pieces hfx hne]
      exact Claims.Recursion.VStarB.LayoutB.assembled_piece hsw hl hp
    · exact Claims.Recursion.VStarB.LayoutB.cmtAgrees_of_eq fun i hi1 _ =>
        (hrd i (by omega)).trans (hshs.2 i hi1).symm
    · exact hRows x ws htrue s ws[s] (List.getElem?_eq_getElem hsw)
  rcases hDec x st hst hf (Claims.coinSeq r) ws ho.1 htrue hchk with hd | hd
  · exact Or.inl hd
  · exact Or.inr (Claims.Recursion.VStar.coll_of_sub htrue (fun _ h => h) hd)

theorem stagedB_deployRec {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape) :
    Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope b) I L shape (showsTr untr) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  have hL : L.rounds x ≤ I.V.k (xIOf x) := ((hHeld x st hst hf).1).symm.le
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  obtain ⟨ps, hps, hfst, hdrawn⟩ := hrb x st hst hf Ω P ω r hacc
  change s < nS x at hs
  have hne := staged_ne_nil H I.V b.st x P ω r hacc hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  obtain ⟨xr, p, -, -, hd⟩ := staged_decode hne
  have hK := run_acc_length H I.V b.st x P ω r hacc
  have hm := roundsMargin_V H I.V b.st x P ω r hacc
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  have hdl : ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast =
      (Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x)) := by
    apply List.ext_getElem
    · simp only [List.length_dropLast, List.length_take, kS]
      omega
    · intro i h₁ h₂
      simp only [List.getElem_dropLast, List.getElem_take]
  have hshown : shownOf untr x
      (staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s) = some ps := by
    unfold shownOf
    rw [hd, Option.bind_some]
    show Option.bind
      (some (b.tr x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)).dropLast)) (untr x) = _
    rw [Option.bind_some, hdl]
    exact hps
  rw [hv]
  refine ⟨⟨_, hshown, hdrawn⟩, fun i hi => ?_⟩
  unfold cmtTr
  rw [hshown]
  exact hfst i (by omega)

theorem stagedB_deployRec_part {W : Type} (H : Msg → Msg) (I : Claims.Recursion.Inner Coin W) (ivScope : Msg → Bool)
    (b : Brings) (untr : TrReader) (L : Claims.Recursion.VStarB.LayoutB) (shape : Claims.Recursion.Form → Prop)
    (hrb : ReadsBackB H I.V b shape untr)
    (hHeld : Claims.Recursion.VStarB.HeldB (deployRec I.V ivScope b) I L shape) :
    Claims.Recursion.VStarB.StagedB H (deployRec I.V ivScope b) I L shape (showsPart untr b) (cmtTr untr) := by
  intro x st hst hf Ω P ω r hacc s hs
  obtain ⟨h1, h2⟩ := stagedB_deployRec H I ivScope b untr L shape hrb hHeld x st hst hf Ω P ω r hacc s hs
  refine ⟨⟨h1, ?_⟩, h2⟩
  change (Claims.run H P (V I.V b.st) x ω r).out = .acc at hacc
  change s < nS x at hs
  obtain ⟨hnum, -⟩ := run_round H I.V b.st x P ω r hacc hs rfl rfl
  have hv : (Claims.Recursion.runVstar H (deployRec I.V ivScope b) I P x ω r).getD s [] =
      staged b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x)) s := by
    show (vstar b.st x ((Claims.run H P (V I.V b.st) x ω r).msgs.take (I.V.k (xIOf x) + kS x))).getD s [] = _
    rw [vstar_getD b.st x _ hs]
    unfold vstarAt
    rw [ite_eq_left hnum]
  rw [hv]
  exact stagedPart_run H I.V b.st x P ω r hacc hs

end RecDeploy
