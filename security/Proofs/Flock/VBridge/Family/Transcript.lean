module

public import Proofs.Flock.VBridge.Family.Join
public import Proofs.Flock.VBridge.Family.Commits
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
import Mathlib.Tactic.ReduceModChar
import Proofs.Flock.Soundness.Defs

/-! `TranscriptIs` is derived after the runs, because a rep's tape is known to be one of its spine only once the rep
runs (`transcriptIs_of_runs`). -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine
open FlockSoundness.Model FlockSoundness.Audit
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

noncomputable def TableSrc.wovenAt (s : TableSrc) (cls : HClass) (P : Fin 2 → Fin cls.mPts → 𝔽) : TableFiles cls :=
  { s.filesAt cls P with
    tape := fun r => repSteps s.st s.S s.sch (extraClaims s.S cls.ptLocal cls.mPts P).length (s.proof r)
      ((s.filesAt cls P).stream r) }

theorem cut_tableSteps_head {cls : HClass} (G : TableFiles cls) :
    (cut ((tableSteps G).map kindOf) ((tableSteps G).filterMap Sum.getLeft?))[0]? =
      some [⟨Cap Dig, capDig (G.proof 0).cap⟩] := rfl

theorem streamRounds_toList_coins (coins : Array (Array F128)) (fr : ℕ → List UInt8) :
    (streamRounds coins fr).toList.map (·.coins) = coins.toList := by
  rw [← Array.toList_map, streamRounds_coins]

section sched

variable {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) {cls : HClass}

theorem length_coinSteps_le (cs : List ((k : Fin nR) → CT k)) : ∀ j, (coinSteps CT j cs).length ≤ nR - j := by
  induction cs with
  | nil => intro j; simp [coinSteps]
  | cons ρ cs ih =>
      intro j
      by_cases hj : j < nR
      · rw [coinSteps, dif_pos hj, List.singleton_append, List.length_cons]
        have := ih (j + 1)
        omega
      · rw [coinSteps_ge CT _ j (by omega)]
        simp

theorem length_coinSteps {cs : List ((k : Fin nR) → CT k)} (h : nR ≤ cs.length) : (coinSteps CT 0 cs).length = nR :=
  le_antisymm (by simpa using length_coinSteps_le CT cs 0) (le_length_coinSteps CT cs 0 nR h (by omega))

theorem ptsOf_head {ws : List (List (SMsg T))} {cs : List ((k : Fin nR) → CT k)} {P : Fin 2 → Fin cls.mPts → 𝔽}
    (hw : ∃ e, ws.head? = some [e]) (hc : (coinSteps CT 0 cs).head? = some ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩) :
    ptsOf T CT (ws.zip cs) = P := by
  obtain ⟨e, he⟩ := hw
  obtain ⟨w0, ws', rfl⟩ : ∃ w0 ws', ws = w0 :: ws' := by
    cases ws with
    | nil => simp at he
    | cons w0 ws' => exact ⟨w0, ws', rfl⟩
  obtain rfl : w0 = [e] := Option.some.inj he
  obtain ⟨c, cs', rfl⟩ : ∃ c cs', cs = c :: cs' := by
    cases cs with
    | nil => simp [coinSteps] at hc
    | cons c cs' => exact ⟨c, cs', rfl⟩
  have h0 : 0 < nR := by
    by_contra h
    rw [coinSteps_ge CT _ 0 (by omega)] at hc
    simp at hc
  have hc' : (⟨CT ⟨0, h0⟩, c ⟨0, h0⟩⟩ : Msg) = ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩ := by
    rw [coinSteps, dif_pos h0, List.singleton_append, List.head?_cons] at hc
    exact Option.some.inj hc
  apply ptsOf_eq T CT (a := Sum.inl ⟨T e.1, e.2⟩) (rest := schedSteps T CT 1 (ws'.zip cs'))
  rw [List.zip_cons_cons, schedSteps, dif_pos h0, hc']
  rfl

theorem TableSrc.transcriptIs_of_runs (s : TableSrc) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (M : Hidden cls) (hS : M.toStatement regs = s.S) (ws : List (List (SMsg T))) (cs : List ((k : Fin nR) → CT k))
    (P : Fin 2 → Fin cls.mPts → 𝔽) (hset : SetupRealizes s.st s.S s.sch cls.ptLocal cls.mPts) (henc : s.Encodes)
    (hP : ptsOf T CT (ws.zip cs) = P) (hruns : (s.files (cls := cls) T CT (ws.zip cs)).Runs)
    (hws : ws.length = nR + 1) (hcs : nR + 1 ≤ cs.length)
    (hmsg : ∀ i (h : i < ws.length), roundMsgs T ws[i] =
      (cut ((tableSteps (s.wovenAt cls P)).map kindOf) ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?))[i]?.getD
        [])
    (hcoin : coinSteps CT 0 cs = ⟨Fin 2 → Fin cls.mPts → 𝔽, P⟩ ::
      ((s.d 0).rounds.toList ++ (s.d 1).rounds.toList).map (coinArr fOf)) :
    s.TranscriptIs T CT cls (ws.zip cs) := by
  have hrG : (s.wovenAt cls P).Runs := by
    have h : (s.filesAt cls (ptsOf T CT (ws.zip cs))).Runs := hruns
    rw [hP] at h
    exact h
  obtain ⟨e, he, hext, -⟩ := s.points_ok hset P
  have hX : s.extrasAt (s.points cls.ptLocal cls.mPts P) = e := s.extrasAt_of he
  have hsz : e.size = 2 * s.S.regions.length := by
    rw [(hext e he).1, FlockSoundness.extraClaims_length]
  obtain ⟨hmS, hkS, hfold, hms, hf, -⟩ := hset
  rw [← hS] at hmS hkS hfold hext
  have hwf : ∀ x ∈ (s.wovenAt cls P).sess.streams.toList, RoundsWF x.rounds := by
    intro x hx
    have hx' : x ∈ [s.stream _ 0, s.stream _ 1] := hx
    simp only [List.mem_cons, List.not_mem_nil, or_false] at hx'
    rcases hx' with rfl | rfl <;> exact s.stream_wf _ _
  have hex : (s.wovenAt cls P).st.extra (s.wovenAt cls P).sess.points = .ok (s.wovenAt cls P).extras := by
    rw [show (s.wovenAt cls P).extras = e from hX]; exact he
  have hdec : ∀ r < 2, decodeProof (shapeOf (s.wovenAt cls P).st (s.wovenAt cls P).sch
      (s.wovenAt cls P).extras.size) ((s.wovenAt cls P).bytes r) = .ok ((s.wovenAt cls P).proof r) := by
    intro r _
    obtain ⟨b, hb⟩ := henc r
    have hb' : decodeProof (shapeOf s.st s.sch (s.extrasAt (s.points cls.ptLocal cls.mPts P)).size) b =
        .ok (s.proof r) := by
      rw [hX, hsz]; exact hb
    exact Classical.epsilon_spec (p := fun b => decodeProof (shapeOf s.st s.sch
      (s.extrasAt (s.points cls.ptLocal cls.mPts P)).size) b = .ok (s.proof r)) ⟨b, hb'⟩
  have hfind : ∀ r < 2, (s.wovenAt cls P).Finds r := by
    intro r hr
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      exact List.find?_cons_of_pos (beq_self_eq_true _)
    · show Array.find? _ #[s.stream _ 0, s.stream _ 1] = _
      rw [← Array.find?_toList]
      refine (List.find?_cons_of_neg ?_).trans (List.find?_cons_of_pos (beq_self_eq_true _))
      rw [Bool.not_eq_true]
      exact streamName_ne s.spec
  have hcap : ∀ r < 2, ((s.wovenAt cls P).proof r).cap.foldl (· ++ ·) ByteArray.empty =
      (s.wovenAt cls P).sess.rootB[(s.wovenAt cls P).i]! := by
    intro r _
    show s.cap.foldl (· ++ ·) ByteArray.empty = #[s.cap.foldl (· ++ ·) ByteArray.empty][0]!
    simp
  have h := reps_of_runs (s.wovenAt cls P) regs M hmS hkS hfold hms hf hext hwf hex hrG hfind hdec
    (fun _ _ => rfl) hcap (fun r _ => by rw [hS]; rfl)
  have hT : ∀ r < 2, RepTape (s.wovenAt cls P).st (M.toStatement regs) (s.wovenAt cls P).sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts (s.wovenAt cls P).pts).length
      ((s.wovenAt cls P).proof r) ((s.wovenAt cls P).stream r) ((s.wovenAt cls P).tape r) :=
    fun r hr => (h r hr).2.2.2.2
  have hc : coinSteps CT 0 cs = (tableSteps (s.wovenAt cls P)).filterMap Sum.getRight? := by
    rw [tableSteps_coins _ _ _ hT, hcoin]
    show _ = _ :: ((streamRounds (s.d (fin2 0)).rounds _).toList.map (·.coins) ++
      (streamRounds (s.d (fin2 1)).rounds _).toList.map (·.coins)).map (coinArr fOf)
    rw [streamRounds_toList_coins, streamRounds_toList_coins]
    rfl
  have hcut : (cut ((tableSteps (s.wovenAt cls P)).map kindOf)
      ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?)).length = ws.length := by
    rw [length_cut, ← hc, length_coinSteps CT (by omega), hws]
  have hmsg' : ws.map (roundMsgs T) = cut ((tableSteps (s.wovenAt cls P)).map kindOf)
      ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?) := by
    refine List.ext_getElem (by rw [List.length_map, hcut]) fun i h₁ h₂ => ?_
    rw [List.getElem_map, hmsg i (by simpa using h₁), List.getElem?_eq_getElem h₂, Option.getD_some]
  have hsched := schedSteps_zip T CT (tableSteps (s.wovenAt cls P)) ws cs (by omega) hmsg' hc
  refine ⟨P, ((s.wovenAt cls P).tape 0, (s.wovenAt cls P).tape 1), ?_, ?_, hsched⟩
  · have h0 := hT 0 (by omega)
    rw [hS] at h0
    exact h0
  · have h1 := hT 1 (by omega)
    rw [hS] at h1
    exact h1

end sched

def coinTy (n : ℕ) : Type := if n = 1 then 𝔽 else (Fin n → 𝔽)

noncomputable instance instFintypeCoinTy (n : ℕ) : Fintype (coinTy n) := by unfold coinTy; split <;> infer_instance

instance instNonemptyCoinTy (n : ℕ) : Nonempty (coinTy n) := by unfold coinTy; split <;> exact ⟨0⟩

def CoinTys (mPts : ℕ) (sh : Shape) {nR : ℕ} (CT : Fin nR → Type) : Prop :=
  nR = 1 + 2 * (roundSizes sh).length ∧ (∀ h : 0 < nR, CT ⟨0, h⟩ = (Fin 2 → Fin mPts → 𝔽)) ∧
    ∀ r < 2, ∀ j (hj : j < (roundSizes sh).length) (h : 1 + r * (roundSizes sh).length + j < nR),
      CT ⟨1 + r * (roundSizes sh).length + j, h⟩ = coinTy (roundSizes sh)[j]

noncomputable def vCT (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) : Type :=
  if k.val = 0 then (Fin 2 → Fin mPts → 𝔽) else coinTy ((roundSizes sh).getD ((k.val - 1) % (roundSizes sh).length) 0)

noncomputable instance instFintypeVCT (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) :
    Fintype (vCT mPts sh k) := by
  unfold vCT; split <;> infer_instance

instance instNonemptyVCT (mPts : ℕ) (sh : Shape) (k : Fin (1 + 2 * (roundSizes sh).length)) : Nonempty (vCT mPts sh k) := by
  unfold vCT; split
  · exact ⟨fun _ _ => 0⟩
  · infer_instance

theorem coinTys_vCT (mPts : ℕ) (sh : Shape) : CoinTys mPts sh (vCT mPts sh) := by
  refine ⟨rfl, fun _ => if_pos rfl, fun r hr j hj h => ?_⟩
  have hk : (1 + r * (roundSizes sh).length + j - 1) % (roundSizes sh).length = j := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · rw [show 1 + 0 * (roundSizes sh).length + j - 1 = j by omega, Nat.mod_eq_of_lt hj]
    · rw [show 1 + 1 * (roundSizes sh).length + j - 1 = j + (roundSizes sh).length by omega, Nat.add_mod_right,
        Nat.mod_eq_of_lt hj]
  show (if 1 + r * (roundSizes sh).length + j = 0 then _ else _) = _
  rw [if_neg (by omega), hk, List.getD_eq_getElem _ _ hj]

theorem exists_coinArr (n : ℕ) (y : coinTy n) :
    ∃ c : Array F128, c.size = n ∧ (⟨coinTy n, y⟩ : Msg) = coinArr fOf c := by
  by_cases hn : n = 1
  · subst hn
    have e : coinTy 1 = 𝔽 := if_pos rfl
    refine ⟨#[cast e y], rfl, ?_⟩
    have hs : (#[cast e y] : Array F128).size = 1 := rfl
    have h1 : fOf (#[cast e y] : Array F128)[0]! = cast e y := rfl
    rw [coinArr, if_pos hs, h1]
    exact Sigma.ext e (cast_heq e y).symm
  · have e : coinTy n = (Fin n → 𝔽) := if_neg hn
    obtain ⟨c, hc, hv⟩ : ∃ c : Array F128, c.size = n ∧ ∀ i (h : i < n), fOf (c)[i]! = cast e y ⟨i, h⟩ :=
      have e' : coinTy n = (Fin n → F128) := e
      ⟨Array.ofFn (n := n) fun i => cast e' y i, Array.size_ofFn, fun i h => by
        have hi : i < (Array.ofFn (n := n) fun i => cast e' y i).size := by
          rw [Array.size_ofFn]; exact h
        rw [getElem!_pos _ i hi, Array.getElem_ofFn]; rfl⟩
    subst hc
    refine ⟨c, rfl, ?_⟩
    rw [coinArr, if_neg hn]
    have hf : (fun i : Fin c.size => fOf (c)[i.val]!) = cast e y := funext fun i => hv i.val i.isLt
    exact Sigma.ext e (by rw [hf]; exact (cast_heq e y).symm)

theorem coinArr_arrOf {n : ℕ} {m : Msg} (h : ∃ c : Array F128, c.size = n ∧ m = coinArr fOf c) :
    coinArr fOf (arrOf n (some m)) = m := by
  have h' : ∃ c : Array F128, c.size = n ∧ some m = some (coinArr fOf c) :=
    let ⟨c, hc, hm⟩ := h; ⟨c, hc, by rw [hm]⟩
  unfold arrOf
  rw [dif_pos h']
  exact (Option.some.inj (Classical.choose_spec h').2).symm

section coins

variable {nR : ℕ} (CT : Fin nR → Type)

theorem getElem?_coinSteps : ∀ (cs : List ((k : Fin nR) → CT k)) (j k n : ℕ) (_ : j + k = n) (h : n < nR)
    (hk : k < cs.length), (coinSteps CT j cs)[k]? = some ⟨CT ⟨n, h⟩, cs[k] ⟨n, h⟩⟩
  | [], _, _, _, _, _, hk => absurd hk (by simp)
  | ρ :: cs, j, 0, n, hn, h, _ => by
      subst hn
      rw [coinSteps, dif_pos (show j < nR from h)]
      rfl
  | ρ :: cs, j, k + 1, n, hn, h, hk => by
      rw [coinSteps, dif_pos (show j < nR by omega), List.singleton_append, List.getElem?_cons_succ]
      exact getElem?_coinSteps cs (j + 1) k n (by omega) h (by simpa using hk)

theorem coinSteps_eq_coinsOf {mPts : ℕ} {sh : Shape} (hCT : CoinTys mPts sh CT) (cs : List ((k : Fin nR) → CT k))
    (hcs : nR ≤ cs.length) :
    coinSteps CT 0 cs = ⟨Fin 2 → Fin mPts → 𝔽, ptsC CT mPts cs⟩ ::
      ((coinsOf CT sh cs 0).toList ++ (coinsOf CT sh cs 1).toList).map (coinArr fOf) := by
  obtain ⟨hn, h0, hk⟩ := hCT
  set ns := roundSizes sh with hns
  have hlen : (coinSteps CT 0 cs).length = nR := length_coinSteps CT hcs
  have hmsg : ∀ r < 2, ∀ j (hj : j < ns.length), ∃ c : Array F128, c.size = ns[j] ∧
      (coinSteps CT 0 cs)[1 + r * ns.length + j]? = some (coinArr fOf c) := by
    intro r hr j hj
    have hlt : 1 + r * ns.length + j < nR := by
      rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl <;> omega
    rw [getElem?_coinSteps CT cs 0 _ _ (Nat.zero_add _) hlt (by omega)]
    obtain ⟨c, hc, he⟩ :=
      exists_coinArr ns[j] (cast (hk r hr j hj hlt) ((cs[1 + r * ns.length + j]'(by omega)) ⟨_, hlt⟩))
    exact ⟨c, hc, by rw [← he]; exact congrArg some (Sigma.ext (hk r hr j hj hlt) (cast_heq _ _).symm)⟩
  have hpos : 0 < nR := by omega
  have hP : (coinSteps CT 0 cs).head? =
      some ⟨Fin 2 → Fin mPts → 𝔽, cast (h0 hpos) ((cs[0]'(by omega)) ⟨0, hpos⟩)⟩ := by
    rw [List.head?_eq_getElem?, getElem?_coinSteps CT cs 0 0 0 rfl hpos (by omega)]
    exact congrArg some (Sigma.ext (h0 hpos) (cast_heq _ _).symm)
  rw [ptsC_eq hP]
  have hro : ∀ r : Fin 2, (coinsOf CT sh cs r).toList.map (coinArr fOf) =
      (List.range ns.length).map fun j => ((coinSteps CT 0 cs)[1 + r.val * ns.length + j]?).getD ⟨Unit, ()⟩ := by
    intro r
    refine List.ext_getElem (by simp [coinsOf, hns]) fun j h₁ h₂ => ?_
    have hj : j < ns.length := by simpa using h₂
    obtain ⟨c, hc, he⟩ := hmsg r.val r.isLt j hj
    have e1 : (coinsOf CT sh cs r).toList[j]'(by simpa using h₁) =
        arrOf ns[j] ((coinSteps CT 0 cs)[1 + r.val * ns.length + j]?) := by
      simp [coinsOf, hns]
    simp only [List.getElem_map, List.getElem_range]
    rw [e1, he, Option.getD_some]
    exact coinArr_arrOf ⟨c, hc, rfl⟩
  rw [List.map_append, hro 0, hro 1]
  refine List.ext_getElem? fun i => ?_
  rcases i with _ | i
  · rw [List.getElem?_cons_zero, ← List.head?_eq_getElem?, hP]
  rw [List.getElem?_cons_succ]
  by_cases hi : i < ns.length
  · rw [List.getElem?_append_left (by simpa using hi)]
    simp only [List.getElem?_map, List.getElem?_range hi, Option.map_some]
    obtain ⟨c, -, he⟩ := hmsg 0 (by omega) i hi
    have e0 : 1 + (0 : Fin 2).val * ns.length + i = i + 1 := by
      have : (0 : Fin 2).val = 0 := rfl
      omega
    have e0' : 1 + 0 * ns.length + i = i + 1 := by omega
    rw [e0]
    rw [e0'] at he
    rw [he, Option.getD_some]
  · by_cases hi' : i < 2 * ns.length
    · rw [List.getElem?_append_right (by simpa using hi)]
      simp only [List.length_map, List.length_range, List.getElem?_map, List.getElem?_range (by omega : i - ns.length <
        ns.length), Option.map_some]
      obtain ⟨c, -, he⟩ := hmsg 1 (by omega) (i - ns.length) (by omega)
      have e1 : 1 + (1 : Fin 2).val * ns.length + (i - ns.length) = i + 1 := by
        have : (1 : Fin 2).val = 1 := rfl
        omega
      have e1' : 1 + 1 * ns.length + (i - ns.length) = i + 1 := by omega
      rw [e1]
      rw [e1'] at he
      rw [he, Option.getD_some]
    · rw [List.getElem?_eq_none (l := coinSteps CT 0 cs) (by rw [hlen]; omega)]
      symm
      apply List.getElem?_eq_none
      simp
      omega

end coins

section msgs

variable {nT : ℕ} (T : Fin nT → Type)

open Classical in
noncomputable def toS (m : Msg) : List (SMsg T) :=
  if h : ∃ j, T j = m.1 then [⟨Classical.choose h, cast (Classical.choose_spec h).symm m.2⟩] else []

noncomputable def toSs (g : List Msg) : List (SMsg T) := g.flatMap (toS T)

theorem roundMsgs_toSs : ∀ (g : List Msg), (∀ m ∈ g, ∃ j, T j = m.1) → roundMsgs T (toSs T g) = g
  | [], _ => rfl
  | m :: g, h => by
      have hm := h m (List.mem_cons_self ..)
      have ih := roundMsgs_toSs g fun m' hm' => h m' (List.mem_cons_of_mem _ hm')
      have e : toSs T (m :: g) = toS T m ++ toSs T g := rfl
      rw [e, toS, dif_pos hm]
      simp only [roundMsgs, List.map_cons, List.singleton_append] at ih ⊢
      rw [ih]
      exact congrArg (· :: g) (Sigma.ext (Classical.choose_spec hm) (cast_heq _ _))

theorem roundMsgs_getElem {ws : List (List (SMsg T))} {i : ℕ} (hi : i < ws.length) {g : List Msg}
    (e : (ws[i]?).map (roundMsgs T) = some g) : roundMsgs T ws[i] = g := by
  rw [List.getElem?_eq_getElem hi, Option.map_some] at e
  exact Option.some.inj e

theorem eq_singleton_of_roundMsgs {w : List (SMsg T)} {m : Msg} (h : roundMsgs T w = [m]) : ∃ e, w = [e] := by
  match w, h with
  | [], h => simp [roundMsgs] at h
  | [e], _ => exact ⟨e, rfl⟩
  | _ :: _ :: _, h => simp [roundMsgs] at h

end msgs

def tableTys (ext : ℕ) : List Type :=
  [Cap Dig, Bool, (Fin 64 → 𝔽) × (Fin 64 → 𝔽), 𝔽 × 𝔽, Fin 64 → 𝔽, Fin 128 → 𝔽, 𝔽, 𝕂 × 𝕂, Fin (2 ^ (ext + 1)) → 𝔽,
    Opens 𝔽 Dig × Salts (List UInt8)]

def AllTys (ext : ℕ) (L : List Msg) : Prop := ∀ m ∈ L, m.1 ∈ tableTys ext

theorem AllTys.nil {ext : ℕ} : AllTys ext [] := fun _ h => by simp at h

theorem AllTys.cons {ext : ℕ} {m : Msg} {L : List Msg} (h : m.1 ∈ tableTys ext) (hL : AllTys ext L) :
    AllTys ext (m :: L) := by
  intro x hx
  rcases List.mem_cons.1 hx with rfl | hx
  exacts [h, hL x hx]

theorem AllTys.append {ext : ℕ} {L L' : List Msg} (h : AllTys ext L) (h' : AllTys ext L') : AllTys ext (L ++ L') := by
  intro x hx
  rcases List.mem_append.1 hx with hx | hx
  exacts [h x hx, h' x hx]

theorem AllTys.map {ext : ℕ} {α : Type} {f : α → Msg} {l : List α} (h : ∀ a, (f a).1 ∈ tableTys ext) :
    AllTys ext (l.map f) := by
  intro x hx
  obtain ⟨a, -, rfl⟩ := List.mem_map.1 hx
  exact h a

theorem tyAt (ext n : ℕ) (h : n < (tableTys ext).length) : (tableTys ext)[n] ∈ tableTys ext := List.getElem_mem h

theorem ty_cap (ext : ℕ) : Cap Dig ∈ tableTys ext := tyAt ext 0 (by simp [tableTys])
theorem ty_bool (ext : ℕ) : Bool ∈ tableTys ext := tyAt ext 1 (by simp [tableTys])
theorem ty_vv (ext : ℕ) : ((Fin 64 → 𝔽) × (Fin 64 → 𝔽)) ∈ tableTys ext := tyAt ext 2 (by simp [tableTys])
theorem ty_ff (ext : ℕ) : (𝔽 × 𝔽) ∈ tableTys ext := tyAt ext 3 (by simp [tableTys])
theorem ty_v64 (ext : ℕ) : (Fin 64 → 𝔽) ∈ tableTys ext := tyAt ext 4 (by simp [tableTys])
theorem ty_v128 (ext : ℕ) : (Fin 128 → 𝔽) ∈ tableTys ext := tyAt ext 5 (by simp [tableTys])
theorem ty_f (ext : ℕ) : 𝔽 ∈ tableTys ext := tyAt ext 6 (by simp [tableTys])
theorem ty_kk (ext : ℕ) : (𝕂 × 𝕂) ∈ tableTys ext := tyAt ext 7 (by simp [tableTys])
theorem ty_yr (ext : ℕ) : (Fin (2 ^ (ext + 1)) → 𝔽) ∈ tableTys ext := tyAt ext 8 (by simp [tableTys])
theorem ty_opens (ext : ℕ) : (Opens 𝔽 Dig × Salts (List UInt8)) ∈ tableTys ext := tyAt ext 9 (by simp [tableTys])

theorem fMsg_ty (ext : ℕ) (a : F128) : (fMsg a).1 ∈ tableTys ext := ty_f ext
theorem kMsg_ty (ext : ℕ) (m : F256 × F256) : (kMsg m).1 ∈ tableTys ext := ty_kk ext
theorem kMsgs_tys (ext : ℕ) (p : LigeritoProof) (start k : ℕ) : AllTys ext (kMsgs p start k) :=
  AllTys.map fun _ => kMsg_ty ext _

/- Reducible transparency throughout: unifying two of the table's types at default transparency unfolds `GF128` and
`GF256`, which does not terminate in practice. -/

theorem recMsgs_tys (p : LigeritoProof) (ext : ℕ) :
    ∀ left i msg ood, AllTys ext (recMsgs p capDig ext left i msg ood)
  | 0, _, _, _ => by rw [recMsgs]; exact AllTys.nil
  | 1, _, _, _ => by
    rw [recMsgs]
    repeat' (first
      | with_reducible exact AllTys.nil
      | with_reducible refine AllTys.append ?_ ?_
      | with_reducible refine AllTys.cons ?_ ?_
      | with_reducible refine AllTys.map fun _ => ?_
      | with_reducible exact kMsgs_tys _ _ _ _ | with_reducible exact fMsg_ty _ _ | with_reducible exact kMsg_ty _ _
      | with_reducible exact ty_cap _ | with_reducible exact ty_vv _ | with_reducible exact ty_ff _
      | with_reducible exact ty_v64 _ | with_reducible exact ty_v128 _ | with_reducible exact ty_yr _)
  | left + 2, i, msg, ood => by
      rw [recMsgs]
      repeat' (first
        | with_reducible exact AllTys.nil
        | with_reducible refine AllTys.append ?_ ?_
        | with_reducible refine AllTys.cons ?_ ?_
        | with_reducible refine AllTys.map fun _ => ?_
        | with_reducible exact kMsgs_tys _ _ _ _ | with_reducible exact fMsg_ty _ _ | with_reducible exact kMsg_ty _ _
        | with_reducible exact ty_cap _ | with_reducible exact ty_vv _ | with_reducible exact ty_ff _
        | with_reducible exact ty_v64 _ | with_reducible exact ty_v128 _ | with_reducible exact ty_yr _)
      exact recMsgs_tys p ext (left + 1) (i + 1) (msg + 7) (ood + 2)

theorem repMsgs_tys (st : Setup) (sch : Flock.Schedule) (n : ℕ) (p : Proof) :
    AllTys sch.ext (repMsgs st sch n p capDig) := by
  unfold repMsgs ligMsgs zcMsgs lcMsgs opMsgs
  repeat' (first
    | with_reducible exact AllTys.nil
    | with_reducible refine AllTys.append ?_ ?_
    | with_reducible refine AllTys.cons ?_ ?_
    | with_reducible refine AllTys.map fun _ => ?_
    | with_reducible exact kMsgs_tys _ _ _ _ | with_reducible exact fMsg_ty _ _ | with_reducible exact kMsg_ty _ _
    | with_reducible exact ty_cap _ | with_reducible exact ty_vv _ | with_reducible exact ty_ff _
    | with_reducible exact ty_v64 _ | with_reducible exact ty_v128 _ | with_reducible exact ty_yr _)
  exact recMsgs_tys _ _ _ _ _ _

theorem mem_cut : ∀ (σ : List Exec.Kind) (ms : List Msg) {g : List Msg}, g ∈ cut σ ms → ∀ m ∈ g, m ∈ ms
  | [], _, g, hg, m, hm => by
      rw [cut, List.mem_singleton] at hg
      subst hg
      simp at hm
  | .msg _ :: σ, m' :: ms, g, hg, m, hm => by
      have ih : ∀ {g : List Msg}, g ∈ cut σ ms → ∀ m ∈ g, m ∈ ms := mem_cut σ ms
      rw [cut] at hg
      cases hc : cut σ ms with
      | nil =>
          rw [hc, consHead, List.mem_singleton] at hg
          subst hg
          rw [List.mem_singleton] at hm
          subst hm
          exact List.mem_cons_self ..
      | cons g0 gs =>
          rw [hc, consHead, List.mem_cons] at hg
          rcases hg with rfl | hg
          · rcases List.mem_cons.1 hm with rfl | hm
            · exact List.mem_cons_self ..
            · exact List.mem_cons_of_mem _ (ih (by rw [hc]; exact List.mem_cons_self ..) m hm)
          · exact List.mem_cons_of_mem _ (ih (by rw [hc]; exact List.mem_cons_of_mem _ hg) m hm)
  | .msg _ :: _, [], g, hg, m, hm => by
      rw [cut, List.mem_singleton] at hg
      subst hg
      simp at hm
  | .coin _ :: σ, ms, g, hg, m, hm => by
      rw [cut, List.mem_cons] at hg
      rcases hg with rfl | hg
      · simp at hm
      · exact mem_cut σ ms hg m hm

theorem mem_weave : ∀ (σ : List Exec.Kind) (ms cs : List Msg) {m : Msg}, Sum.inl m ∈ weave σ ms cs → m ∈ ms
  | .msg _ :: σ, m' :: ms, cs, m, h => by
      rw [weave, List.mem_cons] at h
      rcases h with h | h
      · cases h
        exact List.mem_cons_self ..
      · exact List.mem_cons_of_mem _ (mem_weave σ ms cs h)
  | .coin _ :: σ, ms, c :: cs, m, h => by
      rw [weave, List.mem_cons] at h
      rcases h with h | h
      · cases h
      · exact mem_weave σ ms cs h
  | [], _, _, _, h => by simp [weave] at h
  | .msg _ :: _, [], _, _, h => by simp [weave] at h
  | .coin _ :: _, _, [], _, h => by simp [weave] at h

theorem mem_firstSteps {ts : List Step} {m : Msg} (h : Sum.inl m ∈ firstSteps ts) :
    m = ⟨Bool, true⟩ ∨ Sum.inl m ∈ ts := by
  obtain ⟨s, hs, h⟩ := List.mem_flatMap.1 h
  rcases List.mem_cons.1 h with h | h
  · cases h
    exact Or.inl rfl
  · rw [List.mem_singleton] at h
    subst h
    exact Or.inr hs

theorem TableSrc.wovenAt_tys (s : TableSrc) (cls : HClass) (P : Fin 2 → Fin cls.mPts → 𝔽) :
    ∀ g ∈ cut ((tableSteps (s.wovenAt cls P)).map kindOf) ((tableSteps (s.wovenAt cls P)).filterMap Sum.getLeft?),
      AllTys s.sch.ext g := by
  intro g hg m hm
  obtain ⟨x, hx, he⟩ := List.mem_filterMap.1 (mem_cut _ _ hg m hm)
  obtain ⟨rfl⟩ : x = Sum.inl m := by
    cases x with
    | inl y => cases he; rfl
    | inr y => cases he
  have hrep : ∀ r, Sum.inl m ∈ (s.wovenAt cls P).tape r → m.1 ∈ tableTys s.sch.ext := fun r h =>
    repMsgs_tys s.st s.sch _ (s.proof r) m (mem_weave _ _ _ h)
  rcases List.mem_cons.1 hx with h | hx
  · cases h
    exact ty_cap _
  rcases List.mem_cons.1 hx with h | hx
  · cases h
  rcases List.mem_append.1 hx with hx | hx
  · rcases List.mem_append.1 hx with hx | hx
    · rcases mem_firstSteps hx with rfl | hx
      · exact ty_bool _
      · exact hrep 0 hx
    · exact hrep 1 hx
  · rw [List.mem_singleton] at hx
    cases hx
    exact ty_opens _

/-- A fact of the instantiation's `T`, as `CoinTys` is of its `CT`. -/
def TysIn {nT : ℕ} (T : Fin nT → Type) (ext : ℕ) : Prop := ∀ ty ∈ tableTys ext, ∃ j, T j = ty

def vT (ext : ℕ) (j : Fin (tableTys ext).length) : Type := (tableTys ext)[j]

theorem tysIn_vT (ext : ℕ) : TysIn (vT ext) ext := fun _ h =>
  let ⟨j, hj, e⟩ := List.getElem_of_mem h
  ⟨⟨j, hj⟩, e⟩

section family

variable {nT nR : ℕ} {T : Fin nT → Type} {CT : Fin nR → Type} [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)]
variable {cls : HClass} {regs : List (Region cls.kLog (cls.m - cls.kLog))} {sch : Flock.Schedule}
variable {n : ℕ} {L : SamplingDesign n} {Reg Cm Reg' Reg₂ : Type}

local notation "ℐ" => flockInnerCL T CT execArith H512 enc512 InnerAccepts.leafPlain cls regs (schedOf sch)

namespace VStarFamily

variable (F : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂) (dg : ByteArray → ByteArray) (N k Rw : ℕ)
variable (spec : SessionSpec) (st : Reg → L.Ω → Setup)
    (cap : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Array ByteArray)
    (lig : ((j : Fin F.stmts.size) → (F.Vs j).Bits) → Fin 2 → LigeritoProof)

noncomputable def woven (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin)) (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) :
    TableFiles cls :=
  (F.tableSrc spec st cap lig R ω t l).wovenAt cls (ptsC CT cls.mPts (t.map Prod.snd))

noncomputable def tableMsgs (R : Reg) (ω : L.Ω) (t : List (Cm × (ℐ).Coin))
    (l : (j : Fin F.stmts.size) → (F.Vs j).Bits) : List (List Msg) :=
  cut ((tableSteps (F.woven spec st cap lig R ω t l)).map kindOf)
    ((tableSteps (F.woven spec st cap lig R ω t l)).filterMap Sum.getLeft?)

/-- False as `VBridge` is quantified, at the empty history: no inner message is sent there, and V*'s sessions can
accept. `VBridge` stated at whole histories, as `recGame` reads it, or sessions that refuse a short history
discharge it. -/
def HLeaf : Prop := ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o → (ℐ).r ≤ t.length

def HFilesR : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Runs →
    (F.files R ω t (layers F.zs dg N k Rw R ω t R₂ τ)).Valid regs (F.x R ω) T CT
      ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd))

def MsgsAre : Prop := F.AtAccepted dg N k Rw fun R ω t l =>
  (F.wmsg l).length = (ℐ).r ∧
    ∀ i < (ℐ).r, ((F.wmsg l)[i]?).map (roundMsgs T) = some ((F.tableMsgs spec st cap lig R ω t l)[i]?.getD [])

theorem ptsOf_at (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw)
    (hmsgs : F.MsgsAre dg N k Rw spec st cap lig)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) :
    ptsOf (cls := cls) T CT ((F.wmsg (layers F.zs dg N k Rw R ω t R₂ τ)).zip (t.map Prod.snd)) =
      ptsC CT cls.mPts (t.map Prod.snd) := by
  have hr : nR + 1 ≤ t.length := hleaf R ω t R₂ τ o h
  obtain ⟨-, hm⟩ := hmsgs R ω t R₂ τ o h
  have h0 := hm 0 (Nat.succ_pos nR)
  rw [show (F.tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[0]? = some _ from
    cut_tableSteps_head _, Option.getD_some] at h0
  obtain ⟨w, hw, hw'⟩ := Option.map_eq_some_iff.1 h0
  obtain ⟨e, rfl⟩ := eq_singleton_of_roundMsgs T hw'
  have hr' : nR ≤ (t.map Prod.snd).length := by rw [List.length_map]; omega
  refine ptsOf_head T CT ⟨e, List.head?_eq_getElem?.trans hw⟩ ?_
  exact congrArg List.head? (coinSteps_eq_coinsOf CT hCT _ hr')

theorem hFilesR_of (hF : F.files = F.filesOf spec st cap lig) (hrep : F.rep₀ = F.repOf (F.coinFiles spec st cap lig))
    (hset : ∀ R ω, SetupRealizes (st R ω) ((F.x R ω).toStatement regs) sch cls.ptLocal cls.mPts)
    (henc : F.AtAccepted dg N k Rw fun R ω t l => (F.tableSrc spec st cap lig R ω t l).Encodes)
    (hCT : CoinTys cls.mPts F.sh CT) (hleaf : F.HLeaf dg N k Rw) (hmsgs : F.MsgsAre dg N k Rw spec st cap lig) :
    F.HFilesR dg N k Rw := by
  intro R ω t R₂ τ o h hruns
  rw [hF] at hruns ⊢
  have hr : nR + 1 ≤ t.length := hleaf R ω t R₂ τ o h
  have hr' : nR + 1 ≤ (t.map Prod.snd).length := by rw [List.length_map]; exact hr
  obtain ⟨hlen, hm⟩ := hmsgs R ω t R₂ τ o h
  refine TableSrc.valid T CT _ regs (F.x R ω) _ rfl (hset R ω) (henc R ω t R₂ τ o h) ?_
  refine (F.tableSrc spec st cap lig R ω t _).transcriptIs_of_runs T CT regs (F.x R ω) rfl _ (t.map Prod.snd)
    (ptsC CT cls.mPts (t.map Prod.snd)) (hset R ω) (henc R ω t R₂ τ o h) (F.ptsOf_at dg N k Rw spec st cap lig hCT hleaf
      hmsgs h) hruns hlen hr' (fun i hi => ?_) ?_
  · exact roundMsgs_getElem T hi (hm i (by rw [← hlen]; exact hi))
  · show coinSteps CT 0 _ = _ :: ((F.rep R ω t _ 0).rounds.toList ++ (F.rep R ω t _ 1).rounds.toList).map _
    rw [F.rep_rounds spec st cap lig hrep, F.rep_rounds spec st cap lig hrep]
    exact coinSteps_eq_coinsOf CT hCT _ (Nat.le_of_succ_le hr')

noncomputable def coinFamily : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
  let F₁ : VStarFamily T CT cls regs sch L Reg Cm Reg' Reg₂ :=
    { F with rep₀ := F.repOf (F.coinFiles spec st cap lig), rows := F.rowsOf, acc := F.accOf }
  { F₁ with alpha := F₁.alphaOf }

variable (own : Fin F.stmts.size → Array Flock.Registered.Port) (sel : ℕ → String → ℕ → Prop)

/-- Discharged from the stage files (`rec_vstage`'s staging): round `i`'s table messages read only stage `i`'s masked
commit-string wires. -/
def MsgsRead : Prop :=
  ∀ i R ω t R₂ τ o R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' →
    maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) =
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') →
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ))[i]? =
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t' (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?

def MsgTys : Prop :=
  ∀ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o →
    ∀ g ∈ (F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t (layers F.zs dg N k Rw R ω t R₂ τ),
      ∀ m ∈ g, ∃ j, T j = m.1

theorem msgTys_of (hT : TysIn T sch.ext) : F.MsgTys dg N k Rw spec st cap lig :=
  fun _ _ _ _ _ _ _ g hg m hm => hT _ (TableSrc.wovenAt_tys _ cls _ g hg m hm)

open Classical in
noncomputable def decOf (i : ℕ) (Xs : (j : Fin F.stmts.size) → (F.Vs j).Bits) : (ℐ).Msg :=
  if h : ∃ g : List Msg, ∃ R ω t R₂ τ o, Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o ∧
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) = Xs ∧
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
        (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] = g
  then toSs T (Classical.choose h) else []

theorem decOf_spec (hread : F.MsgsRead dg N k Rw spec st cap lig own sel) (htys : F.MsgTys dg N k Rw spec st cap lig)
    {R : Reg} {ω : L.Ω} {t : List (Cm × (ℐ).Coin)} {R₂ : Reg₂} {τ o}
    (h : Accepted (I := ℐ) F.zs dg N k Rw R ω t R₂ τ o) (i : ℕ) :
    roundMsgs T (F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ))) =
    ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
  have hex : ∃ g : List Msg, ∃ R' ω' t' R₂' τ' o', Accepted (I := ℐ) F.zs dg N k Rw R' ω' t' R₂' τ' o' ∧
      maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R' ω' t' R₂' τ') =
        maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ) ∧
      ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R' ω' t'
        (layers F.zs dg N k Rw R' ω' t' R₂' τ'))[i]?.getD [] = g := ⟨_, R, ω, t, R₂, τ, o, h, rfl, rfl⟩
  have hd : F.decOf dg N k Rw spec st cap lig own sel i
      (maskOf F.Vs (skOf F.Vs own sel) i (layers F.zs dg N k Rw R ω t R₂ τ)) = toSs T (Classical.choose hex) :=
    dif_pos hex
  rw [hd]
  obtain ⟨R', ω', t', R₂', τ', o', h', hX', hg'⟩ := Classical.choose_spec hex
  have e := hread i R ω t R₂ τ o R' ω' t' R₂' τ' o' h h' hX'.symm
  have hc : Classical.choose hex = ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]?.getD [] := by
    rw [← hg', ← e]
  rw [hc]
  apply roundMsgs_toSs
  intro m hm
  cases hti : ((F.coinFamily spec st cap lig).tableMsgs spec st cap lig R ω t
      (layers F.zs dg N k Rw R ω t R₂ τ))[i]? with
  | none => rw [hti] at hm; simp at hm
  | some g =>
      rw [hti, Option.getD_some] at hm
      exact htys R ω t R₂ τ o h g (List.mem_of_getElem? hti) m hm

end VStarFamily

end family

end FlockVBridge.Family
