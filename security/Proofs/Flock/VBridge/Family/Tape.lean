module

public import Proofs.Flock.VBridge.Stages
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
import Proofs.Flock.Soundness.Discharge.Exec.Tables
import Proofs.Flock.Soundness.Discharge.PlaysCoin.Coins

/-! `TableFiles.Valid`'s transcript conjuncts (`transcript_of_runs`). -/

@[expose] public section

namespace FlockVBridge.Family

open FlockSoundness FlockSoundness.Game FlockSoundness.Discharge FlockSoundness.Discharge.Exec
open FlockSoundness.Discharge.Recursion FlockSoundness.Discharge.PlaysCoin FlockSoundness.Refine FlockSoundness.Model
open Flock FlockLevel3 FlockVBridge.Compiled FlockVBridge.InnerAccepts FlockVBridge.Stages

local notation "𝔽" => FlockLevel3.GF128
local notation "𝕂" => FlockLevel3.GF256

theorem weave_of_prefix {ts : List Step} {σ : List Exec.Kind} (h : ts.map kindOf <+: σ) :
    weave σ (ts.filterMap Sum.getLeft?) (ts.filterMap Sum.getRight?) = ts := by
  induction ts generalizing σ with
  | nil =>
      cases σ with
      | nil => rfl
      | cons k σ => cases k <;> rfl
  | cons s ts ih =>
      obtain ⟨σ', rfl⟩ : ∃ σ', σ = kindOf s :: σ' := by
        obtain ⟨u, hu⟩ := h
        exact ⟨ts.map kindOf ++ u, by rw [← hu]; rfl⟩
      have ih' := ih (List.cons_prefix_cons.mp h).2
      cases s with
      | inl m => exact congrArg (Sum.inl m :: ·) ih'
      | inr c => exact congrArg (Sum.inr c :: ·) ih'

def repSteps (st : Setup) (S : Model.Statement) (sch : Flock.Schedule) (n : ℕ) (p : Proof) (s : StreamRec) :
    List Step :=
  weave (repShape 𝔽 𝕂 Dig S.m S.kLog n (schedOf sch)) (repMsgs st sch (2 + n) p capDig)
    ((s.rounds.toList.map (·.coins)).map (coinArr fOf))

theorem repTape_of_plays (r : ℕ) (st : Setup) (S : Model.Statement) (sch : Flock.Schedule) (hm : 13 ≤ S.m)
    (extra : List (Claim 𝔽 S.m)) (p : Proof) (s : StreamRec) {pend : PendL 𝔽 Dig}
    (h : Plays fOf (repC execArith r S (schedOf sch) hm extra) (repMsgs st sch (2 + extra.length) p capDig)
      (s.rounds.toList.map (·.coins)) pend) :
    RepTape st S sch extra.length p s (repSteps st S sch extra.length p s) := by
  obtain ⟨ts, hp, h1, h2⟩ := toT_exact fOf h
  have hk := Spine.prefix hp (repC_spine execArith r S (schedOf sch) hm extra)
  have e : repSteps st S sch extra.length p s = ts := by
    unfold repSteps
    rw [← h1, ← h2]
    exact weave_of_prefix hk
  rw [e]
  exact ⟨hk, h1, h2⟩

section sched

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

def coinSteps : ℕ → List ((k : Fin r) → CT k) → List Msg
  | _, [] => []
  | j, ρ :: cs => (if h : j < r then [⟨CT ⟨j, h⟩, ρ ⟨j, h⟩⟩] else []) ++ coinSteps (j + 1) cs

theorem coinSteps_ge (cs : List ((k : Fin r) → CT k)) : ∀ j, r ≤ j → coinSteps CT j cs = [] := by
  induction cs with
  | nil => intro j _; rfl
  | cons ρ cs ih =>
      intro j h
      rw [coinSteps, dite_eq_right_of_eq_false (eq_false (show ¬ j < r by omega)), List.nil_append]
      exact ih (j + 1) (by omega)

def roundMsgs (e : List (SMsg T)) : List Msg := e.map fun m => ⟨T m.1, m.2⟩

theorem consHead_ne_nil (m : Msg) (l : List (List Msg)) : consHead m l ≠ [] := by
  cases l <;> simp [consHead]

theorem cut_ne_nil (σ : List Exec.Kind) (ms : List Msg) : cut σ ms ≠ [] := by
  match σ, ms with
  | [], _ => simp [cut]
  | .msg _ :: σ, m :: ms => exact consHead_ne_nil m _
  | .msg _ :: _, [] => simp [cut]
  | .coin _ :: σ, ms => simp [cut]

theorem schedSteps_of_cut (ts : List Step) : ∀ (j : ℕ) (t : List (List (SMsg T) × ((k : Fin r) → CT k))),
    t.map (fun e => roundMsgs T e.1) = cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) →
    coinSteps CT j (t.map Prod.snd) = ts.filterMap Sum.getRight? → schedSteps T CT j t = ts := by
  induction ts with
  | nil =>
      intro j t hm hc
      rcases t with _ | ⟨⟨es, ρ⟩, t⟩
      · simp [cut] at hm
      · have hm' : roundMsgs T es :: t.map (fun e => roundMsgs T e.1) = [[]] := hm
        simp only [List.cons.injEq, roundMsgs, List.map_eq_nil_iff] at hm'
        obtain ⟨rfl, rfl⟩ := hm'
        by_cases hj : j < r
        · have hc' : coinSteps CT j [ρ] = [] := hc
          simp [coinSteps, hj] at hc'
        · simp [schedSteps, hj]
  | cons s ts ih =>
      intro j t hm hc
      cases s with
      | inl m =>
          have hm' : t.map (fun e => roundMsgs T e.1) =
              consHead m (cut (ts.map kindOf) (ts.filterMap Sum.getLeft?)) := hm
          have hc' : coinSteps CT j (t.map Prod.snd) = ts.filterMap Sum.getRight? := hc
          obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (cut_ne_nil (ts.map kindOf) (ts.filterMap Sum.getLeft?))
          rw [hg] at hm'
          rcases t with _ | ⟨⟨es, ρ⟩, t⟩
          · simp [consHead] at hm'
          rcases es with _ | ⟨e₀, es⟩
          · simp [consHead, roundMsgs] at hm'
          have hm'' : ((⟨T e₀.1, e₀.2⟩ : Msg) :: roundMsgs T es) :: t.map (fun e => roundMsgs T e.1) =
              (m :: g) :: gs := hm'
          simp only [List.cons.injEq] at hm''
          obtain ⟨⟨rfl, hes⟩, ht⟩ := hm''
          have ih' := ih j ((es, ρ) :: t) (by
            show roundMsgs T es :: t.map (fun e => roundMsgs T e.1) = _
            rw [hes, ht, hg]) hc'
          rw [← ih']
          rfl
      | inr c =>
          have hm' : t.map (fun e => roundMsgs T e.1) = [] :: cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) := hm
          have hc' : coinSteps CT j (t.map Prod.snd) = c :: ts.filterMap Sum.getRight? := hc
          rcases t with _ | ⟨⟨es, ρ⟩, t⟩
          · simp at hm'
          have hm'' : roundMsgs T es :: t.map (fun e => roundMsgs T e.1) =
              [] :: cut (ts.map kindOf) (ts.filterMap Sum.getLeft?) := hm'
          simp only [List.cons.injEq, roundMsgs, List.map_eq_nil_iff] at hm''
          obtain ⟨rfl, ht⟩ := hm''
          have hc'' : coinSteps CT j (ρ :: t.map Prod.snd) = c :: ts.filterMap Sum.getRight? := hc'
          by_cases hj : j < r
          · rw [coinSteps, dite_eq_left_of_eq_true (eq_true hj), List.singleton_append, List.cons.injEq] at hc''
            have ih' := ih (j + 1) t ht hc''.2
            rw [← ih', ← hc''.1, schedSteps, dite_eq_left_of_eq_true (eq_true hj)]
            rfl
          · rw [coinSteps_ge CT _ j (by omega)] at hc''
            cases hc''

theorem length_cut (ts : List Step) :
    (cut (ts.map kindOf) (ts.filterMap Sum.getLeft?)).length = (ts.filterMap Sum.getRight?).length + 1 := by
  induction ts with
  | nil => rfl
  | cons s ts ih =>
      cases s with
      | inl m =>
          show (consHead m (cut (ts.map kindOf) (ts.filterMap Sum.getLeft?))).length = _
          obtain ⟨g, gs, hg⟩ := List.exists_cons_of_ne_nil (cut_ne_nil (ts.map kindOf) (ts.filterMap Sum.getLeft?))
          rw [hg] at ih ⊢
          exact ih
      | inr c =>
          show ([] :: cut (ts.map kindOf) (ts.filterMap Sum.getLeft?)).length = (c :: _).length + 1
          rw [List.length_cons, ih, List.length_cons]

theorem le_length_coinSteps (cs : List ((k : Fin r) → CT k)) :
    ∀ j n, n ≤ cs.length → j + n ≤ r → n ≤ (coinSteps CT j cs).length := by
  induction cs with
  | nil => intro j n h _; simp at h; omega
  | cons ρ cs ih =>
      intro j n h hr
      rcases n with _ | n
      · exact Nat.zero_le _
      rw [coinSteps, dite_eq_left_of_eq_true (eq_true (show j < r by omega)), List.singleton_append,
        List.length_cons]
      have := ih (j + 1) n (by simp at h; omega) (by omega)
      omega

theorem coinSteps_take (cs : List ((k : Fin r) → CT k)) :
    ∀ j n, r ≤ j + n → coinSteps CT j (cs.take n) = coinSteps CT j cs := by
  induction cs with
  | nil => intro j n _; simp
  | cons ρ cs ih =>
      intro j n h
      rcases n with _ | n
      · rw [List.take_zero, coinSteps_ge CT _ j (by omega), coinSteps_ge CT _ j (by omega)]
      · rw [List.take_succ_cons, coinSteps, coinSteps, ih (j + 1) n (by omega)]

theorem map_fst_zip {α β : Type _} : ∀ (a : List α) (b : List β), a.length ≤ b.length → (a.zip b).map Prod.fst = a
  | [], _, _ => rfl
  | _ :: _, [], h => by simp at h
  | x :: a, y :: b, h => by
      rw [List.zip_cons_cons, List.map_cons, map_fst_zip a b (by simpa using h)]

theorem map_snd_zip {α β : Type _} : ∀ (a : List α) (b : List β), (a.zip b).map Prod.snd = b.take a.length
  | [], _ => rfl
  | _ :: _, [] => rfl
  | x :: a, y :: b => by
      rw [List.zip_cons_cons, List.map_cons, map_snd_zip a b, List.length_cons, List.take_succ_cons]

theorem schedSteps_zip (ts : List Step) (ws : List (List (SMsg T))) (cs : List ((k : Fin r) → CT k))
    (hlen : ws.length ≤ cs.length) (hm : ws.map (roundMsgs T) = cut (ts.map kindOf) (ts.filterMap Sum.getLeft?))
    (hc : coinSteps CT 0 cs = ts.filterMap Sum.getRight?) : schedSteps T CT 0 (ws.zip cs) = ts := by
  have hN : ws.length = (ts.filterMap Sum.getRight?).length + 1 := by
    rw [← length_cut ts, ← hm, List.length_map]
  have hr : r ≤ 0 + ws.length := by
    by_contra hlt
    have := le_length_coinSteps CT cs 0 ws.length hlen (by omega)
    rw [hc] at this
    omega
  refine schedSteps_of_cut T CT ts 0 (ws.zip cs) ?_ ?_
  · calc (ws.zip cs).map (fun e => roundMsgs T e.1) = ((ws.zip cs).map Prod.fst).map (roundMsgs T) :=
          (List.map_map ..).symm
      _ = ws.map (roundMsgs T) := by rw [map_fst_zip ws cs hlen]
      _ = _ := hm
  · rw [map_snd_zip, coinSteps_take CT cs 0 _ hr, hc]

end sched

variable {cls : HClass}

def tableSteps (G : TableFiles cls) : List Step :=
  .inl ⟨Cap Dig, capDig (G.proof 0).cap⟩ :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
    tableTape (G.tape 0) (G.tape 1) (opensOf (G.proof 0) (G.proof 1) G.sch, saltsOf (G.proof 0) (G.proof 1) G.sch)

theorem rights_append (a b : List Step) :
    (a ++ b).filterMap Sum.getRight? = a.filterMap Sum.getRight? ++ b.filterMap Sum.getRight? := by
  induction a with
  | nil => rfl
  | cons s a ih =>
      cases s with
      | inl m => exact ih
      | inr c => exact congrArg (c :: ·) ih

theorem firstSteps_rights (ts : List Step) : (firstSteps ts).filterMap Sum.getRight? = ts.filterMap Sum.getRight? := by
  induction ts with
  | nil => rfl
  | cons s ts ih =>
      have e : firstSteps (s :: ts) = Sum.inl ⟨Bool, true⟩ :: s :: firstSteps ts := rfl
      rw [e]
      cases s with
      | inl m => exact ih
      | inr c => exact congrArg (c :: ·) ih

theorem tableSteps_rights (G : TableFiles cls) : (tableSteps G).filterMap Sum.getRight? =
    ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ :: ((G.tape 0).filterMap Sum.getRight? ++ (G.tape 1).filterMap Sum.getRight?) := by
  show ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ :: (tableTape (G.tape 0) (G.tape 1) _).filterMap Sum.getRight? = _
  unfold tableTape
  rw [rights_append, rights_append, firstSteps_rights]
  exact congrArg (_ :: ·) (List.append_nil _)

theorem pcsParams_bne_self (x : PcsParams) : (x != x) = false := by
  obtain ⟨m, l, b, p, n, h⟩ := x
  show (!(m == m && (l == l && (b == b && (p == p && (n == n && h == h)))))) = false
  simp

theorem bytes_beq_self (b : ByteArray) : (b == b) = true := by
  show (b.data == b.data) = true
  exact beq_self_eq_true _

theorem reps_of_runs (G : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    (hmS : G.st.m = (M.toStatement regs).m) (hkS : G.st.kLog = (M.toStatement regs).kLog)
    (hfold : FoldRealizes G.st.fold (M.toStatement regs))
    (hms : G.st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hf : Flock.fast100 G.st.m = some G.sch)
    (hext : ExtraRealizes G.st (M.toStatement regs) G.sess.points
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts))
    (hwf : ∀ s ∈ G.sess.streams.toList, RoundsWF s.rounds) (hex : G.st.extra G.sess.points = .ok G.extras)
    (hruns : G.Runs)
    (hfind : ∀ r < 2, G.Finds r)
    (hdec : ∀ r < 2, decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r))
    (hpar : ∀ r < 2, (G.proof r).params = pcsParams G.st.merkle G.sch)
    (hcap : ∀ r < 2, (G.proof r).cap.foldl (· ++ ·) ByteArray.empty = G.sess.rootB[G.i]!)
    (htape : ∀ r < 2, G.tape r = repSteps G.st (M.toStatement regs) G.sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length (G.proof r) (G.stream r)) :
    (∀ r < 2, G.Finds r ∧ decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r) ∧
      ((G.proof r).params != pcsParams G.st.merkle G.sch) = false ∧
      ((G.proof r).cap.foldl (· ++ ·) ByteArray.empty == G.sess.rootB[G.i]!) = true ∧
      RepTape G.st (M.toStatement regs) G.sch (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length
        (G.proof r) (G.stream r) (G.tape r)) := by
  have hsize := (hext G.extras hex).1
  have hpar' : ∀ r < 2, ((G.proof r).params != pcsParams G.st.merkle G.sch) = false := fun r hr => by
    rw [hpar r hr]; exact pcsParams_bne_self _
  have hcap' : ∀ r < 2, ((G.proof r).cap.foldl (· ++ ·) ByteArray.empty == G.sess.rootB[G.i]!) = true :=
    fun r hr => by rw [hcap r hr]; exact bytes_beq_self _
  have acc : ∀ r < 2, verifyRep G.spec G.st G.sch G.sess G.i r (G.bytes r) = .ok G.sess.rootB[G.i]! := by
    intro r hr2
    obtain ⟨t₄, hprog, hk⟩ := hruns r hr2
    have e := verifyRep_of_checks G.spec G.st G.sch G.sess G.i r (G.bytes r) (G.stream r) G.extras (G.proof r) t₄
      (hfind r hr2) hex (hdec r hr2) (hpar' r hr2) hprog hk (hcap' r hr2)
    rwa [bytes_eq_of_beq (hcap' r hr2)] at e
  obtain ⟨p₀, p₁, s₀, s₁, pend₀, pend₁, hp₀, hp₁, hs₀, hs₁, hpl₀, hpl₁, -⟩ :=
    table_refines_plain G.spec G.st (M.toStatement regs) cls.hm hmS hkS hfold hms
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts) G.sess hext hwf G.sch hf G.i (G.bytes 0)
      (G.bytes 1) _ (acc 0 (by omega)) (acc 1 (by omega))
  have hd₀ := hdec 0 (by omega)
  have hd₁ := hdec 1 (by omega)
  rw [hsize] at hd₀ hd₁
  obtain rfl : p₀ = G.proof 0 := Except.ok.inj (hp₀.symm.trans hd₀)
  obtain rfl : p₁ = G.proof 1 := Except.ok.inj (hp₁.symm.trans hd₁)
  obtain rfl : s₀ = G.stream 0 := Option.some.inj (hs₀.symm.trans (hfind 0 (by omega)))
  obtain rfl : s₁ = G.stream 1 := Option.some.inj (hs₁.symm.trans (hfind 1 (by omega)))
  have ht₀ := repTape_of_plays 0 G.st _ G.sch cls.hm _ _ _ hpl₀
  have ht₁ := repTape_of_plays 1 G.st _ G.sch cls.hm _ _ _ hpl₁
  rw [← htape 0 (by omega)] at ht₀
  rw [← htape 1 (by omega)] at ht₁
  intro r hr
  have hT : RepTape G.st (M.toStatement regs) G.sch (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length
      (G.proof r) (G.stream r) (G.tape r) := by
    rcases (by omega : r = 0 ∨ r = 1) with rfl | rfl
    · exact ht₀
    · exact ht₁
  exact ⟨hfind r hr, hdec r hr, hpar' r hr, hcap' r hr, hT⟩

theorem tableSteps_coins (G : TableFiles cls) (S : Model.Statement) (n : ℕ)
    (hT : ∀ r < 2, RepTape G.st S G.sch n (G.proof r) (G.stream r) (G.tape r)) :
    (tableSteps G).filterMap Sum.getRight? = ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
      ((G.stream 0).rounds.toList.map (·.coins) ++ (G.stream 1).rounds.toList.map (·.coins)).map (coinArr fOf) := by
  rw [tableSteps_rights, (hT 0 (by omega)).2.2, (hT 1 (by omega)).2.2, List.map_append]

theorem transcript_of_runs (G : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) (t : List (List (SMsg T) × ((k : Fin nR) → CT k)))
    (hmS : G.st.m = (M.toStatement regs).m) (hkS : G.st.kLog = (M.toStatement regs).kLog)
    (hfold : FoldRealizes G.st.fold (M.toStatement regs))
    (hms : G.st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hf : Flock.fast100 G.st.m = some G.sch)
    (hext : ExtraRealizes G.st (M.toStatement regs) G.sess.points
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts))
    (hwf : ∀ s ∈ G.sess.streams.toList, RoundsWF s.rounds) (hex : G.st.extra G.sess.points = .ok G.extras)
    (hruns : G.Runs)
    (hfind : ∀ r < 2, G.Finds r)
    (hdec : ∀ r < 2, decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r))
    (hpar : ∀ r < 2, (G.proof r).params = pcsParams G.st.merkle G.sch)
    (hcap : ∀ r < 2, (G.proof r).cap.foldl (· ++ ·) ByteArray.empty = G.sess.rootB[G.i]!)
    (htape : ∀ r < 2, G.tape r = repSteps G.st (M.toStatement regs) G.sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length (G.proof r) (G.stream r))
    (hmsg : t.map (fun e => roundMsgs T e.1) =
      cut ((tableSteps G).map kindOf) ((tableSteps G).filterMap Sum.getLeft?))
    (hcoin : coinSteps CT 0 (t.map Prod.snd) = ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
      ((G.stream 0).rounds.toList.map (·.coins) ++ (G.stream 1).rounds.toList.map (·.coins)).map (coinArr fOf)) :
    (∀ r < 2, G.Finds r ∧ decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r) ∧
      ((G.proof r).params != pcsParams G.st.merkle G.sch) = false ∧
      ((G.proof r).cap.foldl (· ++ ·) ByteArray.empty == G.sess.rootB[G.i]!) = true ∧
      RepTape G.st (M.toStatement regs) G.sch (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length
        (G.proof r) (G.stream r) (G.tape r)) ∧
    schedSteps T CT 0 t = .inl ⟨Cap Dig, capDig (G.proof 0).cap⟩ :: .inr ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
      tableTape (G.tape 0) (G.tape 1) (opensOf (G.proof 0) (G.proof 1) G.sch, saltsOf (G.proof 0) (G.proof 1) G.sch) :=
  have h := reps_of_runs G regs M hmS hkS hfold hms hf hext hwf hex hruns hfind hdec hpar hcap htape
  ⟨h, schedSteps_of_cut T CT (tableSteps G) 0 t hmsg
    (hcoin.trans (tableSteps_coins G _ _ fun r hr => (h r hr).2.2.2.2).symm)⟩

theorem valid_of_runs (G : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) (t : List (List (SMsg T) × ((k : Fin nR) → CT k)))
    (hmS : G.st.m = (M.toStatement regs).m) (hkS : G.st.kLog = (M.toStatement regs).kLog)
    (hfold : FoldRealizes G.st.fold (M.toStatement regs))
    (hms : G.st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hf : Flock.fast100 G.st.m = some G.sch)
    (hext : ExtraRealizes G.st (M.toStatement regs) G.sess.points
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts))
    (hwf : ∀ s ∈ G.sess.streams.toList, RoundsWF s.rounds) (hex : G.st.extra G.sess.points = .ok G.extras)
    (hruns : G.Runs)
    (hfind : ∀ r < 2, G.Finds r)
    (hdec : ∀ r < 2, decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r))
    (hpar : ∀ r < 2, (G.proof r).params = pcsParams G.st.merkle G.sch)
    (hcap : ∀ r < 2, (G.proof r).cap.foldl (· ++ ·) ByteArray.empty = G.sess.rootB[G.i]!)
    (htape : ∀ r < 2, G.tape r = repSteps G.st (M.toStatement regs) G.sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length (G.proof r) (G.stream r))
    (hmsg : t.map (fun e => roundMsgs T e.1) =
      cut ((tableSteps G).map kindOf) ((tableSteps G).filterMap Sum.getLeft?))
    (hcoin : coinSteps CT 0 (t.map Prod.snd) = ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
      ((G.stream 0).rounds.toList.map (·.coins) ++ (G.stream 1).rounds.toList.map (·.coins)).map (coinArr fOf)) :
    G.Valid regs M T CT t :=
  have h := transcript_of_runs G regs M T CT t hmS hkS hfold hms hf hext hwf hex hruns hfind hdec hpar hcap htape hmsg
    hcoin
  ⟨hmS, hkS, hfold, hms, hf, hext, hwf, hex, h.1, h.2⟩

theorem valid_of_layers (G : TableFiles cls) (regs : List (Region cls.kLog (cls.m - cls.kLog))) (M : Hidden cls)
    {nT nR : ℕ} (T : Fin nT → Type) (CT : Fin nR → Type) (ws : List (List (SMsg T))) (cs : List ((k : Fin nR) → CT k))
    (hmS : G.st.m = (M.toStatement regs).m) (hkS : G.st.kLog = (M.toStatement regs).kLog)
    (hfold : FoldRealizes G.st.fold (M.toStatement regs))
    (hms : G.st.merkle = MerkleScheme.sha512 "flock-leaf/sha512-unsalted")
    (hf : Flock.fast100 G.st.m = some G.sch)
    (hext : ExtraRealizes G.st (M.toStatement regs) G.sess.points
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts))
    (hwf : ∀ s ∈ G.sess.streams.toList, RoundsWF s.rounds) (hex : G.st.extra G.sess.points = .ok G.extras)
    (hruns : G.Runs)
    (hfind : ∀ r < 2, G.Finds r)
    (hdec : ∀ r < 2, decodeProof (shapeOf G.st G.sch G.extras.size) (G.bytes r) = .ok (G.proof r))
    (hpar : ∀ r < 2, (G.proof r).params = pcsParams G.st.merkle G.sch)
    (hcap : ∀ r < 2, (G.proof r).cap.foldl (· ++ ·) ByteArray.empty = G.sess.rootB[G.i]!)
    (htape : ∀ r < 2, G.tape r = repSteps G.st (M.toStatement regs) G.sch
      (extraClaims (M.toStatement regs) cls.ptLocal cls.mPts G.pts).length (G.proof r) (G.stream r))
    (hlen : ws.length ≤ cs.length)
    (hmsg : ws.map (roundMsgs T) = cut ((tableSteps G).map kindOf) ((tableSteps G).filterMap Sum.getLeft?))
    (hcoin : coinSteps CT 0 cs = ⟨Fin 2 → Fin cls.mPts → 𝔽, G.pts⟩ ::
      ((G.stream 0).rounds.toList.map (·.coins) ++ (G.stream 1).rounds.toList.map (·.coins)).map (coinArr fOf)) :
    G.Valid regs M T CT (ws.zip cs) :=
  have h := reps_of_runs G regs M hmS hkS hfold hms hf hext hwf hex hruns hfind hdec hpar hcap htape
  ⟨hmS, hkS, hfold, hms, hf, hext, hwf, hex, h, schedSteps_zip T CT (tableSteps G) ws cs hlen hmsg
    (hcoin.trans (tableSteps_coins G _ _ fun r hr => (h r hr).2.2.2.2).symm)⟩

end FlockVBridge.Family
