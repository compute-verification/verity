import FlockSoundness.Discharge.Tables.Steps
import FlockSoundness.Protocol.Discharge.Tables.Sim

/-!
# The simulation of a session of `J` tables (the transfer, R11c, at `--session-tables J`)

flock-circuit's `--session-tables J` (`session_tables`, `config_with`) and flock-verify's `Stmt.setupTables` run one
session of `J` tables: one `Commit` with every table's root, one link exchange whose two points every table reads (the
tables share `m_pts`), the `2J` rep streams `(table j, rep r)` in any order the prover picks, each round's coins drawn
after its bytes, and every proof file after the last coin of every stream (the barrier).

* `liveSession dg J mPts N`: that live session, on the coin server with retained round bytes. `liveSession` at `J = 1`
  is `Refine.liveTable` up to the submissions' table index.
* `modelSession out G`: the model's session of `J` tables, `Commit`'s caps, the link points, then `batch` over the
  tables, each its two reps concurrently (`G j pts r`) and its openings. `sessionBL` is one (`Session.lean`).
* `DecodesTable … j`: table `j`'s decoder is right on accepted sessions: on every live outcome the live verdict
  accepts, table `j`'s reps run on the tapes decoded from its streams `(j, 0)`, `(j, 1)` and its proof files, and the
  model accepts table `j`'s output on those runs. One hypothesis per table.
* `simSession`, `live_le_session`: the simulation, and the transfer: against any live prover, the live verdict holds
  with probability at most the model session's acceptance (every table's) against the simulated prover, whenever
  every table's decoder is right.
-/

namespace FlockSoundness.Discharge.Tables

open Flock FlockLevel3 Game Refine

local notation "𝔽" => FlockLevel3.GF128

/-! ## The live session -/

/-! ## The model's session -/

section Model

variable {P OutT Cp Op : Type} {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)

end Model

/-! ## Tapes -/

section Tapes

variable {J mPts : ℕ} (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg)

end Tapes

/-! ## Following decoded messages -/

section Follow

variable {α : Type}

theorem follow_allInh : ∀ {g : Game α} {ms : List Msg} {g' : Game α}, AllInh g → follow g ms = some g' → AllInh g'
  | _, [], _, hg, h => by simp only [follow, Option.some.injEq] at h; exact h ▸ hg
  | .send M k, ⟨M', m'⟩ :: ms, _, hg, h => by
      by_cases hM : M' = M
      · subst hM
        simp only [follow, dite_true, cast_eq] at h
        exact follow_allInh (hg.send_k _) h
      · simp [follow, hM] at h
  | .ret _, _ :: _, _, _, h => by simp [follow] at h
  | @Game.coin _ _ _ _ _, _ :: _, _, _, h => by simp [follow] at h

end Follow

/-! ## The simulation -/

section Sim

variable {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
  (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
  (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
  (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)

variable {out ok pL opn}

theorem InvJ.mono {caps : Fin J → Cp} {L L' : Game (LiveOutJ J mPts)} {gs : Fin J → Fin 2 → Game P}
    {ts : Fin J → Fin 2 → LiveOutJ J mPts → List Step} (h : InvJ out ok pL opn caps L gs ts)
    (hL : ∀ o, Out L' o → Out L o) : InvJ out ok pL opn caps L' gs ts :=
  fun o ho hp => h o (hL o ho) hp

variable (out)

theorem allInh_batch {A : Type} : ∀ {l : List (Game A)}, (∀ g ∈ l, AllInh g) → AllInh (batch l)
  | [], _ => .ret _
  | g :: gs, h => (AllInh.interleave (h g (by simp)) (allInh_batch fun g' hg' => h g' (by simp [hg']))).map _

theorem allInh_posOf (caps : Fin J → Cp) {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r)) :
    AllInh (posOf out caps gs) :=
  allInh_batch fun g hg => by
    obtain ⟨j, rfl⟩ := List.mem_ofFn.1 hg
    exact AllInh.bind (fun _ => .send inferInstance fun _ => .ret _) (AllInh.interleave (h j 0) (h j 1))

theorem allInh_upd {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, AllInh (gs j r)) (j : Fin J) (r : Fin 2) {g : Game P}
    (hg : AllInh g) : ∀ j' r', AllInh (upd gs j r g j' r') := by
  intro j' r'
  unfold upd
  split_ifs
  · exact hg
  · exact h j' r'

theorem posOf_upd_self (caps : Fin J → Cp) (gs : Fin J → Fin 2 → Game P) (j : Fin J) (r : Fin 2) :
    posOf out caps (upd gs j r (gs j r)) = posOf out caps gs := by
  congr 1
  funext j' r'
  unfold upd
  split_ifs with h
  · rw [h.1, h.2]
  · rfl

theorem sel_pair {P : Type} (p : P × P) : (sel p 0, sel p 1) = p := rfl

variable {out}

theorem dead_follow {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    (hf : follow (gs j r) (dec j (rs j r).length b) = none)
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs)) :
    ∀ o, Out (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) o → ¬ pL o := by
  intro o ho hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  obtain ⟨pend, hpl, -⟩ := hI o ho hp
  have h := hpl j r
  rw [restJ_eq_mid dec one fin rs j r b _ rfl l o hl] at h
  obtain ⟨g', h1, -⟩ := follow_plays _ _ h
  rw [hf] at h1
  cases h1

theorem coinMsg_fst (b : ByteArray) (k : ℕ) (c : Fin k → 𝔽) (one : Bool) :
    (coinMsg one (roundOf dg b k c)).1 = coinTy one k := by
  by_cases hs : one = true ∧ k = 1
  · obtain ⟨rfl, rfl⟩ := hs
    rw [coinMsg_one]
    unfold coinTy
    rw [ite_eq_left ⟨rfl, rfl⟩]
  · rw [coinMsg_vec dg b k c one hs]
    unfold coinTy
    rw [ite_eq_right hs]

theorem dead_mid {caps : Fin J → Cp} {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽} {n : ℕ}
    {rs : Fin J → Fin 2 → List Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ} {gs : Fin J → Fin 2 → Game P}
    {g' : Game P}
    (hg : ∀ (Cm : Type) (iF : Fintype Cm) (iN : Nonempty Cm) (kk : Cm → Game P),
      g' = @Game.coin P Cm iF iN kk → Cm ≠ coinTy (one j (rs j r).length) k)
    (hI : InvJ out ok pL opn caps (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r))) :
    ∀ o, Out (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) o → ¬ pL o := by
  intro o ho hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  obtain ⟨pend, hpl, -⟩ := hI o ho hp
  have h := hpl j r
  rw [upd_self, upd_self, midJ_eq_rest dec one fin rs j r _ l o hl] at h
  cases g' with
  | ret _ => exact absurd (PlaysT.ret_inv h).1 (by simp)
  | send _ _ =>
    obtain ⟨m, hm, -⟩ := PlaysT.send_inv h
    simp at hm
  | coin Cm kk =>
    obtain ⟨x, hx, -⟩ := PlaysT.coin_inv h
    refine hg Cm _ _ kk rfl ?_
    simp only [Sum.inr.injEq] at hx
    exact (congrArg Sigma.fst hx).symm.trans (coinMsg_fst dg b k c _)

theorem allOk_ofFn {v : Fin J → OutT} (h : ∀ j, ok j (v j)) : AllOk ok dflt (List.ofFn v) := fun j => by
  rw [List.getD_eq_getElem?_getD, List.getElem?_ofFn]
  simpa using h j

variable (out ok pL opn)

/-- **The transfer, for a session of `J` tables.** Against any live prover, the live verdict holds with probability at
most the model session's acceptance of every table against the simulated prover, whenever every table's decoder is
right on accepted sessions. -/
theorem live_le_session (N : ℕ) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (hinh : ∀ j pts r, AllInh (G j pts r)) (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j)
    (Pr : Strategy (liveSession dg J mPts N)) :
    prob pL (liveSession dg J mPts N) Pr ≤
      prob (AllOk ok dflt) (modelSession out G)
        ((simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec).strategy Pr) :=
  Sim.prob_le _ Pr

end Sim

end FlockSoundness.Discharge.Tables
