import Proofs.Flock.Claims.Bridge.Dead

/-!
# The simulation's returns are the decoded live outcomes

The session's simulation (`simSession`) returns only where the live verdict holds, and there the model's outcome is
every table's output on its reps' runs of their whole decoded tapes, at the cap the roots decode to and the openings
the proof files decode to (`RetOK`). Its dead branches are at the tables' positions, each rep's game there a position
of its game at the session's link points (`DeadAt`, `Resid`), so each rep's outcome there is an outcome of its game
(`Resid.out`). Both hold of every node (`all_simSession`), by the rounds' recursion, carrying that each rep's position
was reached by its tape's prefix (`Reach`).

So with the dead branches answering the openings with `nul`, a model event `Q` that no outcome at the openings `nul`
satisfies is no likelier, against the simulated strategy, than a live outcome that passes the live verdict and whose
decoded model outcome is in `Q` (`prob_le_decoded`): the model's escapes arise only on runs the live verdict accepts.
-/

namespace Claims.Bridge

open FlockSoundness FlockSoundness.Game FlockSoundness.Refine FlockSoundness.Discharge.Tables

local notation "𝔽" => FlockLevel3.GF128

/-- A run from where the model lands after following messages is a run of the messages, then that run. -/
theorem plays_of_follow {α : Type} : ∀ {g : Game α} {ms : List Msg} {g' : Game α} {t : List Step} {a : α},
    follow g ms = some g' → PlaysT g' t a → PlaysT g (ms.map .inl ++ t) a
  | _, [], _, _, _, h, hp => by
      simp only [follow, Option.some.injEq] at h
      subst h
      simpa using hp
  | .send M k, ⟨M', m'⟩ :: ms, _, _, _, h, hp => by
      by_cases hM : M' = M
      · subst hM
        simp only [follow, dite_true, cast_eq] at h
        exact .send m' (plays_of_follow h hp)
      · simp [follow, hM] at h
  | .ret _, _ :: _, _, _, _, h, _ => by simp [follow] at h
  | @Game.coin _ _ _ _ _, _ :: _, _, _, _, h, _ => by simp [follow] at h

/-- **`g'` is a position of `g`**: reached from it by messages and coins. -/
inductive Resid {α : Type} (g : Game α) : Game α → Prop
  | refl : Resid g g
  | send {M : Type} {k : M → Game α} (m : M) : Resid g (.send M k) → Resid g (k m)
  | coin {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α} (c : C) :
      Resid g (@Game.coin α C iF iN k) → Resid g (k c)

/-- An outcome at a position of `g` is an outcome of `g`. -/
theorem Resid.out {α : Type} {g g' : Game α} {a : α} (h : Resid g g') (ha : Out g' a) : Out g a := by
  induction h with
  | refl => exact ha
  | send m _ ih => exact ih (.send m ha)
  | coin c _ ih => exact ih (.coin c ha)

theorem resid_follow {α : Type} {g₀ : Game α} : ∀ {g : Game α} {ms : List Msg} {g' : Game α},
    Resid g₀ g → follow g ms = some g' → Resid g₀ g'
  | _, [], _, hr, h => by
      simp only [follow, Option.some.injEq] at h
      subst h
      exact hr
  | .send M k, ⟨M', m'⟩ :: ms, _, hr, h => by
      by_cases hM : M' = M
      · subst hM
        simp only [follow, dite_true, cast_eq] at h
        exact resid_follow (.send m' hr) h
      · simp [follow, hM] at h
  | .ret _, _ :: _, _, _, h => by simp [follow] at h
  | @Game.coin _ _ _ _ _, _ :: _, _, _, h => by simp [follow] at h

theorem resid_upd {P : Type} {J mPts : ℕ} {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}
    {pts : Fin 2 → Fin mPts → 𝔽} {gs : Fin J → Fin 2 → Game P} {j : Fin J} {r : Fin 2} {g g' : Game P}
    (h : ∀ j' r', Resid (G j' pts r') (upd gs j r g j' r')) (hg : Resid (G j pts r) g → Resid (G j pts r) g') :
    ∀ j' r', Resid (G j' pts r') (upd gs j r g' j' r') := by
  intro j' r'
  by_cases hjr : j' = j ∧ r' = r
  · obtain ⟨rfl, rfl⟩ := hjr
    rw [upd_self]
    exact hg (by simpa only [upd_self] using h j' r')
  · rw [upd_ne _ _ hjr]
    simpa only [upd_ne _ _ hjr] using h j' r'

theorem resid_upd_of {P : Type} {J mPts : ℕ} {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}
    {pts : Fin 2 → Fin mPts → 𝔽} {gs : Fin J → Fin 2 → Game P} (h : ∀ j r, Resid (G j pts r) (gs j r)) {j : Fin J}
    {r : Fin 2} {g' : Game P} (hg : Resid (G j pts r) g') : ∀ j' r', Resid (G j' pts r') (upd gs j r g' j' r') :=
  resid_upd (g := gs j r) (fun j' r' => by
    by_cases hjr : j' = j ∧ r' = r
    · obtain ⟨rfl, rfl⟩ := hjr
      rw [upd_self]
      exact h j' r'
    · rw [upd_ne _ _ hjr]
      exact h j' r') fun _ => hg

/-- **The positions were reached by the tapes' prefixes**: on every outcome of the live subgame `L`, a run of rep
`(j, r)`'s position `gs j r` on its remaining tape `ts j r` is a run of its initial game on its whole tape. -/
def Reach {P : Type} {J mPts : ℕ} (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (pts : Fin 2 → Fin mPts → 𝔽) (L : Game (LiveOutJ J mPts)) (gs : Fin J → Fin 2 → Game P)
    (ts : Fin J → Fin 2 → LiveOutJ J mPts → List Step) : Prop :=
  ∀ o, Out L o → ∀ j r p, PlaysT (gs j r) (ts j r o) p → PlaysT (G j pts r) (repTapeJ dec one fin o j r) p

theorem Reach.mono {P : Type} {J mPts : ℕ} {dec : Fin J → ℕ → ByteArray → List Msg} {one : Fin J → ℕ → Bool}
    {fin : Fin J → Fin 2 → Array ByteArray → List Msg} {G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P}
    {pts : Fin 2 → Fin mPts → 𝔽} {L L' : Game (LiveOutJ J mPts)} {gs : Fin J → Fin 2 → Game P}
    {ts : Fin J → Fin 2 → LiveOutJ J mPts → List Step} (h : Reach dec one fin G pts L gs ts)
    (hL : ∀ o, Out L' o → Out L o) : Reach dec one fin G pts L' gs ts :=
  fun o ho => h o (hL o ho)

theorem reach_init {P : Type} {J mPts : ℕ} (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P)
    (pts : Fin 2 → Fin mPts → 𝔽) (L : Game (LiveOutJ J mPts)) :
    Reach dec one fin G pts L (fun j => G j pts) (restJ dec one fin fun _ _ => []) := by
  intro o _ j r p hp
  simpa [restJ, repTapeJ] using hp

theorem reach_follow {P : Type} {J mPts : ℕ} (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg)
    (one : Fin J → ℕ → Bool) (fin : Fin J → Fin 2 → Array ByteArray → List Msg)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽}
    {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ}
    {gs : Fin J → Fin 2 → Game P} {g' : Game P} (hf : follow (gs j r) (dec j (rs j r).length b) = some g')
    (hR : Reach dec one fin G pts (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs)) :
    Reach dec one fin G pts (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
      (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)) := by
  intro o ho j' r' p hp
  obtain ⟨c, l, hl⟩ := reps_coin dg ho
  refine hR o ho j' r' p ?_
  by_cases hjr : j' = j ∧ r' = r
  · obtain ⟨rfl, rfl⟩ := hjr
    rw [upd_self, upd_self] at hp
    rw [restJ_eq_mid dec one fin rs j' r' b _ rfl l o hl]
    exact plays_of_follow hf hp
  · rwa [upd_ne _ _ hjr, upd_ne _ _ hjr] at hp

theorem reach_coin {P : Type} {J mPts : ℕ} (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg)
    (one : Fin J → ℕ → Bool) (fin : Fin J → Fin 2 → Array ByteArray → List Msg)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) {rootB : Array ByteArray} {pts : Fin 2 → Fin mPts → 𝔽}
    {n : ℕ} {rs : Fin J → Fin 2 → List Flock.Round} {j : Fin J} {r : Fin 2} {b : ByteArray} {k : ℕ}
    {gs : Fin J → Fin 2 → Game P} {Cm : Type} {iF : Fintype Cm} {iN : Nonempty Cm} {kk : Cm → Game P}
    (e : Cm ≃ (Fin k → 𝔽))
    (he : (∃ (hs : one j (rs j r).length = true ∧ k = 1) (hC : Cm = 𝔽), e = eOne hC hs.2) ∨
      (¬ (one j (rs j r).length = true ∧ k = 1) ∧ ∃ hC : Cm = (Fin k → 𝔽), e = Equiv.cast hC))
    (hR : Reach dec one fin G pts (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k))
      (upd gs j r (@Game.coin P Cm iF iN kk)) (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)))
    (x : Cm) :
    Reach dec one fin G pts (roundL dg rootB pts n rs j r b k (e x)) (upd gs j r (kk x))
      (restJ dec one fin (addRoundJ rs j r (roundOf dg b k (e x)))) := by
  intro o ho j' r' p hp
  obtain ⟨-, -, hl⟩ := out_liveJ dg ho
  refine hR o (.coin (e x) ho) j' r' p ?_
  by_cases hjr : j' = j ∧ r' = r
  · obtain ⟨rfl, rfl⟩ := hjr
    obtain ⟨l, hl'⟩ := hl j' r'
    rw [addRoundJ_self, List.append_assoc, List.singleton_append] at hl'
    rw [upd_self] at hp
    rw [upd_self, upd_self, midJ_eq_rest dec one fin rs j' r' _ l o hl', coinMsg_e dg b k _ x e he]
    exact .coin x hp
  · rw [upd_ne _ _ hjr, restJ_other dec one fin rs hjr] at hp
    rwa [upd_ne _ _ hjr, upd_ne _ _ hjr]

/-- **A return of the session's simulation**: the live outcome `b` passes the live verdict, and the model outcome `a`
is every table's output on its reps' runs `pend` of their whole decoded tapes, at the cap `b`'s roots decode to and
the openings its proof files decode to. -/
def RetOK {P OutT Cp Op : Type} {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT) (pL : LiveOutJ J mPts → Prop)
    (opn : Array ByteArray → Fin J → Op) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (b : LiveOutJ J mPts) (a : List OutT) : Prop :=
  pL b ∧ ∃ pend : Fin J → P × P, (∀ j r, PlaysT (G j b.pts r) (repTapeJ dec one fin b j r) (sel (pend j) r)) ∧
    a = List.ofFn fun j => out j (capOf b.rootB j) (pend j) (opn b.proofs j)

/-- **A dead branch at the tables' positions**, every rep's game there with its message types inhabited and a position
of the rep's game `G j pts r` at the session's link points. -/
def DeadAt {P OutT Cp Op : Type} {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (g : Game (List OutT)) (_ : Strategy g) : Prop :=
  ∃ (caps : Fin J → Cp) (pts : Fin 2 → Fin mPts → 𝔽) (gs : Fin J → Fin 2 → Game P), (∀ j r, AllInh (gs j r)) ∧
    (∀ j r, Resid (G j pts r) (gs j r)) ∧ g = posOf out caps gs

theorem all_simEndJ {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (rs : Fin J → Fin 2 → List Flock.Round) (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn (capOf rootB) (KLJ rootB pts rs) gs (restJ dec one fin rs))
    (hR : Reach dec one fin G pts (KLJ rootB pts rs) gs (restJ dec one fin rs))
    (hres : ∀ j r, Resid (G j pts r) (gs j r)) :
    Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
      (simEndJ out ok dflt pL opn dec one fin (capOf rootB) rootB pts rs gs h hI) := by
  unfold simEndJ
  intro proofs
  dsimp only
  split
  · rename_i hp
    refine Sim.all_ofSteps _ _ ⟨hp, Classical.choose (inv_end dec one fin hI proofs hp), fun j r => ?_, rfl⟩
    refine hR _ (.send proofs (.ret _)) j r _ ?_
    simpa [restJ, tapeFrom] using (Classical.choose_spec (inv_end dec one fin hI proofs hp)).1 j r
  · exact ⟨capOf rootB, pts, gs, h, hres, rfl⟩

theorem all_coinJ {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (j : Fin J) (r : Fin 2) (b : ByteArray) (k : ℕ)
    (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn (capOf rootB) (roundL dg rootB pts n rs j r b k c) gs'
        (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs j r b k c) (posOf out (capOf rootB) gs'))
    (hnext : ∀ c gs' h' hI', Reach dec one fin G pts (roundL dg rootB pts n rs j r b k c) gs'
        (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) → (∀ j r, Resid (G j pts r) (gs' j r)) →
      Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G) (next c gs' h' hI')) :
    ∀ (g' : Game P) (hg : AllInh g')
      (hI : InvJ out ok pL opn (capOf rootB) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
        (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r))),
      Reach dec one fin G pts (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) (upd gs j r g')
        (upd (restJ dec one fin rs) j r (midJ dec one fin rs j r)) →
      (∀ j' r', Resid (G j' pts r') (upd gs j r g' j' r')) →
      Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
        (coinJ out ok dflt pL opn dg dec one fin (capOf rootB) rootB pts n rs j r b k gs h next g' hg hI)
  | @Game.coin _ Cm iF iN kk, hg, hI, hR, hres => by
      have hgo : ∀ (e : Cm ≃ (Fin k → 𝔽))
          (he : (∃ (hs : one j (rs j r).length = true ∧ k = 1) (hC : Cm = 𝔽), e = eOne hC hs.2) ∨
            (¬ (one j (rs j r).length = true ∧ k = 1) ∧ ∃ hC : Cm = (Fin k → 𝔽), e = Equiv.cast hC)),
          Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
            (Sim.ofSteps ((ctxAt out (capOf rootB) gs j r).coin kk).steps (Sim.coin e fun x =>
              Sim.castR (((ctxAt out (capOf rootB) gs j r).coin kk).eq x).symm
                (next (e x) (upd gs j r (kk x)) (allInh_upd h j r (hg.coin_k x))
                  (inv_coin dg dec one fin e he hI x)))) := fun e he =>
        Sim.all_ofSteps _ _ (Sim.all_coin _ _ fun x =>
          Sim.all_castR _ _ (hnext _ _ _ _ (reach_coin dg dec one fin G e he hR x)
            (resid_upd hres fun hc => .coin x hc)))
      have hdead : DeadAt out G (posOf out (capOf rootB) (upd gs j r (@Game.coin P Cm iF iN kk)))
          (Classical.choice (allInh_posOf out (capOf rootB) (allInh_upd h j r hg)).strategy) :=
        ⟨capOf rootB, pts, _, allInh_upd h j r hg, hres, rfl⟩
      unfold coinJ
      dsimp only
      split_ifs with hs hC hC
      · exact hgo (eOne hC hs.2) (.inl ⟨hs, hC, rfl⟩)
      · exact hdead
      · exact hgo (Equiv.cast hC) (.inr ⟨hs, hC, rfl⟩)
      · exact hdead
  | .ret _, hg, _, _, hres => ⟨capOf rootB, pts, _, allInh_upd h j r hg, hres, rfl⟩
  | .send _ _, hg, _, _, hres => ⟨capOf rootB, pts, _, allInh_upd h j r hg, hres, rfl⟩

theorem all_roundJ {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽)
    (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (j : Fin J) (r : Fin 2) (b : ByteArray) (k : ℕ)
    (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
    (hI : InvJ out ok pL opn (capOf rootB) (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs
      (restJ dec one fin rs))
    (next : (c : Fin k → 𝔽) → (gs' : Fin J → Fin 2 → Game P) → (∀ j r, AllInh (gs' j r)) →
      InvJ out ok pL opn (capOf rootB) (roundL dg rootB pts n rs j r b k c) gs'
        (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) →
      Sim pL (AllOk ok dflt) (roundL dg rootB pts n rs j r b k c) (posOf out (capOf rootB) gs'))
    (hnext : ∀ c gs' h' hI', Reach dec one fin G pts (roundL dg rootB pts n rs j r b k c) gs'
        (restJ dec one fin (addRoundJ rs j r (roundOf dg b k c))) → (∀ j r, Resid (G j pts r) (gs' j r)) →
      Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G) (next c gs' h' hI'))
    (hR : Reach dec one fin G pts (.coin (Fin k → 𝔽) (roundL dg rootB pts n rs j r b k)) gs (restJ dec one fin rs))
    (hres : ∀ j r, Resid (G j pts r) (gs j r)) :
    Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
      (roundJ out ok dflt pL opn dg dec one fin (capOf rootB) rootB pts n rs j r b k gs h hI next) := by
  unfold roundJ
  split
  · exact ⟨capOf rootB, pts, gs, h, hres, rfl⟩
  · rename_i g' hf
    exact Sim.all_ofSteps _ _ (all_coinJ out ok dflt pL opn dg dec one fin capOf G rootB pts n rs j r b k gs h next
      hnext g' _ _ (reach_follow dg dec one fin G hf hR) (resid_upd_of hres (resid_follow (hres j r) hf)))

theorem all_simRJ {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) :
    ∀ (n : ℕ) (rs : Fin J → Fin 2 → List Flock.Round) (gs : Fin J → Fin 2 → Game P) (h : ∀ j r, AllInh (gs j r))
      (hI : InvJ out ok pL opn (capOf rootB) ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs
        (restJ dec one fin rs)),
      Reach dec one fin G pts ((liveRoundsJ dg J n rs).bind (KLJ rootB pts)) gs (restJ dec one fin rs) →
      (∀ j r, Resid (G j pts r) (gs j r)) →
      Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
        (simRJ out ok dflt pL opn dg dec one fin (capOf rootB) rootB pts n rs gs h hI)
  | 0, rs, gs, h, hI, hR, hres => all_simEndJ out ok dflt pL opn dec one fin capOf G rootB pts rs gs h hI hR hres
  | n + 1, rs, gs, h, hI, hR, hres => by
      intro m
      rcases m with _ | ⟨j, r, b, k⟩
      · exact all_simEndJ out ok dflt pL opn dec one fin capOf G rootB pts rs gs h _
          (hR.mono fun _ h => .send none h) hres
      · exact all_roundJ out ok dflt pL opn dg dec one fin capOf G rootB pts n rs j r b k gs h _ _
          (fun _ gs' h' hI' hR' hres' =>
            all_simRJ out ok dflt pL opn dg dec one fin capOf G rootB pts n _ gs' h' hI' hR' hres')
          (hR.mono fun _ h => .send (some (⟨j, r, b, k⟩ : SubmitJ J)) h) hres

/-- **Every node of the session's simulation**: its returns are the decoded live outcomes the live verdict accepts,
and its dead branches are at the tables' positions. -/
theorem all_simSession {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp) (N : ℕ)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ j pts r, AllInh (G j pts r))
    (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j) :
    Sim.All (RetOK out pL opn dec one fin capOf G) (DeadAt out G)
      (simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec) :=
  fun rootB pts => all_simRJ out ok dflt pL opn dg dec one fin capOf G rootB pts N _ _ _ _
    (reach_init dec one fin G pts _) fun _ _ => .refl

/-- **The model's escapes arise only on accepted live runs.** With the dead branches answering the openings with
`nul`, a model event `Q` that no outcome with every table's openings `nul` satisfies is no likelier, against the
simulated strategy, than a live outcome passing the live verdict whose decoded model outcome is in `Q`. -/
theorem prob_le_decoded {P OutT Cp Op : Type} [Nonempty Op] {J mPts : ℕ} (out : Fin J → Cp → P × P → Op → OutT)
    (ok : Fin J → OutT → Prop) (dflt : OutT) (pL : LiveOutJ J mPts → Prop) (opn : Array ByteArray → Fin J → Op)
    (dg : ByteArray → ByteArray) (dec : Fin J → ℕ → ByteArray → List Msg) (one : Fin J → ℕ → Bool)
    (fin : Fin J → Fin 2 → Array ByteArray → List Msg) (capOf : Array ByteArray → Fin J → Cp) (N : ℕ)
    (G : Fin J → (Fin 2 → Fin mPts → 𝔽) → Fin 2 → Game P) (hinh : ∀ j pts r, AllInh (G j pts r))
    (hdec : ∀ j, DecodesTable out ok pL opn dec one fin capOf G j) (nul : Op) {Q : List OutT → Prop}
    (hQ : ∀ (caps : Fin J → Cp) outs, (∀ j : Fin J, ∃ pend, outs.getD j dflt = out j (caps j) pend nul) → ¬ Q outs)
    (Pr : Strategy (liveSession dg J mPts N)) :
    prob Q (modelSession out G)
        ((Sim.mapDead (answerWith nul) (simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec)).strategy
          Pr) ≤
      prob (fun b => ∃ a, RetOK out pL opn dec one fin capOf G b a ∧ Q a) (liveSession dg J mPts N) Pr :=
  Sim.prob_le_of_all _
    (Sim.All.mono (R' := fun b a => Q a → ∃ a, RetOK out pL opn dec one fin capOf G b a ∧ Q a)
      (fun _ a hr hq => ⟨a, hr, hq⟩) (fun _ _ h => h) _
      (Sim.all_mapDead (answerWith nul) (D' := fun g σ => prob Q g σ = 0)
        (fun _ σ ⟨caps, _, _, hg, _, e⟩ => by
          subst e
          exact prob_posOf_answerWith out nul caps hg dflt (hQ caps) σ)
        _ (all_simSession out ok dflt pL opn dg dec one fin capOf N G hinh hdec)))
    Pr

end Claims.Bridge
