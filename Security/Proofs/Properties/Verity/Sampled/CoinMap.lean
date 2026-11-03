import Proofs.Properties.Verity.Sampled.Project

/-!
# Sampled proofs' coupling: the coin map

The coin map the projection reads (`CoinMap`, in `Proofs.Properties.Verity.Sampled.Project`) writes the live audit's
coins into the period's tape at the projection's stops. A stop is an opening's coin node, where the device draws the
opened round among the draws the coin tree committed to there (`TreeOpenable`, a fiber of the leaf's hash), and releases
the draw's first coins. A live coin is uniform over its challenges and the fiber's draws aren't uniform over their
releases, so no map writes one into the other as the tape's own: the coin map's round and link premises hold within an
explicit term, the written draw's shortfall below uniform at each stop.

* **A written coin** (`shortfall`, `avg_le_write`, `writeCoin`, `avg_le_writeCoin`): at a coin node, the tape's coin
  replaced by `w c` of itself at a uniform `c`. A reading bounded by 1 of the uniform tape is at most its mean on the
  written tape plus `w`'s shortfall, the total variation between the written coin's law and the uniform coin's. A writer
  whose coin releases the challenge pays at least the distance of the release's law from the challenge's
  (`prCoin_rel_le`).
* **The premises within a term** (`CoinMap.RoundClose`, `CoinMap.LinkClose`) and their composition (`roundsErr`,
  `avg_roundsW_tape_le`, `resample_linkW_le`, `couples_of_close`): the joint law's tape is within the start's term and
  the stops' (`couplesErr`) of the uniform tape at every event (`CouplesWithin`), so `Couples` holds wherever their
  sum is at most the hiding term (`couples_of_within`, `couplingTarget_withProject_of_close`). `RoundPreserves` and
  `LinkPreserves` are the premises at term zero (`CoinMap.RoundPreserves.close`, `CoinMap.LinkPreserves.close`).
* **An opening's node** (`OpeningAt`, `openingOf`, `writeOpening`, `openGap`): a node described as the releases'
  opening branch (`relK`), recognised by choice among its descriptions. There the tape's draw is replaced by one the
  device releases as the written bits at every description (`opened_writeOpening`) wherever the fiber holds one, within
  the opening's gap (`avg_le_writeOpening`).
* **The coin map** (`roundBits`, `linkBits`, `openMap`): each round's live coins and the link points, as bits under an
  encoding `e` of the field, written into the stop's opening; its premises hold within the openings' gaps
  (`openMap_roundClose`, `openMap_linkClose`, `openMap_couples`). Its widening and start are parameters, and
  `FirstStopFresh` and the start's term (`StartWithin`) are hypotheses.
-/

namespace Verity.Game

open scoped ENNReal

variable {α : Type}

/-! ## A written coin -/

theorem avg_sum_comm {C E : Type} [Fintype C] [Fintype E] (f : C → E → ℝ≥0∞) :
    avg (fun c => ∑ e, f c e) = ∑ e, avg fun c => f c e := by
  unfold avg
  simp only [Finset.mul_sum]
  exact Finset.sum_comm

open Classical in
/-- **A written coin's shortfall below uniform**: the total variation between the uniform coin's law and the law of
`w c d` at a uniform `c` and a uniform coin `d`. -/
noncomputable def shortfall {P C : Type} [Fintype P] [Fintype C] (w : P → C → C) : ℝ≥0∞ :=
  ∑ e : C, ((Fintype.card C : ℝ≥0∞)⁻¹ - avg fun c : P => avg fun d : C => if w c d = e then 1 else 0)

open Classical in
theorem avg_ind_write {P C : Type} [Fintype P] [Fintype C] (w : P → C → C) (H : C → ℝ≥0∞) :
    avg (fun c : P => avg fun d : C => H (w c d)) =
      ∑ e, (avg fun c : P => avg fun d : C => if w c d = e then 1 else 0) * H e := by
  have h : ∀ c d, H (w c d) = ∑ e, (if w c d = e then 1 else 0) * H e := fun c d => by
    rw [Finset.sum_eq_single (w c d)]
    · simp
    · intro e _ he
      simp [Ne.symm he]
    · simp
  simp only [h, avg_sum_comm, avg_mul_const]

theorem mul_le_add_tsub_of_le_one {k p h : ℝ≥0∞} (hh : h ≤ 1) : k * h ≤ p * h + (k - p) :=
  calc k * h ≤ (p + (k - p)) * h := by gcongr; exact le_add_tsub
    _ = p * h + (k - p) * h := add_mul _ _ _
    _ ≤ p * h + (k - p) := by gcongr; exact mul_le_of_le_one_right zero_le hh

open Classical in
/-- **A written coin**: over a coin and the rest of the tape, a reading bounded by 1 is at most its mean with the coin
replaced by `w c` of itself at a uniform `c`, plus `w`'s shortfall. -/
theorem avg_le_write {P C : Type} [Fintype P] [Nonempty P] [Fintype C] [Nonempty C] {T : C → Type}
    [∀ c, Fintype (T c)] [∀ c, Nonempty (T c)] (w : P → C → C) (G : C × ((c : C) → T c) → ℝ≥0∞)
    (hG : ∀ z, G z ≤ 1) :
    avg G ≤ avg (fun c : P => avg fun z : C × ((c : C) → T c) => G (w c z.1, z.2)) + shortfall w := by
  have hH : ∀ e, (avg fun f : (c : C) → T c => G (e, f)) ≤ 1 := fun e =>
    (avg_mono fun f => hG (e, f)).trans_eq (avg_const 1)
  have h2 : avg (fun c : P => avg fun z : C × ((c : C) → T c) => G (w c z.1, z.2)) =
      avg (fun c : P => avg fun d : C => avg fun f : (c : C) → T c => G (w c d, f)) :=
    congrArg avg (funext fun c => avg_prod (fun z : C × ((c : C) → T c) => G (w c z.1, z.2)))
  rw [avg_prod G, h2, avg_ind_write w fun e => avg fun f : (c : C) → T c => G (e, f)]
  show ∑ e : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (avg fun f : (c : C) → T c => G (e, f)) ≤ _
  unfold shortfall
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun e _ => mul_le_add_tsub_of_le_one (hH e)

/-- **The tape from a coin node with its coin replaced by `w` of itself.** -/
def writeCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (w : C → C) (s' : Strategy g') : CoinTape g' s' → CoinTape g' s' := by
  subst hg
  exact fun z => (w z.1, z.2)

/-- **The coin a tape from a coin node draws there.** -/
def headCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (s' : Strategy g') : CoinTape g' s' → C := by
  subst hg
  exact fun z => z.1

theorem headCoin_writeCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (w : C → C) (s' : Strategy g') (z : CoinTape g' s') :
    headCoin hg s' (writeCoin hg w s' z) = w (headCoin hg s' z) := by
  subst hg
  rfl

/-- **The coin a tape draws at a node, under two descriptions of the node.** -/
theorem headCoin_congr {g' : Game α} {C C' : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    {iF' : Fintype C'} {iN' : Nonempty C'} {k' : C' → Game α} (hg : g' = @Game.coin _ C iF iN k)
    (hg' : g' = @Game.coin _ C' iF' iN' k') (hC : C = C') (s' : Strategy g') (x : CoinTape g' s') :
    headCoin hg' s' x = cast hC (headCoin hg s' x) := by
  subst hg
  injection hg' with h₁ h₂ h₃
  subst h₁
  obtain rfl := eq_of_heq h₂
  obtain rfl := eq_of_heq h₃
  rfl

open Classical in
/-- **A written coin at a coin node**: a reading bounded by 1 of the uniform tape from the node is at most its mean with
the node's coin replaced by `w c` of itself at a uniform `c`, plus `w`'s shortfall. -/
theorem avg_le_writeCoin {P : Type} [Fintype P] [Nonempty P] {g' : Game α} {C : Type} {iF : Fintype C}
    {iN : Nonempty C} {k : C → Game α} (hg : g' = @Game.coin _ C iF iN k) (w : P → C → C) (s' : Strategy g')
    (G : CoinTape g' s' → ℝ≥0∞) (hG : ∀ z, G z ≤ 1) :
    avg G ≤ avg (fun c : P => avg fun z : CoinTape g' s' => G (writeCoin hg (w c) s' z)) + shortfall w := by
  subst hg
  let _ : ∀ c, Fintype (CoinTape (k c) (s' c)) := fun c => CoinTape.fintype (k c) (s' c)
  have _ : ∀ c, Nonempty (CoinTape (k c) (s' c)) := fun c => CoinTape.nonempty (k c) (s' c)
  exact avg_le_write (T := fun c => CoinTape (k c) (s' c)) w G hG

open Classical in
/-- **No written coin does better than its release's law**: where the coin `w c d` releases `enc c` (`rel`), an event
of the release of a uniform coin is at most as likely as the event of `enc` at a uniform `c`, plus `w`'s shortfall. -/
theorem prCoin_rel_le {P C B : Type} [Fintype P] [Nonempty P] [Fintype C] [Nonempty C] (w : P → C → C)
    (rel : C → B) (enc : P → B) (hw : ∀ c d, rel (w c d) = enc c) (A : B → Prop) :
    prCoin (fun e : C => A (rel e)) ≤ prCoin (fun c : P => A (enc c)) + shortfall w := by
  have hP : prCoin (fun c : P => A (enc c)) =
      avg fun c : P => avg fun d : C => if A (rel (w c d)) then 1 else 0 := by
    unfold prCoin
    refine congrArg avg (funext fun c => ?_)
    simp only [hw]
    exact (avg_const _).symm
  rw [hP, avg_ind_write w fun e => if A (rel e) then (1 : ℝ≥0∞) else 0]
  show ∑ e : C, (Fintype.card C : ℝ≥0∞)⁻¹ * (if A (rel e) then 1 else 0) ≤ _
  unfold shortfall
  rw [← Finset.sum_add_distrib]
  exact Finset.sum_le_sum fun e _ => mul_le_add_tsub_of_le_one (by split_ifs <;> simp)

/-- **The element a uniform `d` picks in `S`**: `S`'s element at `d`'s index modulo `S`'s size; `d` itself where `S` is
empty. -/
noncomputable def pickIn {C : Type} [Fintype C] (S : Finset C) (d : C) : C :=
  if h : S.Nonempty then
    (S.equivFin.symm ⟨(Fintype.equivFin C d).1 % S.card, Nat.mod_lt _ (Finset.card_pos.2 h)⟩).1
  else d

theorem pickIn_mem {C : Type} [Fintype C] {S : Finset C} (h : S.Nonempty) (d : C) : pickIn S d ∈ S := by
  unfold pickIn
  split
  · exact Subtype.mem _
  · contradiction

end Verity.Game

namespace Verity.Sampled

open Verity.Game
open FlockSoundness.Discharge.Tables (SubmitJ LiveOutJ liveRoundsJ KLJ liveSession addRoundJ)
open FlockSoundness.Refine (roundOf)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The coin map's premises within a term -/

/-- **The coin map writes a round's live coins in within `γ` of the tape's own**: at every stop, table count and
submission, a reading bounded by 1 of a uniform tape from the stop is at most its mean with a uniform round's coins
written in, plus `γ` there. -/
def CoinMap.RoundClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (M : CoinMap g σ Ω R mPts)
    (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) : Prop :=
  ∀ (q : Pos g σ Claims.Msg (Fin R)) (J : ℕ) (s : SubmitJ J) (G : CoinTape q.g' q.s' → ℝ≥0∞), (∀ z, G z ≤ 1) →
    avg G ≤ avg (fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.round (q.withTape z) J s c z)) +
      γ q J s

/-- **The coin map writes the live link points in within `γL` of the tape's own**: at every stop, a reading bounded by
1 of a uniform tape from the stop is at most its mean with uniform link points written in, plus `γL` there. -/
def CoinMap.LinkClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (M : CoinMap g σ Ω R mPts)
    (γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) : Prop :=
  ∀ (q : Pos g σ Claims.Msg (Fin R)) (G : CoinTape q.g' q.s' → ℝ≥0∞), (∀ z, G z ≤ 1) →
    avg G ≤ avg (fun pts : Fin 2 → Fin mPts → 𝔽 => avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z)) +
      γL q

theorem CoinMap.RoundPreserves.close {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ}
    {M : CoinMap g σ Ω R mPts} (h : M.RoundPreserves) : M.RoundClose fun _ _ _ => 0 := fun q J s G _ => by
  rw [h q J s G, add_zero]

theorem CoinMap.LinkPreserves.close {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ}
    {M : CoinMap g σ Ω R mPts} (h : M.LinkPreserves) : M.LinkClose fun _ => 0 := fun q G _ => by
  rw [h q G, add_zero]

section Session

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (dg : ByteArray → ByteArray) (J : ℕ)
  (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞)

/-- **One stop's error** at the submission decoded there: none where the play ends; `γ` there and the next rounds'
error over the tapes from the stop where a submission is decoded. -/
noncomputable def stopErr (e : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞) (q : Pos g σ Claims.Msg (Fin R)) :
    Option (SubmitJ J) → ℝ≥0∞
  | none => 0
  | some s => γ q J s + Pos.resample e q.consume

/-- **The rounds' error** from the position `p`: `γ` at each stop a uniform tape from `p` reaches. -/
noncomputable def roundsErr : ℕ → Pos g σ Claims.Msg (Fin R) → ℝ≥0∞
  | 0, _ => 0
  | N + 1, p => stopErr J γ (roundsErr N) p.toStop (D.sub J p.toStop.modes)

variable {D M dg J γ} in
/-- **The rounds' joint law's tape is within the rounds' error of uniform from where it starts**: wherever the coin map
writes each round's coins in within `γ` of the tape's own, an event of the tape has, over a uniform tape from the
position `p`, at most its probability under the rounds' joint law plus the rounds' error as its share of the tapes
through `p`. -/
theorem avg_roundsW_tape_le (hR : M.RoundClose γ) {β : Type} (fin : LiveOutJ J mPts → β)
    (Q : CoinTape g σ → Prop) :
    ∀ (N : ℕ) (rootB : Array ByteArray) (pts : Fin 2 → Fin mPts → 𝔽) (rs : Fin J → Fin 2 → List Flock.Round)
      (p : Pos g σ Claims.Msg (Fin R)),
      avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
        avg (fun z : CoinTape p.g' p.s' => prob (fun o => Q o.2) (roundsW D M dg J fin rootB pts N rs (p.withTape z))
          (roundsWS D M dg J fin rootB pts N rs (p.withTape z))) +
        avg fun z : CoinTape p.g' p.s' => roundsErr D J γ N (p.withTape z)
  | 0, _, _, _, _ => le_self_add
  | N + 1, rootB, pts, rs, p => by
    let nxt : (q : Pos g σ Claims.Msg (Fin R)) → (s : SubmitJ J) → (Fin s.n → 𝔽) → Game (β × CoinTape g σ) :=
      fun q s c => roundsW D M dg J fin rootB pts N (addRoundJ rs s.tab s.rep (roundOf dg s.bytes s.n c))
        (q.rewrite (M.round q J s c)).consume
    let Ψ : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q => prob (fun o => Q o.2)
      (stepW D J fin rootB pts rs q (nxt q) (D.sub J q.modes))
      (stepWS D J fin rootB pts rs q (nxt q) (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) (D.sub J q.modes))
    let E : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q => stopErr J γ (roundsErr D J γ N) q (D.sub J q.modes)
    have hstop : ∀ (q : Pos g σ Claims.Msg (Fin R)) (o : Option (SubmitJ J)),
        avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
          avg (fun z : CoinTape q.g' q.s' => prob (fun o => Q o.2) (stepW D J fin rootB pts rs (q.withTape z)
            (nxt (q.withTape z)) o) (stepWS D J fin rootB pts rs (q.withTape z) (nxt (q.withTape z))
              (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) o)) +
          avg fun z : CoinTape q.g' q.s' => stopErr J γ (roundsErr D J γ N) (q.withTape z) o := fun q o => by
      cases o with
      | none => exact le_self_add
      | some s =>
        let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => prob (fun o => Q o.2)
          (roundsW D M dg J fin rootB pts N rs (q.consume.withTape z))
          (roundsWS D M dg J fin rootB pts N rs (q.consume.withTape z))
        have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 =>
              G (M.round (q.withTape z) J s c z)) + avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s :=
          calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
            _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' =>
                  G (M.round (q.withTape z) J s c z)) + γ (q.withTape z₀) J s :=
                avg_mono fun z₀ => hR (q.withTape z₀) J s G fun _ => prob_le_one _ _ _
            _ = _ := by
                rw [avg_add, avg_const, avg_comm]
        have h1 : avg (fun z : CoinTape q.g' q.s' => prob (fun o => Q o.2) (stepW D J fin rootB pts rs (q.withTape z)
              (nxt (q.withTape z)) (some s)) (stepWS D J fin rootB pts rs (q.withTape z) (nxt (q.withTape z))
                (fun _ _ => roundsWS D M dg J fin rootB pts N _ _) (some s))) =
            avg fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z) :=
          congrArg avg (funext fun z => Finset.sum_congr rfl fun c _ => congrArg (HMul.hMul _)
            (prob_roundsW_tape D M dg J fin fin Q N rootB rootB pts pts _ rs _))
        have h2 : avg (fun z : CoinTape q.g' q.s' => stopErr J γ (roundsErr D J γ N) (q.withTape z) (some s)) =
            (avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErr D J γ N (q.consume.withTape z) := by
          show avg (fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s + Pos.resample (roundsErr D J γ N) q.consume) = _
          rw [avg_add, avg_const]
          rfl
        calc avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z))
            ≤ avg G + avg fun z : CoinTape q.g' q.s' => roundsErr D J γ N (q.consume.withTape z) :=
              avg_roundsW_tape_le hR fin Q N rootB pts rs q.consume
          _ ≤ (avg (fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z)) +
                avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErr D J γ N (q.consume.withTape z) := by
              gcongr
          _ = _ := by rw [h1, h2, add_assoc]
    calc avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z))
        = avg fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z).toStop :=
          congrArg avg (funext fun z => (tapeInd_toStop Q _).symm)
      _ = avg fun z : CoinTape p.g' p.s' => Pos.resample (tapeInd Q) (p.withTape z).toStop := Pos.avg_toStop _ p
      _ ≤ avg fun z : CoinTape p.g' p.s' =>
            Pos.resample Ψ (p.withTape z).toStop + Pos.resample E (p.withTape z).toStop :=
          avg_mono fun z => hstop _ (D.sub J (p.withTape z).toStop.modes)
      _ = (avg fun z : CoinTape p.g' p.s' => Pos.resample Ψ (p.withTape z).toStop) +
            avg fun z : CoinTape p.g' p.s' => Pos.resample E (p.withTape z).toStop := avg_add _ _
      _ = (avg fun z : CoinTape p.g' p.s' => Ψ (p.withTape z).toStop) +
            avg fun z : CoinTape p.g' p.s' => E (p.withTape z).toStop := by
          rw [Pos.avg_toStop Ψ p, Pos.avg_toStop E p]
      _ = _ := rfl

variable {D M dg J γ} in
/-- **The session's joint law's tape is within the link's and the rounds' error of uniform from its first stop**:
wherever the coin map writes the link points in within `γL` of the tape's own and each round's coins within `γ`, an
event of the tape has, over a uniform tape from the stop `q`, at most its probability under the session's joint law
from there plus `γL` and the rounds' error after the stop, as its share of the tapes through `q`. -/
theorem resample_linkW_le {γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hL : M.LinkClose γL) (hR : M.RoundClose γ)
    {β : Type} (fin : LiveOutJ J mPts → β) (Q : CoinTape g σ → Prop) (N : ℕ) (q : Pos g σ Claims.Msg (Fin R)) :
    Pos.resample (tapeInd Q) q ≤
      Pos.resample (fun q' => avg fun pts : Fin 2 → Fin mPts → 𝔽 => prob (fun o => Q o.2)
        (roundsW D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)
        (roundsWS D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)) q +
      (Pos.resample γL q + Pos.resample (roundsErr D J γ N) q.consume) := by
  let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => prob (fun o => Q o.2)
    (roundsW D M dg J fin (D.root q.modes) 0 N (fun _ _ => []) (q.consume.withTape z))
    (roundsWS D M dg J fin (D.root q.modes) 0 N (fun _ _ => []) (q.consume.withTape z))
  have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        G (M.link (q.withTape z) pts z)) + avg fun z : CoinTape q.g' q.s' => γL (q.withTape z) :=
    calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
      _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun pts : Fin 2 → Fin mPts → 𝔽 =>
            avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z)) + γL (q.withTape z₀) :=
          avg_mono fun z₀ => hL (q.withTape z₀) G fun _ => prob_le_one _ _ _
      _ = _ := by rw [avg_add, avg_const, avg_comm]
  have hS : Pos.resample (fun q' => avg fun pts : Fin 2 → Fin mPts → 𝔽 => prob (fun o => Q o.2)
        (roundsW D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)
        (roundsWS D M dg J fin (D.root q'.modes) pts N (fun _ _ => []) (q'.rewrite (M.link q' pts)).consume)) q =
      avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 => G (M.link (q.withTape z) pts z) :=
    congrArg avg (funext fun z => congrArg avg (funext fun pts =>
      prob_roundsW_tape D M dg J fin fin Q N _ _ pts 0 _ _ _))
  calc Pos.resample (tapeInd Q) q
      ≤ avg G + Pos.resample (roundsErr D J γ N) q.consume :=
        avg_roundsW_tape_le hR fin Q N (D.root q.modes) 0 (fun _ _ => []) q.consume
    _ ≤ (avg (fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 => G (M.link (q.withTape z) pts z)) +
          avg fun z : CoinTape q.g' q.s' => γL (q.withTape z)) + Pos.resample (roundsErr D J γ N) q.consume := by
        gcongr
    _ = _ := by rw [hS, add_assoc]; rfl

end Session

/-! ## An opening's node -/

/-- **An opening's coin node, described**: `g'` is the device's draw of stream `p.1`'s round `p.2` of the tree `T` among
the draws the tree committed to there, on the frame `.opening i s n` answering `b`, with the releases' continuation
after it (`relK`'s opening branch). -/
structure OpeningAt {β : Type} (g' : Game β) where
  T : OpenTree
  p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)
  i : ℕ
  s : ℕ
  n : ℕ
  b : ByteArray
  k : ℕ
  fs : List (String × ℕ)
  ts : List OpenTree
  f : List Release → Game β
  iF : Fintype (TreeOpenable T.nonce T.spec T.sample p.1 p.2)
  iN : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2)
  eq : g' = @Game.coin β (TreeOpenable T.nonce T.spec T.sample p.1 p.2) iF iN fun d =>
    relK k fs (ts.set i (T.bump p.1)) fun log => f (⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)

namespace OpeningAt

variable {β : Type} {g' : Game β}

/-- The draws an opening's node draws among. -/
abbrev Openable (o : OpeningAt g') : Type := TreeOpenable o.T.nonce o.T.spec o.T.sample o.p.1 o.p.2

instance (o : OpeningAt g') : Fintype o.Openable := o.iF

instance (o : OpeningAt g') : Nonempty o.Openable := o.iN

/-- Two descriptions of one node draw among one type. -/
theorem openable_eq (o o' : OpeningAt g') : o.Openable = o'.Openable := by
  have h := o.eq.symm.trans o'.eq
  injection h

open Classical in
/-- **The draws released as `bits`**: those whose first `n` coins are `bits` at every description of the node. -/
noncomputable def classOf (o : OpeningAt g') (bits : List Bool) : Finset o.Openable :=
  Finset.univ.filter fun d => ∀ o' : OpeningAt g', TreeDraw.opened (cast (openable_eq o o') d).1 o'.n = bits

end OpeningAt

open Classical in
/-- **The node's description**, where `g'` is an opening's coin node. -/
noncomputable def openingOf {β : Type} (g' : Game β) : Option (OpeningAt g') :=
  if h : Nonempty (OpeningAt g') then some (Classical.choice h) else none

/-- **The release `bits` written into an opening's node**: at an opening's coin node, the tape's draw there replaced by
the draw it picks among those released as `bits`; every other tape unchanged. -/
noncomputable def writeOpening {β : Type} {g' : Game β} (s' : Strategy g') (bits : List Bool) :
    CoinTape g' s' → CoinTape g' s' :=
  (openingOf g').elim id fun o => writeCoin o.eq (pickIn (o.classOf bits)) s'

/-- **The opening's gap** at the node `g'` and the release `bits` of a uniform `P`: the written draw's shortfall below
uniform at an opening's coin node; none at every other node. -/
noncomputable def openGap {β P : Type} [Fintype P] (g' : Game β) (bits : P → List Bool) : ℝ≥0∞ :=
  (openingOf g').elim 0 fun o => shortfall fun c => pickIn (o.classOf (bits c))

/-- **A release written into an opening's node is within the opening's gap of the tape's own**: a reading bounded by 1
of the uniform tape from the node is at most its mean with the release `bits c` of a uniform `c` written in, plus the
opening's gap. -/
theorem avg_le_writeOpening {β P : Type} [Fintype P] [Nonempty P] {g' : Game β} (bits : P → List Bool)
    (s' : Strategy g') (G : CoinTape g' s' → ℝ≥0∞) (hG : ∀ z, G z ≤ 1) :
    avg G ≤ avg (fun c : P => avg fun z : CoinTape g' s' => G (writeOpening s' (bits c) z)) + openGap g' bits := by
  unfold writeOpening openGap
  cases openingOf g' with
  | none =>
    refine le_of_eq ?_
    show avg G = avg (fun _ : P => avg fun z : CoinTape g' s' => G z) + 0
    rw [avg_const, add_zero]
  | some o => exact avg_le_writeCoin o.eq (fun c => pickIn (o.classOf (bits c))) s' G hG

open Classical in
/-- **The written release is `bits`**: at an opening's node whose draws include one released as `bits`, the written
tape's draw there is released as `bits` at every description of the node. -/
theorem opened_writeOpening {β : Type} {g' : Game β} (s' : Strategy g') (bits : List Bool) (z : CoinTape g' s')
    (o : OpeningAt g') (ho : openingOf g' = some o) (h : (o.classOf bits).Nonempty) (o' : OpeningAt g') :
    TreeDraw.opened (headCoin o'.eq s' (writeOpening s' bits z)).1 o'.n = bits := by
  have hw : writeOpening s' bits z = writeCoin o.eq (pickIn (o.classOf bits)) s' z := by
    unfold writeOpening
    rw [ho]
    rfl
  rw [hw, headCoin_congr o.eq o'.eq (OpeningAt.openable_eq o o') s', headCoin_writeCoin]
  exact (Finset.mem_filter.1 (pickIn_mem h (headCoin o.eq s' z))).2 o'

/-! ## The coin map -/

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- **A round's live coins as bits**: each challenge's 128 bits under `e`, in order. -/
def roundBits {m : ℕ} (c : Fin m → 𝔽) : List Bool := (List.ofFn fun i => List.ofFn (e (c i))).flatten

/-- **The live link points as bits**: the first point's, then the second's. -/
def linkBits {m : ℕ} (pts : Fin 2 → Fin m → 𝔽) : List Bool := roundBits e (pts 0) ++ roundBits e (pts 1)

/-- **The coin map that writes the live coins into the openings**, at the widening `T` and the start `start`: at each
stop, the link points' bits or a round's coins' bits under `e`, written into the stop's opening (`writeOpening`). -/
noncomputable def openMap {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (T : Type) [Fintype T]
    [Nonempty T] (start : Ω × T → CoinTape g σ) : CoinMap g σ Ω R mPts where
  T := T
  start := start
  link p pts := writeOpening p.s' (linkBits e pts)
  round p _ _ c := writeOpening p.s' (roundBits e c)

/-- **The coin map writes each round's coins in within the stop's opening's gap.** -/
theorem openMap_roundClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (T : Type) [Fintype T]
    [Nonempty T] (start : Ω × T → CoinTape g σ) :
    (openMap e (R := R) (mPts := mPts) T start).RoundClose fun q _ s =>
      openGap q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) := fun q _ _ G hG =>
  avg_le_writeOpening (roundBits e) q.s' G hG

/-- **The coin map writes the link points in within the stop's opening's gap.** -/
theorem openMap_linkClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ} (T : Type) [Fintype T]
    [Nonempty T] (start : Ω × T → CoinTape g σ) :
    (openMap e (R := R) (mPts := mPts) T start).LinkClose fun q =>
      openGap q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) := fun q G hG =>
  avg_le_writeOpening (linkBits e) q.s' G hG

/-! ## The coin map's premise within a term -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The joint law's tape is within `ε` of the uniform tape**, at every event `Q` of the period's tape. -/
def CouplesWithin (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ε : ℝ≥0∞) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤
    prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) + ε

/-- **The coin map's premise at a term within the hiding term.** -/
theorem couples_of_within {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {ε : ℝ≥0∞}
    (h : CouplesWithin V θ F c A rd σ D M ε) (hε : ε ≤ A.hidingTerm) : Couples V θ F c A rd σ D M := fun Q =>
  (h Q).trans (add_le_add le_rfl hε)

/-- **The coin map's start is within `ε₀` of the uniform tape**: over the widened sample, the period's tape the start
gives is, at every reading of it bounded by 1, within `ε₀` of the uniform tape. -/
def StartWithin (A : Live) (σ : Strategy (V.game θ F c)) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (ε₀ : ℝ≥0∞) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (A.extend M.widening).L.Ω => H (M.start ω)) + ε₀

/-- **The stops' term**: over the widened sample, `γL` at the first stop and the rounds' error after it, over the
tapes from the stop. -/
noncomputable def couplesErr (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (γ : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞) : ℝ≥0∞ :=
  avg fun ω : (A.extend M.widening).L.Ω => Pos.resample γL (firstStop V θ F c A rd σ M ω) +
    Pos.resample (roundsErr D (A.J ((A.extend M.widening).L.draw ω) (regOf V θ F c A rd σ D)) γ A.N)
      (firstStop V θ F c A rd σ M ω).consume

/-- **The coin map's premise within a term, off its local ones**: the joint law's tape is within the start's term and
the stops' of the uniform tape at every event, wherever the coin map writes the link points in within `γL` of the tape's
own and each round's coins within `γ`, its start leaves the tape from the first stop fresh, and its start is within
`ε₀` of the uniform tape. -/
theorem couples_of_close (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    {γ : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞}
    (hR : M.RoundClose γ) (hL : M.LinkClose γL) (hF : FirstStopFresh V θ F c A rd σ M)
    (hS : StartWithin V θ F c A σ M ε₀) :
    CouplesWithin V θ F c A rd σ D M (ε₀ + couplesErr V θ F c A rd σ D M γ γL) := fun Q => by
  let R₀ := regOf V θ F c A rd σ D
  let G : Finset (Fin A.n) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞ := fun dr q =>
    avg fun pts : Fin 2 → Fin A.mP → 𝔽 => prob (fun o => Q o.2)
      (roundsW D M A.dg (A.J dr R₀) (fun t => (R₀, dr, A.wr dr R₀ t)) (D.root q.modes) pts A.N
        (fun _ _ => []) (q.rewrite (M.link q pts)).consume)
      (roundsWS D M A.dg (A.J dr R₀) (fun t => (R₀, dr, A.wr dr R₀ t)) (D.root q.modes) pts A.N
        (fun _ _ => []) (q.rewrite (M.link q pts)).consume)
  have hG : prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M) =
      avg fun ω : (A.extend M.widening).L.Ω => G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω) := by
    calc prob (fun o => Q o.2) (jointG V θ F c A rd σ D M) (jointS V θ F c A rd σ D M)
        = avg fun ω : (A.extend M.widening).L.Ω => prob (fun o => Q o.2)
            (sessionW D M A.dg (A.J ((A.extend M.widening).L.draw ω) R₀)
              (fun t => (R₀, (A.extend M.widening).L.draw ω, A.wr ((A.extend M.widening).L.draw ω) R₀ t)) A.N
              (Pos.root (roundCuts V θ F c rd) (M.start ω)))
            (sessionWS D M A.dg (A.J ((A.extend M.widening).L.draw ω) R₀)
              (fun t => (R₀, (A.extend M.widening).L.draw ω, A.wr ((A.extend M.widening).L.draw ω) R₀ t)) A.N
              (Pos.root (roundCuts V θ F c rd) (M.start ω))) := rfl
      _ = avg fun ω : (A.extend M.widening).L.Ω =>
            G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω) :=
          congrArg avg (funext fun ω => prob_sessionW_tape D M A.dg _ _ Q A.N _)
  have hU : prCoin Q = avg fun x : CoinTape (V.game θ F c) σ =>
      Pos.resample (tapeInd Q) (Pos.root (roundCuts V θ F c rd) x).toStop := by
    refine Eq.trans ?_ (Pos.avg_walk (tapeInd Q) (V.game θ F c) σ (roundCuts V θ F c rd) id)
    refine congrArg avg (funext fun x => ?_)
    show _ = tapeInd Q (Pos.root (roundCuts V θ F c rd) x).toStop
    rw [tapeInd, Pos.toStop_tape]
    rfl
  rw [ofReal_pr_eq_prCoin, hU, hG]
  calc avg (fun x : CoinTape (V.game θ F c) σ => Pos.resample (tapeInd Q) (Pos.root (roundCuts V θ F c rd) x).toStop)
      ≤ avg (fun ω : (A.extend M.widening).L.Ω => Pos.resample (tapeInd Q) (firstStop V θ F c A rd σ M ω)) + ε₀ :=
        hS _ fun x => resample_tapeInd_le_one Q _
    _ ≤ avg (fun ω : (A.extend M.widening).L.Ω =>
          Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω) +
            (Pos.resample γL (firstStop V θ F c A rd σ M ω) +
              Pos.resample (roundsErr D (A.J ((A.extend M.widening).L.draw ω) R₀) γ A.N)
                (firstStop V θ F c A rd σ M ω).consume)) + ε₀ :=
        add_le_add (avg_mono fun ω => resample_linkW_le (D := D) (dg := A.dg) hL hR _ Q A.N _) le_rfl
    _ = (avg fun ω : (A.extend M.widening).L.Ω =>
          Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω)) +
          couplesErr V θ F c A rd σ D M γ γL + ε₀ := by
        rw [avg_add]
        rfl
    _ = _ := by
        rw [← hF G]
        ring

/-- **The coin map that writes the live coins into the openings, within the openings' gaps**: at its widening `T` and
start `start`, the joint law's tape is within the start's term and the openings' gaps over the stops of the uniform
tape, wherever the start leaves the tape from the first stop fresh and is within `ε₀` of the uniform tape. -/
theorem openMap_couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (T : Type) [Fintype T] [Nonempty T]
    (start : A.Ω × T → CoinTape (V.game θ F c) σ) {ε₀ : ℝ≥0∞}
    (hF : FirstStopFresh V θ F c A rd σ (openMap e T start))
    (hS : StartWithin V θ F c A σ (openMap e T start) ε₀) :
    CouplesWithin V θ F c A rd σ D (openMap e T start) (ε₀ + couplesErr V θ F c A rd σ D (openMap e T start)
      (fun q _ s => openGap q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool))) :=
  couples_of_close V θ F c A rd σ D _ (openMap_roundClose e T start) (openMap_linkClose e T start) hF hS

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the projection, off the coin map's premises within a term**: `couplingTarget_withProject` with
the coin map's premise discharged by `couples_of_close` at each commitment and strategy, wherever the start's term and
the stops' sum to at most the hiding term. -/
theorem couplingTarget_withProject_of_close (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (γ : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) →
      (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞)
    (ε₀ : (c : ByteArray) → Strategy (V.game θ F c) → ℝ≥0∞)
    (ha : LoggedBound V k (K.withProject V rd D M) P θ F)
    (hR : ∀ c σ, (M A θ F c σ).RoundClose (γ c σ)) (hL : ∀ c σ, (M A θ F c σ).LinkClose (γL c σ))
    (hF : ∀ c σ, FirstStopFresh V θ F c A (rd A θ F c) σ (M A θ F c σ))
    (hS : ∀ c σ, StartWithin V θ F c A σ (M A θ F c σ) (ε₀ c σ))
    (hε : ∀ c σ, ε₀ c σ + couplesErr V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (γ c σ) (γL c σ) ≤
      A.hidingTerm)
    (hD : ∀ c σ, Decodes V θ F c k P A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)) :
    CouplingTarget V k ((K.withProject V rd D M).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D M A P θ F ha
    (fun c σ => couples_of_within V θ F c
      (couples_of_close V θ F c A _ σ _ _ (hR c σ) (hL c σ) (hF c σ) (hS c σ)) (hε c σ)) hD

end Verity.Sampled
