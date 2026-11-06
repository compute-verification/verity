module

public import Proofs.Properties.Verity.Sampled.Project
public import Proofs.Properties.Verity.SampledCoins

/-!
# Sampled proofs' coupling: the coin map

The coin map the projection reads (`CoinMap`, in `Proofs.Properties.Verity.Sampled.Project`) writes the live audit's
coins into the period's tape at the projection's stops. A stop is an opening's coin node, where the device flips the
opened round among the rounds the coin tree committed to there (`TreeOpenable`, a fiber of the leaf's hash), and
releases the round's first coins. A live coin is uniform over its challenges and the fiber's rounds aren't uniform over
their releases, so no map writes one into the other as the tape's own: the coin map's round and link premises hold
within an explicit term, the written round's shortfall below uniform at each stop. The start's premises hold exactly: a fresh
frame's coins are 512 uniform bits, one for one with 64 of the bytes the sample reads.

* **A filled tape** (`fill`, `avg_fill`, `avg_walk_fill`): along the tape's play up to the first stop, each chosen coin
  node takes the next of the coins `κ`, and nothing from the stop on is written. Over uniform `κ` the filled tape is
  uniform, and at any `κ` the tape from the first stop is uniform and independent of the walk to it.
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
  opening branch (`relK`), recognised by choice among its descriptions. There the tape's round is replaced by one the
  device releases as the written bits at every description (`opened_writeOpening`) wherever the fiber holds one, within
  the opening's gap (`avg_le_writeOpening`).
* **The coin map** (`roundBits`, `linkBits`, `openMap`): each round's live coins and the link points, as bits under an
  encoding `e` of the field, written into the stop's opening; its premises hold within the openings' gaps
  (`openMap_roundClose`, `openMap_linkClose`, `openMap_couples`), at any widening and start that leave the tape from
  the first stop fresh (`FirstStopFresh`) and are within `ε₀` of the uniform tape (`StartWithin`).
* **The sample-coupled coin map** (`FreshAt`, `freshSel`, `srcCoins`, `sampleStart`, `sampleMap`, `sampleMaps`):
  widened by a tape of the period's game, its start fills the fresh frames of the sample's family with the bytes the
  sample reads, as coins (`srcCoins`, whose bytes are the sample's: `bytesOf_srcCoins`). It leaves the tape from the
  first stop fresh (`sampleMap_fresh`, by `avg_prod_walk_fill`) and is the uniform tape wherever the sample's coins are
  uniform (`sampleMap_start`, `avg_srcCoins`), so its premise holds within the openings' gaps alone
  (`sampleMap_couples`, `couplingTarget_withProject_of_sample`).
-/

@[expose] public section

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

/-- **The coin a tape from a coin node holds there.** -/
def headCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (s' : Strategy g') : CoinTape g' s' → C := by
  subst hg
  exact fun z => z.1

theorem headCoin_writeCoin {g' : Game α} {C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game α}
    (hg : g' = @Game.coin _ C iF iN k) (w : C → C) (s' : Strategy g') (z : CoinTape g' s') :
    headCoin hg s' (writeCoin hg w s' z) = w (headCoin hg s' z) := by
  subst hg
  rfl

/-- **The coin a tape holds at a node, under two descriptions of the node.** -/
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

/-! ## A filled tape -/

section Fill

variable {V ι Q : Type}

/-- **Which coin nodes a fill writes**: at a coin node, an equivalence from the written coins onto the node's, or
none. -/
abbrev FillSel (α Q : Type) : Type 1 :=
  ∀ {C : Type} [Fintype C] [Nonempty C], (C → Game α) → Option (Q ≃ C)

open Classical in
/-- **One coin node's fill**: where the node is chosen and a coin is left, the node's coin is the first left under the
node's equivalence and the rest fill its continuation (`r`); else the tape's coin stays, and its continuation is
filled with all of them. -/
noncomputable def fillAt {C : Type} {T : C → Type} (r : (c : C) → (m : ℕ) → (Fin m → Q) → T c → T c)
    (o : Option (Q ≃ C)) :
    (m : ℕ) → (Fin m → Q) → C × ((c : C) → T c) → C × ((c : C) → T c)
  | m + 1, κ, z =>
    match o with
    | some e => (e (κ 0), Function.update z.2 (e (κ 0)) (r (e (κ 0)) m (Fin.tail κ) (z.2 (e (κ 0)))))
    | none => (z.1, Function.update z.2 z.1 (r z.1 (m + 1) κ (z.2 z.1)))
  | 0, κ, z => (z.1, Function.update z.2 z.1 (r z.1 0 κ (z.2 z.1)))

open Classical in
/-- **The tape filled with the coins `κ` up to the next stop**: along the tape's play from the node, through the
prover's messages and the tape's coins, each chosen coin node (`sel`) takes the next of `κ` while any is left, up to
the first node at which one of the cuts is fixed; the tape from there, and every branch off the play, stays. -/
noncomputable def fill (sel : FillSel α Q) : (g' : Game α) → (s' : Strategy g') → (ι → DigestCut V g') → (m : ℕ) →
    (Fin m → Q) → CoinTape g' s' → CoinTape g' s'
  | .ret _, _, _, _, _, z => z
  | .send _ k, s', Ks, m, κ, z =>
    if ∃ i, (Ks i).isHere then z else fill sel (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) m κ z
  | @Game.coin _ C iF iN k, s', Ks, m, κ, z =>
    if ∃ i, (Ks i).isHere then z else
      fillAt (T := fun c => CoinTape (k c) (s' c))
        (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) (@sel C iF iN k) m κ z

open Classical in
theorem avg_comp_eq_sum {Ω C : Type} [Fintype Ω] [Fintype C] (t : Ω → C) (H : C → ℝ≥0∞) :
    avg (fun ω => H (t ω)) = ∑ e, prCoin (fun ω => t ω = e) * H e := by
  have h : ∀ ω, H (t ω) = ∑ e, (if t ω = e then 1 else 0) * H e := fun ω => by
    rw [Finset.sum_eq_single (t ω)]
    · simp
    · intro e _ he
      simp [Ne.symm he]
    · simp
  simp only [h, avg_sum_comm, avg_mul_const]
  rfl

/-- **A map whose events have their probabilities under the uniform law carries means to means.** -/
theorem avg_comp_of_prCoin {Ω C : Type} [Fintype Ω] [Fintype C] {t : Ω → C}
    (h : ∀ E : C → Prop, prCoin (fun ω => E (t ω)) = prCoin E) (H : C → ℝ≥0∞) :
    avg (fun ω => H (t ω)) = avg H := by
  rw [avg_comp_eq_sum t H, show avg H = avg (fun c => H (id c)) from rfl, avg_comp_eq_sum id H]
  exact Finset.sum_congr rfl fun e _ => by rw [h fun c => c = e]; rfl

/-- **A uniform dependent function's value at `c` is uniform and independent of the others.** -/
theorem avg_pi_at {C : Type} [DecidableEq C] [Fintype C] {T : C → Type} [∀ c, Fintype (T c)]
    [∀ c, Nonempty (T c)] (c : C) (F : T c → (T c → (c : C) → T c) → ℝ≥0∞) :
    avg (fun f : (c : C) → T c => F (f c) fun y => Function.update f c y) =
      avg fun f : (c : C) → T c => avg fun a : T c => F a fun y => Function.update f c y := by
  refine (Eq.trans ?_ (avg_update_pi c fun f => F (f c) fun y => Function.update f c y)).symm
  simp only [Function.update_self, Function.update_idem]

/-- The coins `Fin (m + 1) → Q`, as their first and the rest. -/
def piCons (Q : Type) (m : ℕ) : Q × (Fin m → Q) ≃ (Fin (m + 1) → Q) where
  toFun p := Fin.cons p.1 p.2
  invFun f := (f 0, Fin.tail f)
  left_inv p := by simp
  right_inv f := Fin.cons_self_tail f

open Classical in
theorem avg_fillAt_keep [Fintype Q] [Nonempty Q] {C : Type} [Fintype C] [Nonempty C] {T : C → Type}
    [∀ c, Fintype (T c)] [∀ c, Nonempty (T c)] (r : (c : C) → (m : ℕ) → (Fin m → Q) → T c → T c)
    (hr : ∀ c m (Φ : T c → ℝ≥0∞), avg (fun κ : Fin m → Q => avg fun y => Φ (r c m κ y)) = avg Φ) (m : ℕ)
    (Φ : C × ((c : C) → T c) → ℝ≥0∞) :
    avg (fun κ : Fin m → Q => avg fun z : C × ((c : C) → T c) =>
      Φ (z.1, Function.update z.2 z.1 (r z.1 m κ (z.2 z.1)))) = avg Φ := by
  rw [avg_comm]
  calc avg (fun z : C × ((c : C) → T c) => avg fun κ : Fin m → Q =>
        Φ (z.1, Function.update z.2 z.1 (r z.1 m κ (z.2 z.1))))
      = avg fun z : C × ((c : C) → T c) => avg fun a : T z.1 => avg fun κ : Fin m → Q =>
          Φ (z.1, Function.update z.2 z.1 (r z.1 m κ a)) :=
        avg_pick (T := T) fun c a u => avg fun κ : Fin m → Q => Φ (c, u (r c m κ a))
    _ = avg fun z : C × ((c : C) → T c) => avg fun a : T z.1 => Φ (z.1, Function.update z.2 z.1 a) := by
        refine congrArg avg (funext fun z => ?_)
        rw [avg_comm]
        exact hr z.1 m fun y => Φ (z.1, Function.update z.2 z.1 y)
    _ = avg fun z : C × ((c : C) → T c) => Φ (z.1, Function.update z.2 z.1 (z.2 z.1)) :=
        (avg_pick (T := T) fun c a u => Φ (c, u a)).symm
    _ = avg Φ := by simp only [Function.update_eq_self]

open Classical in
theorem avg_fillAt_write [Fintype Q] [Nonempty Q] {C : Type} [Fintype C] [Nonempty C] {T : C → Type}
    [∀ c, Fintype (T c)] [∀ c, Nonempty (T c)] (r : (c : C) → (m : ℕ) → (Fin m → Q) → T c → T c)
    (hr : ∀ c m (Φ : T c → ℝ≥0∞), avg (fun κ : Fin m → Q => avg fun y => Φ (r c m κ y)) = avg Φ) (e : Q ≃ C)
    (m : ℕ) (Φ : C × ((c : C) → T c) → ℝ≥0∞) :
    avg (fun κ : Fin (m + 1) → Q => avg fun z : C × ((c : C) → T c) =>
      Φ (e (κ 0), Function.update z.2 (e (κ 0)) (r (e (κ 0)) m (Fin.tail κ) (z.2 (e (κ 0)))))) = avg Φ := by
  have h1 : ∀ (a : Q) (κ : Fin m → Q), avg (fun z : C × ((c : C) → T c) =>
      Φ (e a, Function.update z.2 (e a) (r (e a) m κ (z.2 (e a))))) =
      avg fun f : (c : C) → T c => avg fun y : T (e a) => Φ (e a, Function.update f (e a) (r (e a) m κ y)) := by
    intro a κ
    rw [avg_prod]
    simp only [avg_const]
    exact avg_pi_at (e a) fun y u => Φ (e a, u (r (e a) m κ y))
  calc avg (fun κ : Fin (m + 1) → Q => avg fun z : C × ((c : C) → T c) =>
        Φ (e (κ 0), Function.update z.2 (e (κ 0)) (r (e (κ 0)) m (Fin.tail κ) (z.2 (e (κ 0))))))
      = avg fun p : Q × (Fin m → Q) => avg fun z : C × ((c : C) → T c) =>
          Φ (e p.1, Function.update z.2 (e p.1) (r (e p.1) m p.2 (z.2 (e p.1)))) :=
        avg_equiv (piCons Q m).symm fun p : Q × (Fin m → Q) => avg fun z : C × ((c : C) → T c) =>
          Φ (e p.1, Function.update z.2 (e p.1) (r (e p.1) m p.2 (z.2 (e p.1))))
    _ = avg fun a : Q => avg fun f : (c : C) → T c => Φ (e a, f) := by
        rw [avg_prod]
        refine congrArg avg (funext fun a => ?_)
        simp only [h1]
        rw [avg_comm]
        calc avg (fun f : (c : C) → T c => avg fun κ : Fin m → Q => avg fun y : T (e a) =>
              Φ (e a, Function.update f (e a) (r (e a) m κ y)))
            = avg fun f : (c : C) → T c => avg fun y : T (e a) => Φ (e a, Function.update f (e a) y) :=
              congrArg avg (funext fun f => hr (e a) m fun y => Φ (e a, Function.update f (e a) y))
          _ = avg fun f : (c : C) → T c => Φ (e a, f) := avg_update_pi (e a) fun f => Φ (e a, f)
    _ = avg Φ := by
        rw [avg_equiv e fun c => avg fun f : (c : C) → T c => Φ (c, f), avg_prod]

/-- **A filled tape is a uniform tape**: over uniform coins `κ` and a uniform tape from the node, the tape filled with
`κ` is a uniform tape from the node. -/
theorem avg_fill [Fintype Q] [Nonempty Q] (sel : FillSel α Q) :
    ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (m : ℕ) (Φ : CoinTape g' s' → ℝ≥0∞),
      avg (fun κ : Fin m → Q => avg fun z : CoinTape g' s' => Φ (fill sel g' s' Ks m κ z)) = avg Φ
  | .ret _, _, _, _, Φ => by simp only [fill]; exact avg_const _
  | .send M k, s', Ks, m, Φ => by
    by_cases hf : ∃ i, (Ks i).isHere
    · have e : ∀ κ z, fill sel (.send M k) s' Ks m κ z = z := fun κ z => by
        rw [fill]; exact ite_eq_left hf
      simp only [e]
      exact avg_const _
    · have e : ∀ κ z, fill sel (.send M k) s' Ks m κ z =
          fill sel (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) m κ z := fun κ z => by
        rw [fill]; exact ite_eq_right hf
      simp only [e]
      exact avg_fill sel (k s'.1) s'.2 _ m Φ
  | @Game.coin _ C iF iN k, s', Ks, m, Φ => by
    classical
    let _ : ∀ c, Fintype (CoinTape (k c) (s' c)) := fun c => CoinTape.fintype (k c) (s' c)
    have _ : ∀ c, Nonempty (CoinTape (k c) (s' c)) := fun c => CoinTape.nonempty (k c) (s' c)
    by_cases hf : ∃ i, (Ks i).isHere
    · have e : ∀ κ z, fill sel (@Game.coin _ C iF iN k) s' Ks m κ z = z := fun κ z => by
        rw [fill]; exact ite_eq_left hf
      simp only [e]
      exact avg_const _
    · let r : (c : C) → (m : ℕ) → (Fin m → Q) → CoinTape (k c) (s' c) → CoinTape (k c) (s' c) :=
        fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y
      have hr : ∀ c m (Φ : CoinTape (k c) (s' c) → ℝ≥0∞),
          avg (fun κ : Fin m → Q => avg fun y => Φ (r c m κ y)) = avg Φ := fun c m Φ =>
        avg_fill sel (k c) (s' c) _ m Φ
      have e : ∀ κ z, fill sel (@Game.coin _ C iF iN k) s' Ks m κ z =
          fillAt (T := fun c => CoinTape (k c) (s' c)) r (@sel C iF iN k) m κ z := fun κ z => by
        rw [fill]; exact ite_eq_right hf
      simp only [e]
      cases m with
      | zero => exact avg_fillAt_keep (T := fun c => CoinTape (k c) (s' c)) r hr 0 Φ
      | succ m =>
        cases @sel C iF iN k with
        | none => exact avg_fillAt_keep (T := fun c => CoinTape (k c) (s' c)) r hr (m + 1) Φ
        | some e => exact avg_fillAt_write (T := fun c => CoinTape (k c) (s' c)) r hr e m Φ

variable {g : Game α} {σ : Strategy g}

/-- **The tape from the next stop stays uniform and independent of the walk to it on a filled tape**: at any coins `κ`,
averaged over the tape from a node, any reading of the walk's stop on the tape filled with `κ` is its average over the
tapes from the stop. -/
theorem avg_walk_fill (sel : FillSel α Q) (G : Pos g σ V ι → ℝ≥0∞) :
    ∀ (g' : Game α) (s' : Strategy g') (Ks : ι → DigestCut V g') (m : ℕ) (κ : Fin m → Q)
      (back : CoinTape g' s' → CoinTape g σ),
      avg (fun z : CoinTape g' s' => G (Pos.walk g' s' Ks (fill sel g' s' Ks m κ z) back)) =
        avg fun z : CoinTape g' s' => Pos.resample G (Pos.walk g' s' Ks (fill sel g' s' Ks m κ z) back)
  | .ret a, s', Ks, _, _, back => Pos.avg_resample_lit G (.ret a) s' Ks back
  | .send M k, s', Ks, m, κ, back => by
    by_cases hf : ∃ i, (Ks i).isHere
    · have e : ∀ z, Pos.walk (.send M k) s' Ks (fill sel (.send M k) s' Ks m κ z) back =
          ⟨.send M k, s', Ks, z, back⟩ := fun z => by
        rw [fill]; exact (congrArg (fun x => Pos.walk _ s' Ks x back) (ite_eq_left hf)).trans (ite_eq_left hf)
      simp only [e]
      exact Pos.avg_resample_lit G _ s' Ks back
    · have e : ∀ z, Pos.walk (.send M k) s' Ks (fill sel (.send M k) s' Ks m κ z) back =
          Pos.walk (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1)
            (fill sel (k s'.1) s'.2 (fun i => (Ks i).atSend s'.1) m κ z) back := fun z => by
        rw [fill]; exact (congrArg (fun x => Pos.walk _ s' Ks x back) (ite_eq_right hf)).trans (ite_eq_right hf)
      simp only [e]
      exact avg_walk_fill sel G (k s'.1) s'.2 _ m κ back
  | @Game.coin _ C iF iN k, s', Ks, m, κ, back => by
    classical
    let _ : ∀ c, Fintype (CoinTape (k c) (s' c)) := fun c => CoinTape.fintype (k c) (s' c)
    have _ : ∀ c, Nonempty (CoinTape (k c) (s' c)) := fun c => CoinTape.nonempty (k c) (s' c)
    by_cases hf : ∃ i, (Ks i).isHere
    · have e : ∀ z, Pos.walk (@Game.coin _ C iF iN k) s' Ks (fill sel (@Game.coin _ C iF iN k) s' Ks m κ z) back =
          ⟨@Game.coin _ C iF iN k, s', Ks, z, back⟩ := fun z => by
        rw [fill]; exact (congrArg (fun x => Pos.walk _ s' Ks x back) (ite_eq_left hf)).trans (ite_eq_left hf)
      simp only [e]
      exact Pos.avg_resample_lit G _ s' Ks back
    · let W : (c : C) → (m' : ℕ) → (Fin m' → Q) → CoinTape (k c) (s' c) →
          (CoinTape (k c) (s' c) → (c : C) → CoinTape (k c) (s' c)) → Pos g σ V ι :=
        fun c m' κ' a u => Pos.walk (k c) (s' c) (fun i => (Ks i).atCoin c)
          (fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m' κ' a) fun y => back (c, u y)
      have hw : ∀ (c : C) (m' : ℕ) (κ' : Fin m' → Q) (z : CoinTape (@Game.coin _ C iF iN k) s'),
          Pos.walk (@Game.coin _ C iF iN k) s' Ks
            (c, Function.update z.2 c (fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m' κ' (z.2 c))) back =
            W c m' κ' (z.2 c) fun y => Function.update z.2 c y := fun c m' κ' z => by
        refine (ite_eq_right hf).trans ?_
        simp only [W, Function.update_self, Function.update_idem]
      have key : ∀ (hd : C → C) (m' : ℕ) (κ' : Fin m' → Q) (Ψ : Pos g σ V ι → ℝ≥0∞),
          avg (fun z : CoinTape (@Game.coin _ C iF iN k) s' =>
            Ψ (W (hd z.1) m' κ' (z.2 (hd z.1)) fun y => Function.update z.2 (hd z.1) y)) =
          avg fun z : CoinTape (@Game.coin _ C iF iN k) s' => avg fun a : CoinTape (k (hd z.1)) (s' (hd z.1)) =>
            Ψ (W (hd z.1) m' κ' a fun y => Function.update z.2 (hd z.1) y) := by
        intro hd m' κ' Ψ
        refine (avg_prod (C := C) (D := (c : C) → CoinTape (k c) (s' c)) _).trans
          (Eq.trans ?_ (avg_prod (C := C) (D := (c : C) → CoinTape (k c) (s' c)) _).symm)
        exact congrArg avg (funext fun z₁ =>
          avg_pi_at (T := fun c => CoinTape (k c) (s' c)) (hd z₁) fun a u => Ψ (W (hd z₁) m' κ' a u))
      have hstep : ∀ (hd : C → C) (m' : ℕ) (κ' : Fin m' → Q),
          avg (fun z : CoinTape (@Game.coin _ C iF iN k) s' =>
            G (W (hd z.1) m' κ' (z.2 (hd z.1)) fun y => Function.update z.2 (hd z.1) y)) =
          avg fun z : CoinTape (@Game.coin _ C iF iN k) s' =>
            Pos.resample G (W (hd z.1) m' κ' (z.2 (hd z.1)) fun y => Function.update z.2 (hd z.1) y) := by
        intro hd m' κ'
        rw [key hd m' κ' G, key hd m' κ' (Pos.resample G)]
        exact congrArg avg (funext fun z =>
          avg_walk_fill sel G (k (hd z.1)) (s' (hd z.1)) _ m' κ' fun y => back (hd z.1, Function.update z.2 (hd z.1) y))
      have e : ∀ z, fill sel (@Game.coin _ C iF iN k) s' Ks m κ z =
          fillAt (T := fun c => CoinTape (k c) (s' c))
            (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) (@sel C iF iN k) m κ z :=
        fun z => by rw [fill]; exact ite_eq_right hf
      simp only [e]
      cases m with
      | zero =>
        have e' : ∀ z, Pos.walk (@Game.coin _ C iF iN k) s' Ks (fillAt (T := fun c => CoinTape (k c) (s' c))
            (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) (@sel C iF iN k) 0 κ z) back =
            W (id z.1) 0 κ (z.2 (id z.1)) fun y => Function.update z.2 (id z.1) y := fun z => hw z.1 0 κ z
        exact (congrArg avg (funext fun z => congrArg G (e' z))).trans ((hstep id 0 κ).trans
          (congrArg avg (funext fun z => congrArg (Pos.resample G) (e' z))).symm)
      | succ m =>
        cases @sel C iF iN k with
        | none =>
          have e' : ∀ z, Pos.walk (@Game.coin _ C iF iN k) s' Ks (fillAt (T := fun c => CoinTape (k c) (s' c))
              (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) none (m + 1) κ z) back =
              W (id z.1) (m + 1) κ (z.2 (id z.1)) fun y => Function.update z.2 (id z.1) y :=
            fun z => hw z.1 (m + 1) κ z
          exact (congrArg avg (funext fun z => congrArg G (e' z))).trans ((hstep id (m + 1) κ).trans
            (congrArg avg (funext fun z => congrArg (Pos.resample G) (e' z))).symm)
        | some e =>
          have e' : ∀ z, Pos.walk (@Game.coin _ C iF iN k) s' Ks (fillAt (T := fun c => CoinTape (k c) (s' c))
              (fun c m κ y => fill sel (k c) (s' c) (fun i => (Ks i).atCoin c) m κ y) (some e) (m + 1) κ z) back =
              W ((fun _ => e (κ 0)) z.1) m (Fin.tail κ) (z.2 ((fun _ => e (κ 0)) z.1))
                fun y => Function.update z.2 ((fun _ => e (κ 0)) z.1) y :=
            fun z => hw (e (κ 0)) m (Fin.tail κ) z
          exact (congrArg avg (funext fun z => congrArg G (e' z))).trans
            ((hstep (fun _ => e (κ 0)) m (Fin.tail κ)).trans
              (congrArg avg (funext fun z => congrArg (Pos.resample G) (e' z))).symm)

/-- **`avg_walk_fill` beside a uniform point**: over a uniform `C` and a tape from the root, any reading `G` at the
point of the walk's stop on the tape filled with the point's coins `κ` is its average over the tapes from the stop.
Stated over an arbitrary game so that a concrete game's instance unifies term by term: solving `avg_prod`'s function at
a concrete game approximates `?f (c, d)` against the walk and evaluates the game. -/
theorem avg_prod_walk_fill {C : Type} [Fintype C] (sel : FillSel α Q) (G : C → Pos g σ V ι → ℝ≥0∞)
    (Ks : ι → DigestCut V g) (m : ℕ) (κ : C → Fin m → Q) :
    avg (fun ω : C × CoinTape g σ => G ω.1 (Pos.walk g σ Ks (fill sel g σ Ks m (κ ω.1) ω.2) id)) =
      avg fun ω : C × CoinTape g σ => Pos.resample (G ω.1) (Pos.walk g σ Ks (fill sel g σ Ks m (κ ω.1) ω.2) id) := by
  refine (avg_prod (C := C) (D := CoinTape g σ) _).trans
    (Eq.trans ?_ (avg_prod (C := C) (D := CoinTape g σ) _).symm)
  exact congrArg avg (funext fun c => avg_walk_fill sel (G c) g σ Ks m (κ c) id)

end Fill

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

/-- **An opening's coin node, described**: `g'` is the device's flip of stream `p.1`'s round `p.2` of the tree `T` among
the rounds the tree committed to there, on the frame `.opening i s n` answering `b`, with the releases' continuation
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

/-- The rounds an opening's node flips among. -/
abbrev Openable (o : OpeningAt g') : Type := TreeOpenable o.T.nonce o.T.spec o.T.sample o.p.1 o.p.2

instance instFintypeOpenable (o : OpeningAt g') : Fintype o.Openable := o.iF

instance instNonemptyOpenable (o : OpeningAt g') : Nonempty o.Openable := o.iN

/-- Two descriptions of one node flip among one type. -/
theorem openable_eq (o o' : OpeningAt g') : o.Openable = o'.Openable := by
  have h := o.eq.symm.trans o'.eq
  injection h

open Classical in
/-- **The rounds released as `bits`**: those whose first `n` coins are `bits` at every description of the node. -/
noncomputable def classOf (o : OpeningAt g') (bits : List Bool) : Finset o.Openable :=
  Finset.univ.filter fun d => ∀ o' : OpeningAt g', TreeDraw.opened (cast (openable_eq o o') d).1 o'.n = bits

end OpeningAt

open Classical in
/-- **The node's description**, where `g'` is an opening's coin node. -/
noncomputable def openingOf {β : Type} (g' : Game β) : Option (OpeningAt g') :=
  if h : Nonempty (OpeningAt g') then some (Classical.choice h) else none

/-- **The release `bits` written into an opening's node**: at an opening's coin node, the tape's round there replaced by
the round it picks among those released as `bits`; every other tape unchanged. -/
noncomputable def writeOpening {β : Type} {g' : Game β} (s' : Strategy g') (bits : List Bool) :
    CoinTape g' s' → CoinTape g' s' :=
  (openingOf g').elim id fun o => writeCoin o.eq (pickIn (o.classOf bits)) s'

/-- **The opening's gap** at the node `g'` and the release `bits` of a uniform `P`: the written round's shortfall below
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
/-- **The written release is `bits`**: at an opening's node whose rounds include one released as `bits`, the written
tape's round there is released as `bits` at every description of the node. -/
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

/-- **The coin map's start is within `ε₀` of the uniform tape**: over the widened coins, the period's tape the start
gives is, at every reading of it bounded by 1, within `ε₀` of the uniform tape. -/
def StartWithin (A : Live) (σ : Strategy (V.game θ F c)) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (ε₀ : ℝ≥0∞) : Prop :=
  ∀ H : CoinTape (V.game θ F c) σ → ℝ≥0∞, (∀ x, H x ≤ 1) →
    avg H ≤ avg (fun ω : (A.extend M.widening).L.Ω => H (M.start ω)) + ε₀

/-- **The stops' term**: over the widened coins, `γL` at the first stop and the rounds' error after it, over the
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

/-! ## The sample's coins in the fresh frames of the sample's family -/

/-- **A fresh frame's coin node, described**: `g'` is the device's flip of the coins of a fresh frame of the family
`label` answering `after`, with the releases' continuation after it (`relK`'s fresh branch). -/
structure FreshAt {β : Type} (label : String) (after : ByteArray) (g' : Game β) where
  k : ℕ
  fs : List (String × ℕ)
  ts : List OpenTree
  f : List Release → Game β
  iF : Fintype (Fin 512 → Bool)
  iN : Nonempty (Fin 512 → Bool)
  eq : g' = @Game.coin β (Fin 512 → Bool) iF iN fun x =>
    relK k fs ts fun log => f (⟨.fresh label, after, List.ofFn x⟩ :: log)

/-- A fresh frame's node flips 512 bits. -/
theorem FreshAt.coin_eq {β C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β} {label : String}
    {after : ByteArray} (o : FreshAt label after (@Game.coin β C iF iN k)) : C = (Fin 512 → Bool) := by
  have h := o.eq
  injection h

open Classical in
/-- **The fresh frames of the family `label`, as a fill writes them** (`fill`): the coin nodes described as fresh
frames of the family `label` answering `after`, each taking a written coin as its 512 bits. -/
noncomputable def freshSel (label : String) (after : ByteArray) : FillSel Exchange (Fin 512 → Bool) :=
  fun {C : Type} [iF : Fintype C] [iN : Nonempty C] (k : C → Game Exchange) =>
    if h : Nonempty (FreshAt label after (@Game.coin _ C iF iN k)) then
      some (Equiv.cast (FreshAt.coin_eq (Classical.choice h)).symm)
    else none

/-- The device's one stream of bytes, at its length. -/
def srcEquiv {L : Fin 1 → ℕ} {n : ℕ} (h : L 0 = n) :
    ((i : Fin 1) → Fin (L i) → Fin 256) ≃ (Fin 1 → Fin n → Fin 256) :=
  Equiv.piCongrRight fun i =>
    (finCongr (by rw [Subsingleton.elim i 0]; exact h)).arrowCongr (Equiv.refl (Fin 256))

/-- **The sample's bytes as fresh frames' coins**: the device's `m * 64` bytes the sample reads, as `m` fresh frames'
512 bits each (`coinBytesEquiv`). -/
noncomputable def srcCoins (A : Live) {m : ℕ} (hL : A.Lb 0 = m * 64) (ω : A.Ω) : Fin m → Fin 512 → Bool :=
  coinBytesEquiv m 64 (srcEquiv hL (A.src ω))

/-- **The sample's coins are uniform**, by A3 at the device's bytes (`Live.hA3`). -/
theorem avg_srcCoins (A : Live) {m : ℕ} (hL : A.Lb 0 = m * 64) (H : (Fin m → Fin 512 → Bool) → ℝ≥0∞) :
    avg (fun ω : A.Ω => H (srcCoins A hL ω)) = avg H :=
  avg_comp_of_prCoin (fun E => (A.hA3 fun s => E (coinBytesEquiv m 64 (srcEquiv hL s))).trans
    (FlockSoundness.ZK.prCoin_congr_equiv ((srcEquiv hL).trans (coinBytesEquiv m 64)) E)) H

/-- **The sample reads the bytes of its coins**: the device's bytes the sample reads are those `coinBytes` reads off
`srcCoins`, the premise `coupled_draw_coins_setsUp` takes. -/
theorem bytesOf_srcCoins (A : Live) {m : ℕ} (hL : A.Lb 0 = m * 64) (ω : A.Ω) :
    _root_.Properties.SampledProofs.bytesOf (A.src ω 0) =
      _root_.Properties.SampledProofs.bytesOf (coinBytes m 64 (srcCoins A hL ω) 0) := by
  have h : coinBytes m 64 (srcCoins A hL ω) = srcEquiv hL (A.src ω) := (coinBytesEquiv m 64).symm_apply_apply _
  have key : ∀ {L L' : ℕ} (h : L = L') (f : Fin L → Fin 256), _root_.Properties.SampledProofs.bytesOf f =
      _root_.Properties.SampledProofs.bytesOf fun j : Fin L' => f (Fin.cast h.symm j) := by
    intro L L' h f
    subst h
    rfl
  rw [h]
  exact key hL (A.src ω 0)

/-- **The start that writes the sample's coins into the fresh frames of the sample's family**: on the device's coins and
a tape of the period's game, the tape filled with the coins `κ` at the fresh frames of the family `label` answering
`after`, up to the first node a round's cut fixes (`fill`). -/
noncomputable def sampleStart {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {Ω : Type} {m : ℕ} (κ : Ω → Fin m → Fin 512 → Bool) (ω : Ω × CoinTape (V.game θ F c) σ) :
    CoinTape (V.game θ F c) σ :=
  fill (freshSel label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2

/-- **The sample-coupled coin map**: widened by a tape of the period's game, its start writes the sample's coins `κ`
into the fresh frames of the sample's family (`sampleStart`), and at each stop it writes the live coins into the stop's
opening (`openMap`). -/
noncomputable def sampleMap (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool) : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP :=
  openMap e (CoinTape (V.game θ F c) σ) (sampleStart V θ F c rd σ label after κ)

/-- The sample at widened coins is the sample at the device's coins. By `dsimp`, not `rfl`: elaborating the two samples
as definitionally equal reduces their `match` and evaluates `drawUnits`. -/
theorem sample_extend (A : Live) (S : Widening) (ω : (A.extend S).L.Ω) : (A.extend S).L.draw ω = A.L.draw ω.1 := by
  dsimp only [Live.L, Live.extend]

/-- The projection's first stop is the walk from the start's tape to the first stop. -/
theorem firstStop_eq (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (ω : (A.extend M.widening).L.Ω) :
    firstStop V θ F c A rd σ M ω = Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) (M.start ω) id :=
  rfl

/-- The sample-coupled start is the fill of the widening's tape with the coins `κ` at the device's coins. -/
theorem sampleMap_start_eq (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω) :
    (sampleMap e V θ F c A rd σ label after κ).start ω =
      fill (freshSel label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2 :=
  rfl

/-- The sample-coupled map's first stop is the walk to it from the filled tape. -/
theorem firstStop_sampleMap (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω) :
    firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω =
      Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd)
        (fill (freshSel label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2) id :=
  (firstStop_eq V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω).trans
    (congrArg (fun t => Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd) t id)
      (sampleMap_start_eq e V θ F c A rd σ label after κ ω))

/-- **The sample-coupled start leaves the tape from the first stop fresh**: it writes nothing from the first stop on,
and the sample reads only the device's coins (`avg_prod_walk_fill`). -/
theorem sampleMap_fresh (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool) :
    FirstStopFresh V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) := fun G => by
  have h₁ := congrArg avg (funext fun ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω =>
    congrArg₂ G (sample_extend A _ ω) (firstStop_sampleMap e V θ F c A rd σ label after κ ω))
  have h₂ := congrArg avg (funext fun ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω =>
    congrArg₂ (fun s q => Pos.resample (G s) q) (sample_extend A _ ω)
      (firstStop_sampleMap e V θ F c A rd σ label after κ ω))
  refine h₁.trans (Eq.trans ?_ h₂.symm)
  generalize A.L.draw = dr
  exact avg_prod_walk_fill (C := A.Ω) (freshSel label after) (fun ω₁ : A.Ω => G (dr ω₁)) (roundCuts V θ F c rd) m κ

/-- **The sample-coupled start is the uniform tape** wherever the sample's coins are uniform (`avg_srcCoins`). -/
theorem sampleMap_start (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    StartWithin V θ F c A σ (sampleMap e V θ F c A rd σ label after κ) 0 := fun H _ => by
  rw [add_zero]
  refine le_of_eq (Eq.trans ?_ (avg_prod (C := A.Ω) (D := CoinTape (V.game θ F c) σ) _).symm)
  exact (avg_fill (freshSel label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H).symm.trans
    (hκ fun κ' => avg fun t => H (fill (freshSel label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).symm

/-- **The sample-coupled coin map's premise within the openings' gaps**: the joint law's tape is within the openings'
gaps over the stops of the uniform tape, wherever the sample's coins are uniform. -/
theorem sampleMap_couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (label : String) (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    CouplesWithin V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
      (couplesErr V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
        (fun q _ s => openGap q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
        (fun q => openGap q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool))) := by
  have h := openMap_couples e V θ F c A rd σ D (CoinTape (V.game θ F c) σ) (sampleStart V θ F c rd σ label after κ)
    (sampleMap_fresh e V θ F c A rd σ label after κ) (sampleMap_start e V θ F c A rd σ label after κ hκ)
  rwa [zero_add] at h

omit θ F c in
/-- **The sample-coupled coin maps** at the readings `rd`: at each session, commitment and strategy, `sampleMap` at the
family `label θ F c` answering `after θ F c` and the session's coins `κ A`. -/
noncomputable def sampleMaps (rd : Readings) (label : ByteArray → Array ByteArray → ByteArray → String)
    (after : ByteArray → Array ByteArray → ByteArray → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP :=
  sampleMap e V θ F c A (rd A θ F c) σ (label θ F c) (after θ F c) (κ A)

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

omit θ F c in
/-- **The coupling at the projection with the sample-coupled coin maps**: `couplingTarget_withProject` at
`sampleMaps`, whose coin map's premise holds within the openings' gaps over the stops wherever the session's coins are
uniform (`sampleMap_couples`), at each commitment and strategy where those gaps are at most the hiding term. -/
theorem couplingTarget_withProject_of_sample (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : ByteArray → Array ByteArray → ByteArray → String)
    (after : ByteArray → Array ByteArray → ByteArray → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ A ω)) = avg H)
    (ha : LoggedBound V k (K.withProject V rd D (sampleMaps e V rd label after κ)) P θ F)
    (hε : ∀ c σ, couplesErr V θ F c A (rd A θ F c) σ (D A θ F c σ) (sampleMaps e V rd label after κ A θ F c σ)
      (fun q _ s => openGap q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤ A.hidingTerm)
    (hD : ∀ c σ, Decodes V θ F c k P A (rd A θ F c) σ (D A θ F c σ) (sampleMaps e V rd label after κ A θ F c σ)) :
    CouplingTarget V k ((K.withProject V rd D (sampleMaps e V rd label after κ)).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D (sampleMaps e V rd label after κ) A P θ F ha
    (fun c σ => couples_of_within V θ F c (sampleMap_couples e V θ F c A (rd A θ F c) σ (D A θ F c σ) (label θ F c)
      (after θ F c) (κ A) hκ) (hε c σ)) hD

end Verity.Sampled
