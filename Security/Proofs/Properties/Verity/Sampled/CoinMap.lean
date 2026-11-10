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
  `avg_roundsOff_le`, `resample_sessionOff_le`, `couples_of_close`): the joint law's tape off held-off events at the
  stops (`LinkHold`, `RoundHold`) is within the start's term and the stops' (`couplesErr`) of the uniform tape at every
  event (`CouplesWithin`). The stops' term carries each held-off event's probability beside the stop's own
  (`avg_le_off`), so `Couples` holds wherever their sum is at most the hiding term (`couples_of_within`,
  `couplingTarget_withProject_of_close`).
* **A play's outputs** (`exists_playReaches`, `PlayReaches.of_bind`, `playReaches_relK`, `playReaches_closeThen`,
  `log_of_finish`): every strategy reaches an output, and an output of the device's releases is one of their
  continuation at some log. A close records the device's log (`RecordsLog`) where every output of its play at a log
  reads that log; the period's close does (`finish_recordsLog`). Releases continuing as one game to such a close, after
  two logs and two releases, continue after one log and one release (`relK_log_eq`).
* **An opening's node** (`OpeningAt`, `openingOf`, `writeOpening`, `openGap`, `EmptyAt`, `openEmpty`): a node described
  as the releases' opening branch (`relK`) on the frame `.opening i s n` the prover sent, continuing to the play's close
  `E` after the releases before the frame and the frame's. Where `E` records its log, every description has one log
  before the frame and one release (`OpeningAt.release_eq`), so one count `n`, the sent frame's (`OpeningAt.n_eq`).
  The node's class at `bits` is that of its canonical description (`openingOf`, `OpeningAt.classOf`): the rounds every
  description releases as `bits` (`OpeningAt.mem_classOf_iff`), nonempty wherever one releases them
  (`OpeningAt.classOf_nonempty`), empty exactly where no round of the fiber does (`OpeningAt.classOf_eq_empty_iff`).
  There the tape's round is replaced by one of the class, released as the written bits at every description
  (`opened_writeOpening`) wherever the class is nonempty, that is off the empty class (`EmptyAt`,
  `opened_writeOpening_off`), within the opening's gap (`avg_le_writeOpening`).
* **The coin map** (`roundBits`, `linkBits`, `openMap`, `linkEmpty`, `roundEmpty`): each round's live coins and the link
  points, as bits under an encoding `e` of the field, written into the stop's opening in a play that ends in `E`. Held
  off the empty classes, its premises hold within the openings' gaps and empty classes (`openMap_roundClose`,
  `openMap_linkClose`, `openMap_couples`), at any widening and start that leave the tape from the first stop fresh
  (`FirstStopFresh`) and are within `ε₀` of the uniform tape (`StartWithin`).
* **The sample-coupled coin map** (`FreshAt`, `freshSel`, `sampleSel`, `srcCoins`, `sampleStart`, `sampleMap`,
  `sampleMaps`): the coin map into the openings of the strategy's play, which ends in the close after its request.
  Widened by a tape of the period's game, its start fills the fresh frames of the sample's family in the
  strategy's play, after its request, with the bytes the sample reads, as coins (`srcCoins`, whose bytes are the
  sample's: `bytesOf_srcCoins`). A fresh frame is pinned to its place in the play: its log ends in the close after the
  strategy's request, with the frame's release after the releases before it. It leaves the tape from the
  first stop fresh (`sampleMap_fresh`, by `avg_prod_walk_fill`) and is the uniform tape wherever the sample's coins are
  uniform (`sampleMap_start`, `avg_srcCoins`), so its premise holds within the openings' gaps and empty classes alone
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
/-- **A written coin's shortfall below uniform**: the total variation between the uniform coin's law and the sampling design of
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

/-! ## A play's outputs -/

/-- **An output a play reaches, at an equal game.** -/
theorem playReaches_of_eq {g g' : Game α} (e : g = g') (s : Strategy g) {o : α} (h : PlayReaches g s o) :
    ∃ s' : Strategy g', PlayReaches g' s' o := by
  subst e
  exact ⟨s, h⟩

/-- **Every strategy's plays reach an output.** -/
theorem exists_playReaches : ∀ (g : Game α) (s : Strategy g), ∃ o, PlayReaches g s o
  | .ret a, _ => ⟨a, rfl⟩
  | .send _ k, s => exists_playReaches (k s.1) s.2
  | @Game.coin _ _ _ iN k, s => by
    obtain ⟨c⟩ := iN
    obtain ⟨o, h⟩ := exists_playReaches (k c) (s c)
    exact ⟨o, c, h⟩

/-- **An output a play of `g.bind f` reaches is one a play of some `f b` reaches.** -/
theorem PlayReaches.of_bind {β : Type} {f : β → Game α} {o : α} :
    ∀ (g : Game β) (s : Strategy (g.bind f)), PlayReaches (g.bind f) s o →
      ∃ b, ∃ s' : Strategy (f b), PlayReaches (f b) s' o
  | .ret b, s, h => ⟨b, s, h⟩
  | .send _ k, s, h => PlayReaches.of_bind (k s.1) s.2 h
  | @Game.coin _ _ _ _ k, s, h => by
    obtain ⟨c, hc⟩ := h
    exact PlayReaches.of_bind (k c) (s c) hc

/-- **Equal functions on equal types agree at a point and its cast.** -/
theorem apply_cast_of_heq {C C' : Type} {B : Sort*} {k : C → B} {k' : C' → B} (hC : C = C') (hk : HEq k k') (c : C) :
    k c = k' (cast hC c) := by
  subst hC
  rw [eq_of_heq hk]
  rfl

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

open Classical in
/-- **A mean with an event's points counted out**: a mean of values at most 1 is at most its mean with the points `H`
holds of counted out, plus `H`'s probability. -/
theorem avg_le_off {C : Type} [Fintype C] (H : C → Prop) (f : C → ℝ≥0∞) (hf : ∀ c, f c ≤ 1) :
    avg f ≤ avg (fun c => if H c then 0 else f c) + prCoin H :=
  (avg_mono fun c => show f c ≤ (if H c then 0 else f c) + (if H c then 1 else 0) by
    by_cases h : H c
    · rw [ite_eq_left h, ite_eq_left h, zero_add]
      exact hf c
    · rw [ite_eq_right h, ite_eq_right h, add_zero]).trans_eq (avg_add _ _)

section Session

variable {g : Game Exchange} {σ : Strategy g} {Ω Reg : Type} {R mPts : ℕ} (D : Decoders R Reg)
  (M : CoinMap g σ Ω R mPts) (J : ℕ) (γ : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞) (B : RoundHold)

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

variable {D M J γ B} in
/-- **The rounds' share off `B` is within the rounds' error of uniform from where it starts**: wherever the coin map
writes each round's coins in within `γ` of the tape's own, an event of the tape has, over a uniform tape from the
position `p`, at most its share off `B` (`roundsOff`) plus the rounds' error at `γ` and `B`'s probability at each stop,
as its share of the tapes through `p`. -/
theorem avg_roundsOff_le (hR : M.RoundClose γ) (Q : CoinTape g σ → Prop) :
    ∀ (N : ℕ) (p : Pos g σ Claims.Msg (Fin R)),
      avg (fun z : CoinTape p.g' p.s' => tapeInd Q (p.withTape z)) ≤
        avg (fun z : CoinTape p.g' p.s' => roundsOff D M J B Q N (p.withTape z)) +
        avg fun z : CoinTape p.g' p.s' =>
          roundsErr D J (fun q j s => γ q j s + prCoin (B q.g' j s)) N (p.withTape z)
  | 0, _ => le_self_add
  | N + 1, p => by
    let γB : Pos g σ Claims.Msg (Fin R) → (j : ℕ) → SubmitJ j → ℝ≥0∞ := fun q j s => γ q j s + prCoin (B q.g' j s)
    let Ψ : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q =>
      stopOff M J B Q (roundsOff D M J B Q N) q (D.sub J q.modes)
    let E : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞ := fun q => stopErr J γB (roundsErr D J γB N) q (D.sub J q.modes)
    have hstop : ∀ (q : Pos g σ Claims.Msg (Fin R)) (o : Option (SubmitJ J)),
        avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z)) ≤
          avg (fun z : CoinTape q.g' q.s' => stopOff M J B Q (roundsOff D M J B Q N) (q.withTape z) o) +
          avg fun z : CoinTape q.g' q.s' => stopErr J γB (roundsErr D J γB N) (q.withTape z) o := fun q o => by
      cases o with
      | none => exact le_self_add
      | some s =>
        let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => roundsOff D M J B Q N (q.consume.withTape z)
        have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 =>
              G (M.round (q.withTape z) J s c z)) + avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s :=
          calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
            _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun c : Fin s.n → 𝔽 => avg fun z : CoinTape q.g' q.s' =>
                  G (M.round (q.withTape z) J s c z)) + γ (q.withTape z₀) J s :=
                avg_mono fun z₀ => hR (q.withTape z₀) J s G fun _ => roundsOff_le_one D M J B Q N _
            _ = _ := by
                rw [avg_add, avg_const, avg_comm]
        have hB : (avg fun z : CoinTape q.g' q.s' => avg fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z)) ≤
            (avg fun z : CoinTape q.g' q.s' => stopOff M J B Q (roundsOff D M J B Q N) (q.withTape z) (some s)) +
              prCoin (B q.g' J s) :=
          (avg_mono fun z : CoinTape q.g' q.s' => avg_le_off (B q.g' J s)
            (fun c : Fin s.n → 𝔽 => G (M.round (q.withTape z) J s c z))
              fun _ => roundsOff_le_one D M J B Q N _).trans_eq (by rw [avg_add, avg_const]; rfl)
        have h2 : avg (fun z : CoinTape q.g' q.s' => stopErr J γB (roundsErr D J γB N) (q.withTape z) (some s)) =
            (avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) + prCoin (B q.g' J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErr D J γB N (q.consume.withTape z) := by
          show avg (fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s + prCoin (B q.g' J s) +
            Pos.resample (roundsErr D J γB N) q.consume) = _
          rw [avg_add, avg_add, avg_const, avg_const]
          rfl
        calc avg (fun z : CoinTape q.g' q.s' => tapeInd Q (q.withTape z))
            ≤ avg G + avg fun z : CoinTape q.g' q.s' => roundsErr D J γB N (q.consume.withTape z) :=
              avg_roundsOff_le hR Q N q.consume
          _ ≤ ((avg fun z : CoinTape q.g' q.s' => stopOff M J B Q (roundsOff D M J B Q N) (q.withTape z) (some s)) +
                prCoin (B q.g' J s) + avg fun z : CoinTape q.g' q.s' => γ (q.withTape z) J s) +
              avg fun z : CoinTape q.g' q.s' => roundsErr D J γB N (q.consume.withTape z) :=
              add_le_add (hγ.trans (add_le_add hB le_rfl)) le_rfl
          _ = _ := by
              rw [h2]
              ring
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

open Classical in
variable {D M J γ B} in
/-- **The session's share off `BL` and `B` is within the link's and the rounds' error of uniform from its first
stop**: wherever the coin map writes the link points in within `γL` of the tape's own and each round's coins within
`γ`, an event of the tape has, over a uniform tape from the stop `q`, at most its share off `BL` and `B` from there,
plus `γL` and `BL`'s probability at the stop and the rounds' error at `γ` and `B`'s probability after it, as its share
of the tapes through `q`. -/
theorem resample_sessionOff_le {γL : Pos g σ Claims.Msg (Fin R) → ℝ≥0∞} (hL : M.LinkClose γL) (hR : M.RoundClose γ)
    (BL : LinkHold mPts) (Q : CoinTape g σ → Prop) (N : ℕ) (q : Pos g σ Claims.Msg (Fin R)) :
    Pos.resample (tapeInd Q) q ≤
      Pos.resample (fun q' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        if BL q'.g' pts then 0 else roundsOff D M J B Q N (q'.rewrite (M.link q' pts)).consume) q +
      (Pos.resample (fun q' => γL q' + prCoin (BL q'.g')) q +
        Pos.resample (roundsErr D J (fun q j s => γ q j s + prCoin (B q.g' j s)) N) q.consume) := by
  let G : CoinTape q.g' q.s' → ℝ≥0∞ := fun z => roundsOff D M J B Q N (q.consume.withTape z)
  have hγ : avg G ≤ avg (fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        G (M.link (q.withTape z) pts z)) + avg fun z : CoinTape q.g' q.s' => γL (q.withTape z) :=
    calc avg G = avg fun _ : CoinTape q.g' q.s' => avg G := (avg_const _).symm
      _ ≤ avg fun z₀ : CoinTape q.g' q.s' => avg (fun pts : Fin 2 → Fin mPts → 𝔽 =>
            avg fun z : CoinTape q.g' q.s' => G (M.link (q.withTape z) pts z)) + γL (q.withTape z₀) :=
          avg_mono fun z₀ => hL (q.withTape z₀) G fun _ => roundsOff_le_one D M J B Q N _
      _ = _ := by rw [avg_add, avg_const, avg_comm]
  have hB : (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        G (M.link (q.withTape z) pts z)) ≤
      (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
        if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) + prCoin (BL q.g') :=
    (avg_mono fun z : CoinTape q.g' q.s' => avg_le_off (BL q.g')
      (fun pts : Fin 2 → Fin mPts → 𝔽 => G (M.link (q.withTape z) pts z))
        fun _ => roundsOff_le_one D M J B Q N _).trans_eq (by rw [avg_add, avg_const])
  calc Pos.resample (tapeInd Q) q
      ≤ avg G + Pos.resample (roundsErr D J (fun q j s => γ q j s + prCoin (B q.g' j s)) N) q.consume :=
        avg_roundsOff_le hR Q N q.consume
    _ ≤ ((avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
            if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) + prCoin (BL q.g') +
          avg fun z : CoinTape q.g' q.s' => γL (q.withTape z)) +
        Pos.resample (roundsErr D J (fun q j s => γ q j s + prCoin (B q.g' j s)) N) q.consume :=
        add_le_add (hγ.trans (add_le_add hB le_rfl)) le_rfl
    _ = _ := by
        show _ = (avg fun z : CoinTape q.g' q.s' => avg fun pts : Fin 2 → Fin mPts → 𝔽 =>
              if BL q.g' pts then 0 else G (M.link (q.withTape z) pts z)) +
            ((avg fun z : CoinTape q.g' q.s' => γL (q.withTape z) + prCoin (BL q.g')) +
              Pos.resample (roundsErr D J (fun q j s => γ q j s + prCoin (B q.g' j s)) N) q.consume)
        rw [avg_add, avg_const]
        ring

end Session

/-! ## An opening's node -/

/-- **An output a play of the device's releases reaches is one a play of the continuation at some log reaches.** -/
theorem playReaches_relK {β : Type} (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree)
    (f : List Release → Game β) (s : Strategy (relK k fs ts f)) {o : β} (h : PlayReaches (relK k fs ts f) s o) :
    ∃ L, ∃ s' : Strategy (f L), PlayReaches (f L) s' o := by
  obtain ⟨s', h'⟩ := playReaches_of_eq (releases_bind k fs ts f).symm s h
  exact PlayReaches.of_bind _ s' h'

/-- **The prover that closes the device at once reaches only the continuation at the empty log** (`closeThen`). -/
theorem playReaches_closeThen {β : Type} (f : List Release → Game β) (t : Strategy (f [])) {o : β} :
    ∀ (k : ℕ) (fs : List (String × ℕ)) (ts : List OpenTree),
      PlayReaches ((releases k fs ts).bind f) (closeThen f t k fs ts) o → PlayReaches (f []) t o
  | 0, _, _, h => h
  | _ + 1, _, _, h => h

/-- **The close after a log ends the play at that log.** -/
theorem log_of_finish {sg : Signer} {θ : ByteArray} {F : Array ByteArray} {c rq : ByteArray} {log : List Release}
    {s : Strategy (finish sg θ F c rq log)} {ex : Exchange} (h : PlayReaches (finish sg θ F c rq log) s ex) :
    ex.log = log := by
  obtain ⟨ρ, he⟩ : ∃ ρ : Fin sg.draws → Bool, ex = ⟨θ, F, c, rq, log, s.1, List.ofFn ρ,
      sg.signatures rq log s.1 (List.ofFn ρ), (s.2 ρ).1.1, (s.2 ρ).1.2⟩ := h
  rw [he]

/-- **A close that records the device's log**: every output a play of `E log` reaches reads, under `lg`, the log
`log`, and `E log` has a strategy. -/
structure RecordsLog {β : Type} (lg : β → List Release) (E : List Release → Game β) : Prop where
  reads : ∀ log (s : Strategy (E log)) o, PlayReaches (E log) s o → lg o = log
  plays : ∀ log, Nonempty (Strategy (E log))

/-- **The period's close records the device's log** (`log_of_finish`). -/
theorem finish_recordsLog (sg : Signer) (θ : ByteArray) (F : Array ByteArray) (c rq : ByteArray) :
    RecordsLog Exchange.log (finish sg θ F c rq) where
  reads _ _ _ h := log_of_finish h
  plays _ := ⟨⟨ByteArray.empty, fun _ => ⟨(ByteArray.empty, #[]), PUnit.unit⟩⟩⟩

/-- **The log the device's releases continue to, at a close that records its log**: where the releases from two states,
continuing to the close after `pre`, the release `r` and theirs, and to the close after `pre'`, `r'` and theirs, are
one game, the prover that closes the first at once ends it at `pre` and `r`, a log the second reaches. -/
theorem relK_logs {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    {k k' : ℕ} {fs fs' : List (String × ℕ)} {ts ts' : List OpenTree} {pre pre' : List Release} {r r' : Release}
    (h : relK k fs ts (fun log => E (pre ++ r :: log)) = relK k' fs' ts' (fun log => E (pre' ++ r' :: log))) :
    ∃ L, pre ++ [r] = pre' ++ r' :: L := by
  obtain ⟨t⟩ := hE.plays (pre ++ [r])
  obtain ⟨o, ho⟩ := exists_playReaches _ (closeThen (fun log => E (pre ++ r :: log)) t k fs ts)
  have h₁ : lg o = pre ++ [r] :=
    hE.reads _ t o (playReaches_closeThen (fun log => E (pre ++ r :: log)) t k fs ts ho)
  obtain ⟨s', ho'⟩ := playReaches_of_eq ((releases_bind k fs ts _).trans h) _ ho
  obtain ⟨L, s'', ho''⟩ := playReaches_relK k' fs' ts' _ s' ho'
  exact ⟨L, h₁.symm.trans (hE.reads _ s'' o ho'')⟩

/-- **One log before the release, and one release**, where the device's releases continue as one game to a close that
records its log (`relK_logs` both ways). -/
theorem relK_log_eq {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    {k k' : ℕ} {fs fs' : List (String × ℕ)} {ts ts' : List OpenTree} {pre pre' : List Release} {r r' : Release}
    (h : relK k fs ts (fun log => E (pre ++ r :: log)) = relK k' fs' ts' (fun log => E (pre' ++ r' :: log))) :
    pre = pre' ∧ r = r' := by
  obtain ⟨L, hL⟩ := relK_logs hE h
  obtain ⟨L', hL'⟩ := relK_logs hE h.symm
  have e₁ := congrArg List.length hL
  have e₂ := congrArg List.length hL'
  simp only [List.length_append, List.length_cons, List.length_nil] at e₁ e₂
  obtain rfl : L = [] := List.eq_nil_of_length_eq_zero (by omega)
  obtain ⟨hp, hr⟩ := List.append_inj hL (by omega)
  exact ⟨hp, (List.cons.inj hr).1⟩

/-- **An opening's coin node, described in the play**: `g'` is the device's flip of stream `p.1`'s round `p.2` of the
tree `T` among the rounds the tree committed to there, on the frame `.opening i s n` the prover sent answering `b`,
with the releases' continuation after it (`relK`'s opening branch), ending in `E` at the device's log: the releases
`pre` before the frame, the frame's, then the rest. -/
structure OpeningAt {β : Type} (E : List Release → Game β) (g' : Game β) where
  T : OpenTree
  p : Fin T.spec.streams.size × Fin (2 ^ T.spec.roundsLog)
  i : ℕ
  s : ℕ
  n : ℕ
  b : ByteArray
  k : ℕ
  fs : List (String × ℕ)
  ts : List OpenTree
  pre : List Release
  iF : Fintype (TreeOpenable T.nonce T.spec T.sample p.1 p.2)
  iN : Nonempty (TreeOpenable T.nonce T.spec T.sample p.1 p.2)
  eq : g' = @Game.coin β (TreeOpenable T.nonce T.spec T.sample p.1 p.2) iF iN fun d =>
    relK k fs (ts.set i (T.bump p.1)) fun log => E (pre ++ ⟨.opening i s n, b, TreeDraw.opened d.1 n⟩ :: log)

namespace OpeningAt

variable {β : Type} {E : List Release → Game β} {g' : Game β}

/-- The rounds an opening's node flips among. -/
abbrev Openable (o : OpeningAt E g') : Type := TreeOpenable o.T.nonce o.T.spec o.T.sample o.p.1 o.p.2

instance instFintypeOpenable (o : OpeningAt E g') : Fintype o.Openable := o.iF

instance instNonemptyOpenable (o : OpeningAt E g') : Nonempty o.Openable := o.iN

/-- Two descriptions of one node flip among one type. -/
theorem openable_eq (o o' : OpeningAt E g') : o.Openable = o'.Openable := by
  have h := o.eq.symm.trans o'.eq
  injection h

/-- **Two descriptions of one node continue alike**, from a round and its cast. -/
theorem cont_eq (o o' : OpeningAt E g') (d : o.Openable) :
    relK o.k o.fs (o.ts.set o.i (o.T.bump o.p.1))
        (fun log => E (o.pre ++ ⟨.opening o.i o.s o.n, o.b, TreeDraw.opened d.1 o.n⟩ :: log)) =
      relK o'.k o'.fs (o'.ts.set o'.i (o'.T.bump o'.p.1)) (fun log =>
        E (o'.pre ++ ⟨.opening o'.i o'.s o'.n, o'.b, TreeDraw.opened (cast (openable_eq o o') d).1 o'.n⟩ :: log)) := by
  have h := o.eq.symm.trans o'.eq
  injection h with hC _ hk
  exact apply_cast_of_heq hC hk d

/-- **One log before the frame, and one release, at every description** of a node whose play ends in a close that
records its log (`relK_log_eq`): the frame the prover sent, the bytes it answers and the round's released coins. -/
theorem release_eq {lg : β → List Release} (hE : RecordsLog lg E) (o o' : OpeningAt E g') (d : o.Openable) :
    o.pre = o'.pre ∧ (⟨.opening o.i o.s o.n, o.b, TreeDraw.opened d.1 o.n⟩ : Release) =
      ⟨.opening o'.i o'.s o'.n, o'.b, TreeDraw.opened (cast (openable_eq o o') d).1 o'.n⟩ :=
  relK_log_eq hE (o.cont_eq o' d)

/-- **One count at every description**: the `n` of the frame the prover sent (`release_eq`). -/
theorem n_eq {lg : β → List Release} (hE : RecordsLog lg E) (o o' : OpeningAt E g') : o.n = o'.n := by
  obtain ⟨d⟩ := o.iN
  exact (Verity.Frame.opening.inj (congrArg Release.frame (o.release_eq hE o' d).2)).2.2

/-- **One release of each round at every description** (`release_eq`). -/
theorem opened_eq {lg : β → List Release} (hE : RecordsLog lg E) (o o' : OpeningAt E g') (d : o.Openable) :
    TreeDraw.opened (cast (openable_eq o o') d).1 o'.n = TreeDraw.opened d.1 o.n :=
  (congrArg Release.coins (o.release_eq hE o' d).2).symm

/-- **The rounds a description releases as `bits`**: those whose first `n` coins are `bits`. -/
def classOf (o : OpeningAt E g') (bits : List Bool) : Finset o.Openable :=
  Finset.univ.filter fun d => TreeDraw.opened d.1 o.n = bits

/-- **A node's class is its canonical description's**: at a node whose play ends in a close that records its log, a
round is in the class of a description `o` exactly where every other description `o'` releases it as `bits`. -/
theorem mem_classOf_iff {lg : β → List Release} (hE : RecordsLog lg E) (o o' : OpeningAt E g') (bits : List Bool)
    (d : o.Openable) : d ∈ o.classOf bits ↔ TreeDraw.opened (cast (openable_eq o o') d).1 o'.n = bits := by
  rw [classOf, Finset.mem_filter, opened_eq hE o o' d]
  exact and_iff_right (Finset.mem_univ d)

/-- **The class is nonempty where the sample is released there**: a round some description of the node releases as
`bits` is in the class of each. -/
theorem classOf_nonempty {lg : β → List Release} (hE : RecordsLog lg E) (o o' : OpeningAt E g') {bits : List Bool}
    (d' : o'.Openable) (h : TreeDraw.opened d'.1 o'.n = bits) : (o.classOf bits).Nonempty :=
  ⟨cast (openable_eq o' o) d', (o.mem_classOf_iff hE o' bits _).2 (by rw [cast_cast, cast_eq]; exact h)⟩

/-- **The class is empty exactly where no round releases `bits`** at the description's count. -/
theorem classOf_eq_empty_iff (o : OpeningAt E g') (bits : List Bool) :
    o.classOf bits = ∅ ↔ ∀ d : o.Openable, TreeDraw.opened d.1 o.n ≠ bits := by
  rw [classOf, Finset.filter_eq_empty_iff]
  exact ⟨fun h d => h (Finset.mem_univ d), fun h d _ => h d⟩

end OpeningAt

open Classical in
/-- **The node's canonical description**, where `g'` is an opening's coin node in a play that ends in `E`: one chosen
among its descriptions. Where `E` records its log, every description has its log before the frame, its release and its
count (`OpeningAt.release_eq`, `OpeningAt.n_eq`): the `n` of the frame the prover sent. -/
noncomputable def openingOf {β : Type} (E : List Release → Game β) (g' : Game β) : Option (OpeningAt E g') :=
  if h : Nonempty (OpeningAt E g') then some (Classical.choice h) else none

/-- **A node with a description is one `openingOf` describes.** -/
theorem openingOf_some {β : Type} {E : List Release → Game β} {g' : Game β} (o' : OpeningAt E g') :
    ∃ o, openingOf E g' = some o :=
  ⟨Classical.choice ⟨o'⟩, by
    unfold openingOf
    exact dite_eq_left ⟨o'⟩⟩

/-- **The release `bits` written into an opening's node**: at an opening's coin node in a play that ends in `E`, the
tape's round there replaced by the round it picks among those the node's canonical description releases as `bits`;
every other tape unchanged. -/
noncomputable def writeOpening {β : Type} (E : List Release → Game β) {g' : Game β} (s' : Strategy g')
    (bits : List Bool) : CoinTape g' s' → CoinTape g' s' :=
  (openingOf E g').elim id fun o => writeCoin o.eq (pickIn (o.classOf bits)) s'

/-- **The opening's gap** at the node `g'` in a play that ends in `E` and the release `bits` of a uniform `P`: the
written round's shortfall below uniform at an opening's coin node; none at every other node. -/
noncomputable def openGap {β P : Type} [Fintype P] (E : List Release → Game β) (g' : Game β)
    (bits : P → List Bool) : ℝ≥0∞ :=
  (openingOf E g').elim 0 fun o => shortfall fun c => pickIn (o.classOf (bits c))

/-- **A release written into an opening's node is within the opening's gap of the tape's own**: a reading bounded by 1
of the uniform tape from the node is at most its mean with the release `bits c` of a uniform `c` written in, plus the
opening's gap. -/
theorem avg_le_writeOpening {β P : Type} [Fintype P] [Nonempty P] (E : List Release → Game β) {g' : Game β}
    (bits : P → List Bool) (s' : Strategy g') (G : CoinTape g' s' → ℝ≥0∞) (hG : ∀ z, G z ≤ 1) :
    avg G ≤ avg (fun c : P => avg fun z : CoinTape g' s' => G (writeOpening E s' (bits c) z)) + openGap E g' bits := by
  unfold writeOpening openGap
  cases openingOf E g' with
  | none =>
    refine le_of_eq ?_
    show avg G = avg (fun _ : P => avg fun z : CoinTape g' s' => G z) + 0
    rw [avg_const, add_zero]
  | some o => exact avg_le_writeCoin o.eq (fun c => pickIn (o.classOf (bits c))) s' G hG

/-- **The written release is `bits` at every description**: at an opening's node whose play ends in a close that records
its log, where the canonical description's class at `bits` is nonempty, the written tape's round there is released as
`bits` at every description of the node (`OpeningAt.mem_classOf_iff`). -/
theorem opened_writeOpening {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    {g' : Game β} (s' : Strategy g') (bits : List Bool) (z : CoinTape g' s') (o : OpeningAt E g')
    (ho : openingOf E g' = some o) (h : (o.classOf bits).Nonempty) (o' : OpeningAt E g') :
    TreeDraw.opened (headCoin o'.eq s' (writeOpening E s' bits z)).1 o'.n = bits := by
  have hw : writeOpening E s' bits z = writeCoin o.eq (pickIn (o.classOf bits)) s' z := by
    unfold writeOpening
    rw [ho]
    rfl
  rw [hw, headCoin_congr o.eq o'.eq (OpeningAt.openable_eq o o') s', headCoin_writeCoin]
  exact (o.mem_classOf_iff hE o' bits _).1 (pickIn_mem h (headCoin o.eq s' z))

/-- **No round of an opening's node is released as `bits`**: `g'` is an opening's coin node in a play that ends in
`E` and none of the rounds its canonical description flips among is released as `bits`. Only there does the written
release miss `bits` (`opened_writeOpening`). -/
def EmptyAt {β : Type} (E : List Release → Game β) (g' : Game β) (bits : List Bool) : Prop :=
  ∃ o, openingOf E g' = some o ∧ ¬ (o.classOf bits).Nonempty

/-- **The opening's empty class** at the node `g'` in a play that ends in `E` and the release `bits` of a uniform `P`:
the probability that no round of the node is released as `bits` of it. -/
noncomputable def openEmpty {β P : Type} [Fintype P] (E : List Release → Game β) (g' : Game β)
    (bits : P → List Bool) : ℝ≥0∞ :=
  prCoin fun c => EmptyAt E g' (bits c)

/-- **The written release is `bits` off the empty class** (`opened_writeOpening`). -/
theorem opened_writeOpening_off {β : Type} {lg : β → List Release} {E : List Release → Game β} (hE : RecordsLog lg E)
    {g' : Game β} (s' : Strategy g') (bits : List Bool) (z : CoinTape g' s') (h : ¬ EmptyAt E g' bits)
    (o : OpeningAt E g') (ho : openingOf E g' = some o) (o' : OpeningAt E g') :
    TreeDraw.opened (headCoin o'.eq s' (writeOpening E s' bits z)).1 o'.n = bits :=
  opened_writeOpening hE s' bits z o ho (not_not.1 fun hn => h ⟨o, ho, hn⟩) o'

/-! ## The coin map -/

variable (e : 𝔽 ≃ (Fin 128 → Bool))

/-- **A round's live coins as bits**: each challenge's 128 bits under `e`, in order. -/
def roundBits {m : ℕ} (c : Fin m → 𝔽) : List Bool := (List.ofFn fun i => List.ofFn (e (c i))).flatten

/-- **The live link points as bits**: the first point's, then the second's. -/
def linkBits {m : ℕ} (pts : Fin 2 → Fin m → 𝔽) : List Bool := roundBits e (pts 0) ++ roundBits e (pts 1)

/-- **The link's empty class, the event the joint law is held off at the first stop**: no round of the stop's opening,
in a play that ends in `E`, is released as the link points' bits. -/
def linkEmpty (E : List Release → Game Exchange) {mPts : ℕ} : LinkHold mPts := fun g' pts =>
  EmptyAt E g' (linkBits e pts)

/-- **A round's empty class, the event the joint law is held off at a later stop**: no round of the stop's opening, in
a play that ends in `E`, is released as the round's coins' bits. -/
def roundEmpty (E : List Release → Game Exchange) : RoundHold := fun g' _ _ c => EmptyAt E g' (roundBits e c)

/-- **The coin map that writes the live coins into the openings**, at the widening `T` and the start `start`: at each
stop, the link points' bits or a round's coins' bits under `e`, written into the stop's opening in a play that ends in
`E` (`writeOpening`). -/
noncomputable def openMap {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    CoinMap g σ Ω R mPts where
  T := T
  start := start
  link p pts := writeOpening E p.s' (linkBits e pts)
  round p _ _ c := writeOpening E p.s' (roundBits e c)

/-- **The coin map writes each round's coins in within the stop's opening's gap.** -/
theorem openMap_roundClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    (openMap e (R := R) (mPts := mPts) E T start).RoundClose fun q _ s =>
      openGap E q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) := fun q _ _ G hG =>
  avg_le_writeOpening E (roundBits e) q.s' G hG

/-- **The coin map writes the link points in within the stop's opening's gap.** -/
theorem openMap_linkClose {g : Game Exchange} {σ : Strategy g} {Ω : Type} {R mPts : ℕ}
    (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T] (start : Ω × T → CoinTape g σ) :
    (openMap e (R := R) (mPts := mPts) E T start).LinkClose fun q =>
      openGap E q.g' (linkBits e : (Fin 2 → Fin mPts → 𝔽) → List Bool) := fun q G hG =>
  avg_le_writeOpening E (linkBits e) q.s' G hG

/-! ## The coin map's premise within a term -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

variable (V : Vocab d α) (θ : ByteArray) (F : Array ByteArray) (c : ByteArray)

/-- **The joint law's tape off `BL` and `B` is within `ε` of the uniform tape**, at every event `Q` of the period's
tape. -/
def CouplesWithin (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (D : Decoders A.rounds A.Reg)
    (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP) (BL : LinkHold A.mP) (B : RoundHold) (ε : ℝ≥0∞) : Prop :=
  ∀ Q : CoinTape (V.game θ F c) σ → Prop, ENNReal.ofReal (Claims.Pr Q) ≤ jointOff V θ F c A rd σ D M BL B Q + ε

/-- **The coin map's premise at a term within the hiding term.** -/
theorem couples_of_within {A : Live} {rd : Reading A.rounds} {σ : Strategy (V.game θ F c)}
    {D : Decoders A.rounds A.Reg} {M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP} {BL : LinkHold A.mP}
    {B : RoundHold} {ε : ℝ≥0∞} (h : CouplesWithin V θ F c A rd σ D M BL B ε) (hε : ε ≤ A.hidingTerm) :
    Couples V θ F c A rd σ D M BL B := fun Q =>
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

open Classical in
/-- **The coin map's premise within a term, off its local ones**: the joint law's tape off `BL` and `B` is within the
start's term and the stops' of the uniform tape at every event, wherever the coin map writes the link points in within
`γL` of the tape's own and each round's coins within `γ`, its start leaves the tape from the first stop fresh, and its
start is within `ε₀` of the uniform tape. The stops' term carries, beside `γL` and `γ`, the probability of `BL` and of
`B` at each stop. -/
theorem couples_of_close (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (M : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    {γ : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → (j : ℕ) → SubmitJ j → ℝ≥0∞}
    {γL : Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞} {ε₀ : ℝ≥0∞} (BL : LinkHold A.mP) (B : RoundHold)
    (hR : M.RoundClose γ) (hL : M.LinkClose γL) (hF : FirstStopFresh V θ F c A rd σ M)
    (hS : StartWithin V θ F c A σ M ε₀) :
    CouplesWithin V θ F c A rd σ D M BL B (ε₀ + couplesErr V θ F c A rd σ D M
      (fun q j s => γ q j s + prCoin (B q.g' j s)) (fun q => γL q + prCoin (BL q.g'))) := fun Q => by
  let R₀ := regOf V θ F c A rd σ D
  let G : Finset (Fin A.n) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞ := fun dr q =>
    avg fun pts : Fin 2 → Fin A.mP → 𝔽 =>
      if BL q.g' pts then 0 else roundsOff D M (A.J dr R₀) B Q A.N (q.rewrite (M.link q pts)).consume
  have hG : jointOff V θ F c A rd σ D M BL B Q =
      avg fun ω : (A.extend M.widening).L.Ω => G ((A.extend M.widening).L.draw ω) (firstStop V θ F c A rd σ M ω) :=
    rfl
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
            (Pos.resample (fun q' => γL q' + prCoin (BL q'.g')) (firstStop V θ F c A rd σ M ω) +
              Pos.resample (roundsErr D (A.J ((A.extend M.widening).L.draw ω) R₀)
                (fun q j s => γ q j s + prCoin (B q.g' j s)) A.N) (firstStop V θ F c A rd σ M ω).consume)) + ε₀ :=
        add_le_add (avg_mono fun ω => resample_sessionOff_le (D := D) (B := B) hL hR BL Q A.N _) le_rfl
    _ = (avg fun ω : (A.extend M.widening).L.Ω =>
          Pos.resample (G ((A.extend M.widening).L.draw ω)) (firstStop V θ F c A rd σ M ω)) +
          couplesErr V θ F c A rd σ D M (fun q j s => γ q j s + prCoin (B q.g' j s))
            (fun q => γL q + prCoin (BL q.g')) + ε₀ := by
        rw [avg_add]
        rfl
    _ = _ := by
        rw [← hF G]
        ring

/-- **The coin map that writes the live coins into the openings, within the openings' gaps and empty classes**: at
the plays' close `E`, its widening `T` and start `start`, the joint law's tape off the empty classes (`linkEmpty`,
`roundEmpty`) is within the start's term and, over the stops, the openings' gaps and empty classes of the uniform tape,
wherever the start leaves the tape from the first stop fresh and is within `ε₀` of the uniform tape. -/
theorem openMap_couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (E : List Release → Game Exchange) (T : Type) [Fintype T] [Nonempty T]
    (start : A.Ω × T → CoinTape (V.game θ F c) σ) {ε₀ : ℝ≥0∞}
    (hF : FirstStopFresh V θ F c A rd σ (openMap e E T start))
    (hS : StartWithin V θ F c A σ (openMap e E T start) ε₀) :
    CouplesWithin V θ F c A rd σ D (openMap e E T start) (linkEmpty e E) (roundEmpty e E)
      (ε₀ + couplesErr V θ F c A rd σ D (openMap e E T start)
        (fun q _ s => openGap E q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
          openEmpty E q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
        (fun q => openGap E q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
          openEmpty E q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool))) :=
  couples_of_close V θ F c A rd σ D _ (linkEmpty e E) (roundEmpty e E) (openMap_roundClose e E T start)
    (openMap_linkClose e E T start) hF hS

/-! ## The sample's coins in the fresh frames of the sample's family -/

/-- **A fresh frame's coin node, described**: `g'` is the device's flip of the coins of a fresh frame of the family
`label` answering `after`, with the releases' continuation after it (`relK`'s fresh branch), ending in `E` at the
device's log: the releases `pre` before the frame, the frame's, then the rest. -/
structure FreshAt {β : Type} (E : List Release → Game β) (label : String) (after : ByteArray) (g' : Game β) where
  k : ℕ
  fs : List (String × ℕ)
  ts : List OpenTree
  pre : List Release
  iF : Fintype (Fin 512 → Bool)
  iN : Nonempty (Fin 512 → Bool)
  eq : g' = @Game.coin β (Fin 512 → Bool) iF iN fun x =>
    relK k fs ts fun log => E (pre ++ ⟨.fresh label, after, List.ofFn x⟩ :: log)

/-- A fresh frame's node flips 512 bits. -/
theorem FreshAt.coin_eq {β C : Type} {iF : Fintype C} {iN : Nonempty C} {k : C → Game β}
    {E : List Release → Game β} {label : String} {after : ByteArray}
    (o : FreshAt E label after (@Game.coin β C iF iN k)) : C = (Fin 512 → Bool) := by
  have h := o.eq
  injection h

open Classical in
/-- **The fresh frames of the family `label`, as a fill writes them** (`fill`): the coin nodes described as fresh
frames of the family `label` answering `after` in a play that ends in `E`, each taking a written coin as its 512
bits. -/
noncomputable def freshSel (E : List Release → Game Exchange) (label : String) (after : ByteArray) :
    FillSel Exchange (Fin 512 → Bool) :=
  fun {C : Type} [iF : Fintype C] [iN : Nonempty C] (k : C → Game Exchange) =>
    if h : Nonempty (FreshAt E label after (@Game.coin _ C iF iN k)) then
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
    _root_.Properties.Spot.bytesOf (A.src ω 0) =
      _root_.Properties.Spot.bytesOf (coinBytes m 64 (srcCoins A hL ω) 0) := by
  have h : coinBytes m 64 (srcCoins A hL ω) = srcEquiv hL (A.src ω) := (coinBytesEquiv m 64).symm_apply_apply _
  have key : ∀ {L L' : ℕ} (h : L = L') (f : Fin L → Fin 256), _root_.Properties.Spot.bytesOf f =
      _root_.Properties.Spot.bytesOf fun j : Fin L' => f (Fin.cast h.symm j) := by
    intro L L' h f
    subst h
    rfl
  rw [h]
  exact key hL (A.src ω 0)

/-- **The fresh frames of the family `label` answering `after` in `σ`'s play**: those whose log ends in the close
after `σ`'s request (`freshSel`). -/
noncomputable def sampleSel (σ : Strategy (V.game θ F c)) (label : String) (after : ByteArray) :
    FillSel Exchange (Fin 512 → Bool) :=
  freshSel (finish (V.signer θ) θ F c σ.1) label after

/-- **The start that writes the sample's coins into the fresh frames of the sample's family**: on the device's coins and
a tape of the period's game, the tape filled with the coins `κ` at the fresh frames of the family `label` answering
`after` in `σ`'s play, up to the first node a round's cut fixes (`fill`). -/
noncomputable def sampleStart {R : ℕ} (rd : Reading R) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {Ω : Type} {m : ℕ} (κ : Ω → Fin m → Fin 512 → Bool) (ω : Ω × CoinTape (V.game θ F c) σ) :
    CoinTape (V.game θ F c) σ :=
  fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2

/-- **The sample-coupled coin map**: widened by a tape of the period's game, its start writes the sample's coins `κ`
into the fresh frames of the sample's family (`sampleStart`), and at each stop it writes the live coins into the stop's
opening in `σ`'s play, whose log ends in the close after `σ`'s request (`openMap`). -/
noncomputable def sampleMap (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool) : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP :=
  openMap e (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ) (sampleStart V θ F c rd σ label after κ)

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
      fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2 :=
  rfl

/-- The sample-coupled map's first stop is the walk to it from the filled tape. -/
theorem firstStop_sampleMap (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (ω : (A.extend (sampleMap e V θ F c A rd σ label after κ).widening).L.Ω) :
    firstStop V θ F c A rd σ (sampleMap e V θ F c A rd σ label after κ) ω =
      Pos.walk (V.game θ F c) σ (roundCuts V θ F c rd)
        (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m (κ ω.1) ω.2) id :=
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
  exact avg_prod_walk_fill (C := A.Ω) (sampleSel V θ F c σ label after) (fun ω₁ : A.Ω => G (dr ω₁))
    (roundCuts V θ F c rd) m κ

/-- **The sample-coupled start is the uniform tape** wherever the sample's coins are uniform (`avg_srcCoins`). -/
theorem sampleMap_start (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c)) (label : String)
    (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    StartWithin V θ F c A σ (sampleMap e V θ F c A rd σ label after κ) 0 := fun H _ => by
  rw [add_zero]
  refine le_of_eq (Eq.trans ?_ (avg_prod (C := A.Ω) (D := CoinTape (V.game θ F c) σ) _).symm)
  exact (avg_fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m H).symm.trans
    (hκ fun κ' => avg fun t =>
      H (fill (sampleSel V θ F c σ label after) (V.game θ F c) σ (roundCuts V θ F c rd) m κ' t)).symm

/-- **The sample-coupled coin map's premise within the openings' gaps and empty classes**: the joint law's tape off the
empty classes is within the openings' gaps and empty classes over the stops of the uniform tape, wherever the sample's
coins are uniform. -/
theorem sampleMap_couples (A : Live) (rd : Reading A.rounds) (σ : Strategy (V.game θ F c))
    (D : Decoders A.rounds A.Reg) (label : String) (after : ByteArray) {m : ℕ} (κ : A.Ω → Fin m → Fin 512 → Bool)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ ω)) = avg H) :
    CouplesWithin V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
      (linkEmpty e (finish (V.signer θ) θ F c σ.1)) (roundEmpty e (finish (V.signer θ) θ F c σ.1))
      (couplesErr V θ F c A rd σ D (sampleMap e V θ F c A rd σ label after κ)
        (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
          openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
        (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
          openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool))) := by
  have h := openMap_couples e V θ F c A rd σ D (finish (V.signer θ) θ F c σ.1) (CoinTape (V.game θ F c) σ)
    (sampleStart V θ F c rd σ label after κ) (sampleMap_fresh e V θ F c A rd σ label after κ)
    (sampleMap_start e V θ F c A rd σ label after κ hκ)
  rwa [zero_add] at h

omit θ F c in
/-- **The sample-coupled coin maps** at the readings `rd`: at each session, commitment and strategy, `sampleMap` at the
family `label θ F c σ` answering `after θ F c σ` and the session's coins `κ A`. The family and the digest it answers
are the strategy's: the commitment `c` fixes the registration digest only as a preimage, and the play reads it. -/
noncomputable def sampleMaps (rd : Readings)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (θ : ByteArray) (F : Array ByteArray)
    (c : ByteArray) (σ : Strategy (V.game θ F c)) : CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP :=
  sampleMap e V θ F c A (rd A θ F c) σ (label θ F c σ) (after θ F c σ) (κ A)

variable (k : Verity.Registration.Composite)

omit θ F c in
/-- **The coupling at the projection, off the coin map's premises within a term**: `couplingTarget_withProject` at the
cut `P` and `Q`, off the held-off events `BL` and `B`, with the coin map's premise discharged by `couples_of_close` at
each commitment and strategy, wherever the start's term and the stops' (`γL` and `BL`'s probability at the first stop,
`γ` and `B`'s at each later one) sum to at most the hiding term. -/
theorem couplingTarget_withProject_of_close (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (M : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → (σ : Strategy (V.game θ F c)) →
      CoinMap (V.game θ F c) σ A.Ω A.rounds A.mP)
    (A : Live) (P : Exchange → Prop) (θ : ByteArray) (F : Array ByteArray)
    (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (BL : (c : ByteArray) → Strategy (V.game θ F c) → LinkHold A.mP)
    (B : (c : ByteArray) → Strategy (V.game θ F c) → RoundHold)
    (γ : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) →
      (j : ℕ) → SubmitJ j → ℝ≥0∞)
    (γL : (c : ByteArray) → (σ : Strategy (V.game θ F c)) → Pos (V.game θ F c) σ Claims.Msg (Fin A.rounds) → ℝ≥0∞)
    (ε₀ : (c : ByteArray) → Strategy (V.game θ F c) → ℝ≥0∞)
    (ha : LoggedBound V k (K.withProject V rd D M) P θ F) (hq : DigestBound V k (K.withProject V rd D M) A θ F Q)
    (hR : ∀ c σ, (M A θ F c σ).RoundClose (γ c σ)) (hL : ∀ c σ, (M A θ F c σ).LinkClose (γL c σ))
    (hF : ∀ c σ, FirstStopFresh V θ F c A (rd A θ F c) σ (M A θ F c σ))
    (hS : ∀ c σ, StartWithin V θ F c A σ (M A θ F c σ) (ε₀ c σ))
    (hε : ∀ c σ, ε₀ c σ + couplesErr V θ F c A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ)
      (fun q j s => γ c σ q j s + prCoin (B c σ q.g' j s)) (fun q => γL c σ q + prCoin (BL c σ q.g')) ≤
        A.hidingTerm)
    (hD : ∀ c σ, Decodes V θ F c k (fun o => P o ∧ Q c σ o) A (rd A θ F c) σ (D A θ F c σ) (M A θ F c σ) (BL c σ)
      (B c σ)) :
    CouplingTarget V k ((K.withProject V rd D M).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D M A P θ F Q BL B ha hq
    (fun c σ => couples_of_within V θ F c
      (couples_of_close V θ F c A _ σ _ _ (BL c σ) (B c σ) (hR c σ) (hL c σ) (hF c σ) (hS c σ)) (hε c σ)) hD

omit θ F c in
/-- **The coupling at the projection with the sample-coupled coin maps**: `couplingTarget_withProject` at
`sampleMaps` and the cut `P` and `Q`, off the empty classes (`linkEmpty`, `roundEmpty`), whose coin map's premise
holds within the openings' gaps and empty classes over the stops wherever the session's coins are uniform
(`sampleMap_couples`), at each commitment and strategy where those terms are at most the hiding term. -/
theorem couplingTarget_withProject_of_sample (rd : Readings) (K : Constructions V)
    (D : (A : Live) → (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) →
      Decoders A.rounds A.Reg)
    (label : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → String)
    (after : (θ : ByteArray) → (F : Array ByteArray) → (c : ByteArray) → Strategy (V.game θ F c) → ByteArray) {m : ℕ}
    (κ : (A : Live) → A.Ω → Fin m → Fin 512 → Bool) (A : Live) (P : Exchange → Prop) (θ : ByteArray)
    (F : Array ByteArray) (Q : (c : ByteArray) → Strategy (V.game θ F c) → Exchange → Prop)
    (hκ : ∀ H : (Fin m → Fin 512 → Bool) → ℝ≥0∞, avg (fun ω => H (κ A ω)) = avg H)
    (ha : LoggedBound V k (K.withProject V rd D (sampleMaps e V rd label after κ)) P θ F)
    (hq : DigestBound V k (K.withProject V rd D (sampleMaps e V rd label after κ)) A θ F Q)
    (hε : ∀ c σ, couplesErr V θ F c A (rd A θ F c) σ (D A θ F c σ) (sampleMaps e V rd label after κ A θ F c σ)
      (fun q _ s => openGap (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (roundBits e : (Fin s.n → 𝔽) → List Bool))
      (fun q => openGap (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool) +
        openEmpty (finish (V.signer θ) θ F c σ.1) q.g' (linkBits e : (Fin 2 → Fin A.mP → 𝔽) → List Bool)) ≤
        A.hidingTerm)
    (hD : ∀ c σ, Decodes V θ F c k (fun o => P o ∧ Q c σ o) A (rd A θ F c) σ (D A θ F c σ)
      (sampleMaps e V rd label after κ A θ F c σ) (linkEmpty e (finish (V.signer θ) θ F c σ.1))
      (roundEmpty e (finish (V.signer θ) θ F c σ.1))) :
    CouplingTarget V k ((K.withProject V rd D (sampleMaps e V rd label after κ)).withMessages rd) A θ F :=
  couplingTarget_withProject V k rd K D (sampleMaps e V rd label after κ) A P θ F Q
    (fun c σ => linkEmpty e (finish (V.signer θ) θ F c σ.1)) (fun c σ => roundEmpty e (finish (V.signer θ) θ F c σ.1))
    ha hq (fun c σ => couples_of_within V θ F c (sampleMap_couples e V θ F c A (rd A θ F c) σ (D A θ F c σ)
      (label θ F c σ) (after θ F c σ) (κ A) hκ) (hε c σ)) hD

end Verity.Sampled
