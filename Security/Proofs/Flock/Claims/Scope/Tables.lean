import Proofs.Flock.Claims.Scope.System
import Proofs.Flock.Soundness.ExecPlacement
import Proofs.Flock.Soundness.Refine.Regions

/-!
# The verifier's tables as the readable relation

Each table of the verifier's statement is a block R1CS over GF(2), the model's `Statement` (`Layout.model`: `m`, `k_log`,
the pin, the level-0 matrices `placedA`, `placedB`, the regions `regionsOf`). `Tab` is that data without its proofs
(`Tab.ofStmt`), `sysOf m kLog ts` the tables as one `Sys` in blocks of `2^kLog` bits, `2^(m − kLog)` of them, and
`relation` the deployed relation's fields: `sysOf` at the first table's `m` and `k_log`, over every table.

* `tab_iff`: a table's rows, pin and fixed bits hold in every block exactly when the model's statement is satisfied.
* `sysOf_iff`: the system holds in every block exactly when every table fits the blocks (`Tab.Fits`: its own `m` and
  `k_log`, `6 ≤ k_log ≤ m`, its pin in the block) and its model statement is satisfied by its share of the hidden
  values (`zOf`: table `c`'s bit `a` is hidden value `c·2^m + a`). A table that does not fit is unsatisfiable.
* `relS_relation`: `Claims.Flock.RelS` of `toReadable relation p` is that, for every table of `p`.
* `toModel_ofStmt`: at a table whose statement has the verifier's layout, the model statement is `Layout.model` at
  `regionsOf`, the one the soundness proof reads.
-/

namespace Flock.Readable

open FlockSoundness

/-- A bit as an element of GF(2). -/
def toZ (x : Bool) : ZMod 2 := if x then 1 else 0

/-- An element of GF(2) as a bit. -/
def ofZ (x : ZMod 2) : Bool := decide (x = 1)

theorem toZ_ofZ : ∀ x : ZMod 2, toZ (ofZ x) = x := by decide

theorem toZ_eq_iff : ∀ (a : Bool) (r : ZMod 2), toZ a = r ↔ a = ofZ r := by decide

theorem toZ_inj : ∀ a b : Bool, toZ a = toZ b ↔ a = b := by decide

theorem toZ_xor : ∀ a b : Bool, toZ (xor a b) = toZ a + toZ b := by decide

theorem toZ_and : ∀ a b : Bool, toZ (a && b) = toZ a * toZ b := by decide

theorem toZ_xsum (A x : ℕ → Bool) (n : ℕ) :
    toZ (xsum A x n) = ∑ r ∈ Finset.range n, toZ (A r) * toZ (x r) := by
  induction n with
  | zero => rfl
  | succ n ih => rw [xsum_succ, toZ_xor, toZ_and, ih, Finset.sum_range_succ]

/-- **One table**: the model's block R1CS statement without its proofs. -/
structure Tab where
  m : ℕ
  kLog : ℕ
  pin : ℕ
  A : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  B : Matrix (Fin (2 ^ kLog)) (Fin (2 ^ kLog)) (ZMod 2)
  regions : List (Model.Region kLog (m - kLog))

instance : Inhabited Tab := ⟨⟨0, 0, 0, 0, 0, []⟩⟩

namespace Tab

/-- The table of a verifier's statement: its `m`, `k_log` and pin, the level-0 matrices it folds and its regions. -/
def ofStmt (st : Stmt) : Tab := ⟨st.m, st.c.kLog, st.pin, FlockLevel3.placedA st, FlockLevel3.placedB st, Refine.regionsOf st⟩

/-- What the model's statement needs of a table: `6 ≤ k_log ≤ m` and the pin in the block. -/
def Ok (t : Tab) : Prop := 6 ≤ t.kLog ∧ t.kLog ≤ t.m ∧ t.pin < 2 ^ t.kLog

/-- The table fits blocks of `2^kLog` bits, `2^(m − kLog)` of them. -/
def Fits (t : Tab) (m kLog : ℕ) : Prop := t.m = m ∧ t.kLog = kLog ∧ t.Ok

instance (t : Tab) : Decidable t.Ok := by unfold Ok; infer_instance

instance (t : Tab) (m kLog : ℕ) : Decidable (t.Fits m kLog) := by unfold Fits; infer_instance

/-- **The model's statement** of a table that has what it needs. -/
abbrev toModel (t : Tab) (h : t.Ok) : Model.Statement where
  m := t.m
  kLog := t.kLog
  hk := h.2.1
  hk6 := h.1
  pin := ⟨t.pin, h.2.2⟩
  A₀ := t.A
  B₀ := t.B
  regions := t.regions

/-- Row `i`'s columns on the A side. -/
def a (t : Tab) (i r : ℕ) : Bool := if h : i < 2 ^ t.kLog ∧ r < 2 ^ t.kLog then ofZ (t.A ⟨i, h.1⟩ ⟨r, h.2⟩) else false

/-- Row `i`'s columns on the B side. -/
def b (t : Tab) (i r : ℕ) : Bool := if h : i < 2 ^ t.kLog ∧ r < 2 ^ t.kLog then ofZ (t.B ⟨i, h.1⟩ ⟨r, h.2⟩) else false

/-- The regions' bits as fixed bits: each region bit's in-block address, and its value in each block. -/
def fixed (t : Tab) : List (ℕ × (ℕ → Bool)) :=
  t.regions.flatMap fun R => ((List.range (2 ^ R.free.length)).filter fun i => decide (R.inBlock i < 2 ^ t.kLog)).map
    fun i => (R.inBlock i, fun o => if ho : o < 2 ^ (t.m - t.kLog) then ofZ (R.bits ⟨o, ho⟩ i) else false)

/-- **The table holds in `x`** (`x o i`: bit `i` of block `o`): every block's rows, pin and fixed bits. -/
def Holds (t : Tab) (x : ℕ → ℕ → Bool) : Prop :=
  ∀ o < 2 ^ (t.m - t.kLog),
    (∀ i < 2 ^ t.kLog, (xsum (t.a i) (x o) (2 ^ t.kLog) && xsum (t.b i) (x o) (2 ^ t.kLog)) = x o i) ∧
      x o t.pin = true ∧ ∀ f ∈ t.fixed, x o f.1 = f.2 o

theorem toZ_row (t : Tab) (M : Matrix (Fin (2 ^ t.kLog)) (Fin (2 ^ t.kLog)) (ZMod 2)) (s : ℕ → ℕ → Bool)
    (hs : ∀ (i r : Fin (2 ^ t.kLog)), s i r = ofZ (M i r)) (v : Fin (2 ^ t.kLog) → ZMod 2) (x : ℕ → Bool)
    (hx : ∀ r : Fin (2 ^ t.kLog), toZ (x r) = v r) (i : Fin (2 ^ t.kLog)) :
    toZ (xsum (s i) x (2 ^ t.kLog)) = M.mulVec v i := by
  rw [toZ_xsum, Finset.sum_range (fun r => toZ (s i r) * toZ (x r))]
  simp only [Matrix.mulVec, dotProduct, hs, toZ_ofZ, hx]

/-- **A table holds exactly when its model statement is satisfied**, at a witness whose blocks are `x`'s. -/
theorem tab_iff (t : Tab) (h : t.Ok) (z : Model.Witness t.m) (x : ℕ → ℕ → Bool)
    (hx : ∀ (o : Fin (2 ^ (t.m - t.kLog))) (i : Fin (2 ^ t.kLog)), toZ (x o i) = (t.toModel h).block z o i) :
    t.Holds x ↔ (t.toModel h).Satisfies z := by
  have hrow : ∀ (o : Fin (2 ^ (t.m - t.kLog))) (i : Fin (2 ^ t.kLog)),
      ((t.toModel h).A₀.mulVec ((t.toModel h).block z o) i * (t.toModel h).B₀.mulVec ((t.toModel h).block z o) i =
        (t.toModel h).block z o i) ↔
      (xsum (t.a i) (x o) (2 ^ t.kLog) && xsum (t.b i) (x o) (2 ^ t.kLog)) = x o i := by
    intro o i
    have ha : ∀ i r : Fin (2 ^ t.kLog), t.a i r = ofZ (t.A i r) := fun i r => by simp [a]
    have hb : ∀ i r : Fin (2 ^ t.kLog), t.b i r = ofZ (t.B i r) := fun i r => by simp [b]
    rw [← toZ_inj, toZ_and, toZ_row t t.A t.a ha _ (x o) (hx o), toZ_row t t.B t.b hb _ (x o) (hx o), hx o i]
  have hfix : ∀ o : Fin (2 ^ (t.m - t.kLog)), (∀ f ∈ t.fixed, x o f.1 = f.2 o) ↔
      ∀ R ∈ (t.toModel h).regions, ∀ (i : ℕ) (_hi : i < 2 ^ R.free.length) (hb : R.inBlock i < 2 ^ (t.toModel h).kLog),
        (t.toModel h).block z o ⟨R.inBlock i, hb⟩ = R.bits o i := by
    intro o
    simp only [fixed, List.mem_flatMap, List.mem_map, List.mem_filter, List.mem_range, decide_eq_true_eq]
    constructor
    · intro hf R hR i hi hb
      have := hf _ ⟨R, hR, i, ⟨hi, hb⟩, rfl⟩
      simp only [o.isLt, ↓reduceDIte] at this
      rw [← hx o ⟨R.inBlock i, hb⟩, toZ_eq_iff]
      exact this
    · rintro hf _ ⟨R, hR, i, ⟨hi, hb⟩, rfl⟩
      simp only [o.isLt, ↓reduceDIte]
      rw [← toZ_eq_iff]
      exact (hx o ⟨R.inBlock i, hb⟩).trans (hf R hR i hi hb)
  unfold Holds Model.Statement.Satisfies
  constructor
  · intro H
    refine ⟨fun o i => (hrow o i).2 ((H o o.isLt).1 i i.isLt), fun o => ?_, fun R hR o => (hfix o).1 (H o o.isLt).2.2 R hR⟩
    rw [show (t.toModel h).pin = ⟨t.pin, h.2.2⟩ from rfl, ← hx o ⟨t.pin, h.2.2⟩, (H o o.isLt).2.1]
    rfl
  · rintro ⟨H1, H2, H3⟩ o ho
    refine ⟨fun i hi => (hrow ⟨o, ho⟩ ⟨i, hi⟩).1 (H1 ⟨o, ho⟩ ⟨i, hi⟩), ?_,
      (hfix ⟨o, ho⟩).2 fun R hR => H3 R hR ⟨o, ho⟩⟩
    have := H2 ⟨o, ho⟩
    rw [show (t.toModel h).pin = ⟨t.pin, h.2.2⟩ from rfl, ← hx ⟨o, ho⟩ ⟨t.pin, h.2.2⟩] at this
    exact (toZ_eq_iff _ _).1 this

/-- **The model statement of a verifier's table** is `Layout.model` at its regions, wherever the statement has the
verifier's layout. -/
theorem toModel_ofStmt {st : Stmt} (L : FlockSoundness.Layout st) (h : (ofStmt st).Ok) :
    (ofStmt st).toModel h = L.model (Refine.regionsOf st) := rfl

theorem ok_ofStmt {st : Stmt} (L : FlockSoundness.Layout st) : (ofStmt st).Ok :=
  ⟨L.kLog6, by rw [show (ofStmt st).m = st.m from rfl, L.m]; exact Nat.le_add_right _ _, L.pin_lt⟩

end Tab

/-- Table `c`'s share of the hidden values, `T` bits per table, as a model witness. -/
def zOf (T : ℕ) (w : Claims.Flock.Wit) (c m : ℕ) : Model.Witness m := fun a => toZ (w.hid.getD (c * T + a.val) false)

/-- A table's fixed bits in the system: its regions' when it fits the blocks, else bit 0 fixed to `0` (against the pin,
which is bit 0 then: no witness satisfies it). -/
def fixedOf (m kLog : ℕ) (t : Tab) : List (ℕ × (ℕ → Bool)) := if t.Fits m kLog then t.fixed else [(0, fun _ => false)]

/-- **The tables as one system**, in blocks of `2^kLog` bits, `2^(m − kLog)` of them, `2^(m − kLog)·2^kLog` hidden bits
per table. -/
def sysOf (m kLog : ℕ) (ts : List Tab) : Sys where
  J := ts.length
  N := 2 ^ kLog
  T := 2 ^ (m - kLog) * 2 ^ kLog
  nB := 2 ^ (m - kLog)
  A c := (ts.getD c default).a
  B c := (ts.getD c default).b
  pin c := if (ts.getD c default).Fits m kLog then (ts.getD c default).pin else 0
  slots := (List.range ts.length).flatMap fun c => (fixedOf m kLog (ts.getD c default)).map fun f => (c, f.1, f.2)

theorem sysOf_wf (m kLog : ℕ) (ts : List Tab) : (sysOf m kLog ts).WF where
  N_pos := Nat.two_pow_pos _
  pin_lt c _ := by
    simp only [sysOf]
    split
    · rename_i h; exact h.2.1 ▸ h.2.2.2.2
    · exact Nat.two_pow_pos _
  slot_lt s hs := by
    simp only [sysOf, List.mem_flatMap, List.mem_map, List.mem_range] at hs
    obtain ⟨c, hc, f, hf, rfl⟩ := hs
    refine ⟨hc, ?_⟩
    simp only [fixedOf] at hf
    split at hf
    · rename_i h
      simp only [Tab.fixed, List.mem_flatMap, List.mem_map, List.mem_filter, List.mem_range, decide_eq_true_eq] at hf
      obtain ⟨R, -, i, ⟨-, hb⟩, rfl⟩ := hf
      exact h.2.1 ▸ hb
    · simp only [List.mem_singleton] at hf
      subst hf
      exact Nat.two_pow_pos _

/-- **The system holds in every block** exactly when every table fits the blocks and its share of the hidden values
satisfies its model statement. -/
theorem sysOf_iff (m kLog : ℕ) (ts : List Tab) (w : Claims.Flock.Wit) :
    (∀ b < (sysOf m kLog ts).nB, (sysOf m kLog ts).Holds w b) ↔
      ∀ c < ts.length, ∃ h : (ts.getD c default).Fits m kLog,
        ((ts.getD c default).toModel h.2.2).Satisfies (zOf (2 ^ (m - kLog) * 2 ^ kLog) w c (ts.getD c default).m) := by
  have hx : ∀ c b, (sysOf m kLog ts).bit w c b = fun a => w.hid.getD (c * (2 ^ (m - kLog) * 2 ^ kLog) + b * 2 ^ kLog + a)
      false := fun _ _ => rfl
  have hslots : ∀ b, (∀ s ∈ (sysOf m kLog ts).slots, (sysOf m kLog ts).bit w s.1 b s.2.1 = s.2.2 b) ↔
      ∀ c < ts.length, ∀ f ∈ fixedOf m kLog (ts.getD c default), (sysOf m kLog ts).bit w c b f.1 = f.2 b := by
    intro b
    simp only [sysOf, List.mem_flatMap, List.mem_map, List.mem_range]
    constructor
    · intro H c hc f hf
      exact H _ ⟨c, hc, f, hf, rfl⟩
    · rintro H _ ⟨c, hc, f, hf, rfl⟩
      exact H c hc f hf
  have hsplit : (∀ b < (sysOf m kLog ts).nB, (sysOf m kLog ts).Holds w b) ↔
      ∀ c < ts.length, ∀ b < 2 ^ (m - kLog),
        (∀ i < 2 ^ kLog, (xsum ((ts.getD c default).a i) ((sysOf m kLog ts).bit w c b) (2 ^ kLog) &&
            xsum ((ts.getD c default).b i) ((sysOf m kLog ts).bit w c b) (2 ^ kLog)) =
          (sysOf m kLog ts).bit w c b i) ∧
        (sysOf m kLog ts).bit w c b ((sysOf m kLog ts).pin c) = true ∧
        ∀ f ∈ fixedOf m kLog (ts.getD c default), (sysOf m kLog ts).bit w c b f.1 = f.2 b := by
    simp only [Sys.Holds, hslots]
    constructor
    · intro H c hc b hb
      exact ⟨(H b hb).1 c hc, (H b hb).2.1 c hc, (H b hb).2.2 c hc⟩
    · intro H b hb
      exact ⟨fun c hc => (H c hc b hb).1, fun c hc => (H c hc b hb).2.1, fun c hc => (H c hc b hb).2.2⟩
  rw [hsplit]
  refine forall₂_congr fun c hc => ?_
  by_cases hf : (ts.getD c default).Fits m kLog
  · have hpin : (sysOf m kLog ts).pin c = (ts.getD c default).pin := ite_eq_left hf
    have hfix : fixedOf m kLog (ts.getD c default) = (ts.getD c default).fixed := ite_eq_left hf
    rw [hpin, hfix]
    obtain ⟨hm, hk, hok⟩ := hf
    have key := Tab.tab_iff (ts.getD c default) hok (zOf (2 ^ (m - kLog) * 2 ^ kLog) w c (ts.getD c default).m)
      (fun o => (sysOf m kLog ts).bit w c o) (fun o i => by
        simp only [Model.Statement.block, Model.Statement.addr, zOf, Sys.bit, sysOf, hk, Nat.add_assoc])
    constructor
    · intro H
      exact ⟨⟨hm, hk, hok⟩, key.1 (by subst hm hk; exact H)⟩
    · rintro ⟨_, H⟩
      have := key.2 H
      subst hm hk
      exact this
  · have hpin : (sysOf m kLog ts).pin c = 0 := ite_eq_right hf
    have hfix : fixedOf m kLog (ts.getD c default) = [(0, fun _ => false)] := ite_eq_right hf
    rw [hpin, hfix]
    refine iff_of_false (fun H => ?_) (fun ⟨h, _⟩ => hf h)
    have h1 := (H 0 (Nat.two_pow_pos _)).2.1
    have h2 := (H 0 (Nat.two_pow_pos _)).2.2 _ (List.mem_singleton_self _)
    exact absurd (h1.symm.trans h2) (by decide)

/-- **The deployed relation's fields**: the tables of the verifier's statement as one system, at the first table's `m`
and `k_log` (`toReadable`'s). -/
def relation (p : Parsed) : Relation :=
  (sysOf ((p.sts[0]?.map (·.m)).getD 0) ((p.sts[0]?.map (·.c.kLog)).getD 0) (p.sts.toList.map Tab.ofStmt)).rel

/-- **`RelS` at the deployed relation**: a witness satisfies the readable statement exactly when every table of the
verifier's statement fits the first table's blocks and its share of the hidden values satisfies its model
statement. -/
theorem relS_relation (H : Claims.Msg → Claims.Msg) (p : Parsed) (w : Claims.Flock.Wit) :
    Claims.Flock.RelS H (toReadable relation p) w ↔
      ∀ c < p.sts.size, ∃ h : (Tab.ofStmt p.sts[c]!).Fits (toReadable relation p).m (toReadable relation p).kLog,
        ((Tab.ofStmt p.sts[c]!).toModel h.2.2).Satisfies
          (zOf (2 ^ ((toReadable relation p).m - (toReadable relation p).kLog) * 2 ^ (toReadable relation p).kLog) w c
            p.sts[c]!.m) := by
  rw [Sys.relS_iff H (sysOf_wf _ _ _) (toReadable relation p) w rfl rfl rfl rfl rfl rfl, sysOf_iff]
  simp only [List.length_map, Array.length_toList]
  refine forall₂_congr fun c hc => ?_
  have : (p.sts.toList.map Tab.ofStmt).getD c default = Tab.ofStmt p.sts[c]! := by
    rw [List.getD_eq_getElem?_getD, List.getElem?_map, Array.getElem?_toList, Array.getElem?_eq_getElem hc,
      Option.map_some, Option.getD_some, getElem!_pos p.sts c hc]
  rw [this]
  rfl

end Flock.Readable
