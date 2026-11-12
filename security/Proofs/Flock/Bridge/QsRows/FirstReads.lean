module

public import Proofs.Core.Game.Tape
public import Proofs.Flock.Bridge.Draw
public import Proofs.Flock.Bridge.QsRows.RunsOut
public import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The deployed sample runs out of bytes with probability at most `k_d·2⁻⁶⁴`

The deployed sample (`Spot.Draw.drawUnits`) reads one byte stream in order (`Spot.Draw.take`): a subset sample is one
stage, a stratified or work sample one stage per stratum, each stage's sampler run on its first `need` bytes and on
`need` more whenever it returns nothing (`Flock.Draw.onWith`). A stage whose sampler returns a sample on its first read
reads nothing more, so the next stage's first read starts where it ends. On a stream that holds every stage's first
read the sample therefore ends unless some stage's sampler returns nothing on its first read, and a stratum's does with
probability at most `k_s·2⁻⁶⁴` (`Law.subset_prefix_none_le`), whatever the bytes before it.

* `FirstOk`: every stage's first read is there and its sampler returns a sample on it.
* `firstOk_fail_le`: on a uniform stream that holds the stages' first reads, some stage's first read fails with
  probability at most the stages' bounds added (the stages' first reads are disjoint blocks of the stream).
* `onWith_take_first`, `stratifiedWith_take_first`, `units_ok_of_firstOk`: where every first read samples,
  `onWith`, the stratified sample and `drawUnits` return.
* `Flock.Bridge.firstRead`: the bytes the stages read first, in all.
* `Flock.Bridge.strata_fail_le`: some stratum's first read fails with probability at most `(Σ_s k_s)·2⁻⁶⁴`.
* `Flock.Bridge.units_error_le`: under a sampling design on strata that samples `kd` units (`LawOn`), on a uniform
  stream of at least `firstRead law n` bytes, `drawUnits` returns no sample with probability at most `kd·2⁻⁶⁴`.
* `Flock.Bridge.execSel_error_le`, `execSel_card_le`: the same of the deployed sample on a uniform byte source
  (A3), and a deployed sample that ends holds at most `kd` proof blocks.
-/

@[expose] public section

namespace Properties.Spot.SecurityProofs

open FlockSoundness FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.DrawExec Finset
open scoped ENNReal

/-! ## Every stage's first read samples -/

/-- **Every stage's first read samples**: from `p`, each stage's `need` bytes are there and its sampler returns a sample
on them, the next stage reading from where they end. -/
def FirstOk : List Stage → ByteArray → ℕ → Prop
  | [], _, _ => True
  | s :: ss, src, p => p + s.need ≤ src.size ∧ (∃ D, s.run (src.extract p (p + s.need)) = some D) ∧
      FirstOk ss src (p + s.need)

/-- The stages' first reads from `p + q` of `src` are their first reads from `q` of `src`'s bytes after `p`. -/
theorem firstOk_shift {src : ByteArray} {p : ℕ} (hp : p ≤ src.size) :
    ∀ (ss : List Stage) (q : ℕ), FirstOk ss src (p + q) ↔ FirstOk ss (src.extract p src.size) q
  | [], _ => ⟨fun _ => trivial, fun _ => trivial⟩
  | s :: ss, q => by
    have hsz : (src.extract p src.size).size = src.size - p := by
      rw [ByteArray.size_extract, Nat.min_self]
    have hex : p + q + s.need ≤ src.size →
        (src.extract p src.size).extract q (q + s.need) = src.extract (p + q) (p + q + s.need) := by
      intro hi
      rw [ByteArray.extract_extract, Nat.min_eq_left (by omega), Nat.add_assoc]
    have ih := firstOk_shift hp ss (q + s.need)
    have hpos : p + q + s.need = p + (q + s.need) := Nat.add_assoc _ _ _
    constructor
    · intro h
      have h' : p + q + s.need ≤ src.size ∧ (∃ D, s.run (src.extract (p + q) (p + q + s.need)) = some D) ∧
          FirstOk ss src (p + q + s.need) := h
      obtain ⟨h1, ⟨D, hD⟩, h3⟩ := h'
      have h0 : q + s.need ≤ (src.extract p src.size).size := by omega
      have hD' : s.run ((src.extract p src.size).extract q (q + s.need)) = some D := by
        rw [hex h1]
        exact hD
      have h3' : FirstOk ss src (p + (q + s.need)) := by
        rw [← hpos]
        exact h3
      show q + s.need ≤ (src.extract p src.size).size ∧
        (∃ D, s.run ((src.extract p src.size).extract q (q + s.need)) = some D) ∧
        FirstOk ss (src.extract p src.size) (q + s.need)
      exact ⟨h0, ⟨D, hD'⟩, ih.1 h3'⟩
    · intro h
      have h' : q + s.need ≤ (src.extract p src.size).size ∧
          (∃ D, s.run ((src.extract p src.size).extract q (q + s.need)) = some D) ∧
          FirstOk ss (src.extract p src.size) (q + s.need) := h
      obtain ⟨h0, ⟨D, hD⟩, h3⟩ := h'
      have h1 : p + q + s.need ≤ src.size := by omega
      have hD' : s.run (src.extract (p + q) (p + q + s.need)) = some D := by
        rw [← hex h1]
        exact hD
      have h3' : FirstOk ss src (p + q + s.need) := by
        rw [hpos]
        exact ih.2 h3
      show p + q + s.need ≤ src.size ∧ (∃ D, s.run (src.extract (p + q) (p + q + s.need)) = some D) ∧
        FirstOk ss src (p + q + s.need)
      exact ⟨h1, ⟨D, hD'⟩, h3'⟩

/-- A stage's first read fails with probability at most `q`, on a uniform stream that holds it. -/
def Stage.FirstBounded (s : Stage) (q : ℝ≥0∞) : Prop :=
  ∀ L, s.need ≤ L → prCoin (fun f : Fin L → Fin 256 => s.run ((Law.streamOf f).extract 0 s.need) = none) ≤ q

/-- **Some stage's first read fails with probability at most the stages' bounds added**: on a uniform stream that holds
the stages' first reads, stage `i`'s failing with probability at most `wt i · 2⁻⁶⁴`. -/
theorem firstOk_fail_le {ι : Type} (stg : ι → Stage) (wt : ι → ℕ) :
    ∀ (l : List ι), (∀ i ∈ l, (stg i).FirstBounded ((wt i : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹)) →
      ∀ L, ((l.map stg).map (·.need)).sum ≤ L →
        prCoin (fun f : Fin L → Fin 256 => ¬ FirstOk (l.map stg) (Law.streamOf f) 0) ≤
          (((l.map wt).sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹
  | [], _, _, _ => by
    refine le_trans (prCoin_mono (q := fun _ => False) fun f hf => hf trivial) ?_
    rw [prCoin_false]
    exact zero_le
  | i :: l, hb, L, hL => by
    have hs : (stg i).FirstBounded ((wt i : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹) := hb i List.mem_cons_self
    have ih := firstOk_fail_le stg wt l fun j hj => hb j (List.mem_cons_of_mem _ hj)
    have hL' : (stg i).need + ((l.map stg).map (·.need)).sum ≤ L := hL
    obtain ⟨ℓ, rfl⟩ : ∃ ℓ, L = (stg i).need + ℓ := ⟨L - (stg i).need, by omega⟩
    have hℓ : ((l.map stg).map (·.need)).sum ≤ ℓ := by omega
    have hor : ∀ f : Fin ((stg i).need + ℓ) → Fin 256, ¬ FirstOk ((i :: l).map stg) (Law.streamOf f) 0 →
        (stg i).run ((Law.streamOf f).extract 0 (stg i).need) = none ∨
          ¬ FirstOk (l.map stg) (Law.streamOf f) (stg i).need := by
      intro f hf
      by_cases h1 : (stg i).run ((Law.streamOf f).extract 0 (stg i).need) = none
      · exact Or.inl h1
      · obtain ⟨D, hD0⟩ := Option.ne_none_iff_exists'.1 h1
        refine Or.inr fun h2 => hf ?_
        have hsz : 0 + (stg i).need ≤ (Law.streamOf f).size := by
          rw [size_streamOf]
          omega
        have hD : (stg i).run ((Law.streamOf f).extract 0 (0 + (stg i).need)) = some D := by
          rw [Nat.zero_add]
          exact hD0
        have h2' : FirstOk (l.map stg) (Law.streamOf f) (0 + (stg i).need) := by
          rw [Nat.zero_add]
          exact h2
        show 0 + (stg i).need ≤ (Law.streamOf f).size ∧
          (∃ D, (stg i).run ((Law.streamOf f).extract 0 (0 + (stg i).need)) = some D) ∧
          FirstOk (l.map stg) (Law.streamOf f) (0 + (stg i).need)
        exact ⟨hsz, ⟨D, hD⟩, h2'⟩
    have hP1 : prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 =>
        (stg i).run ((Law.streamOf f).extract 0 (stg i).need) = none) ≤ (wt i : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ :=
      hs ((stg i).need + ℓ) (Nat.le_add_right _ _)
    have hP2 : prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 =>
        ¬ FirstOk (l.map stg) (Law.streamOf f) (stg i).need) ≤
        (((l.map wt).sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
      have happ := prCoin_append (w := (stg i).need) (ℓ := ℓ)
        (fun src => ¬ FirstOk (l.map stg) src (stg i).need) (fun _ => True)
        (fun b => ¬ FirstOk (l.map stg) b 0) (fun a b ha => by
          show ¬ FirstOk (l.map stg) (a ++ b) (stg i).need ↔ True ∧ ¬ FirstOk (l.map stg) b 0
          have hp : (stg i).need ≤ (a ++ b).size := by
            rw [ByteArray.size_append]
            omega
          have hsh := firstOk_shift hp (l.map stg) 0
          have hext : (a ++ b).extract (stg i).need (a ++ b).size = b :=
            ByteArray.extract_append_eq_right ha.symm ByteArray.size_append
          rw [hext] at hsh
          exact ⟨fun h => ⟨trivial, fun hb' => h (hsh.2 hb')⟩, fun h hab => h.2 (hsh.1 hab)⟩)
      refine le_trans (le_of_eq happ) ?_
      refine le_trans (mul_le_mul' (prCoin_le_one _) (ih ℓ hℓ)) ?_
      exact le_of_eq (one_mul _)
    calc prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 => ¬ FirstOk ((i :: l).map stg) (Law.streamOf f) 0)
        ≤ prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 =>
            (stg i).run ((Law.streamOf f).extract 0 (stg i).need) = none ∨
              ¬ FirstOk (l.map stg) (Law.streamOf f) (stg i).need) := prCoin_mono hor
      _ ≤ prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 =>
            (stg i).run ((Law.streamOf f).extract 0 (stg i).need) = none) +
          prCoin (fun f : Fin ((stg i).need + ℓ) → Fin 256 =>
            ¬ FirstOk (l.map stg) (Law.streamOf f) (stg i).need) := Verity.Game.prCoin_or_le _ _
      _ ≤ (wt i : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ + (((l.map wt).sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ :=
          add_le_add hP1 hP2
      _ = ((((i :: l).map wt).sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
          rw [List.map_cons, List.sum_cons, Nat.cast_add, add_mul]

/-- **A stratum's first read fails with probability at most `k_s·2⁻⁶⁴`**: its `subset` on its first `need` bytes
(`Law.subset_prefix_none_le`). -/
theorem stage_firstBounded (s : Flock.Draw.Stratum) (hk : s.k ≤ s.size) (miss : Array ℕ → Prop) :
    (stage s miss).FirstBounded ((s.k : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹) := by
  intro L hL
  have hN : s.k * Flock.Draw.width s.size ≤ 2 * s.k * Flock.Draw.width s.size + 64 := by
    rw [Nat.mul_assoc]
    omega
  exact Law.subset_prefix_none_le s.size s.k hk (2 * s.k * Flock.Draw.width s.size + 64) L hN hL

/-! ## Where every first read samples, the sample ends -/

/-- **`onWith` on `take` returns on its first read** when its sampler samples on it: the `N` bytes from `p`. -/
theorem onWith_take_first (src : ByteArray) (N : ℕ) (run : ByteArray → Option (Array ℕ)) (p : ℕ) (D : Array ℕ)
    (hle : p + N ≤ src.size) (hrun : run (src.extract p (p + N)) = some D) :
    (Flock.Draw.onWith (fun need => _root_.Spot.Draw.take src need) N run).run p = .ok (some D, p + N) := by
  have hnot : ¬ src.size < p + N := by omega
  have hfuel : (onWithFuel (fun need => _root_.Spot.Draw.take src need) N run (0 + 1)
      (src.extract p (p + N))).run (p + N) = .ok (some D, p + N) :=
    (congrArg (fun x : StateT ℕ (Except String) (Option (Array ℕ)) => x.run (p + N))
      (Law.onWithFuel_succ_some (fun need => _root_.Spot.Draw.take src need) N run 0 _ D hrun)).trans rfl
  rw [onWith_fuel (fun need => _root_.Spot.Draw.take src need) N run (0 + 1), StateT.run_bind, take_run,
    ite_eq_right hnot, bind_ok]
  exact hfuel

/-- **`stratifiedWith` on `take` returns a sample** where every stratum's first read samples: each stratum reads its
`need` bytes once, in turn from `p`. -/
theorem stratifiedWith_take_first (src : ByteArray) (strata : Array Flock.Draw.Stratum) (p : ℕ)
    (miss : ℕ → Array ℕ → Prop)
    (h : FirstOk ((List.range' 0 strata.size).map fun i => stage strata[i]! (miss i)) src p) :
    ∃ D p', (Flock.Draw.stratifiedWith (fun _ need => _root_.Spot.Draw.take src need) strata).run p =
      .ok (some D, p') := by
  have loop : ∀ (l : List ℕ) (out : Array ℕ) (pos : ℕ),
      FirstOk (l.map fun i => stage strata[i]! (miss i)) src pos →
      ∃ out' pos₁,
        (forIn (m := StateT ℕ (Except String)) l ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (fun need => _root_.Spot.Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = .ok ((none, out'), pos₁) := by
    intro l
    induction l with
    | nil =>
      intro out pos _
      exact ⟨out, pos, rfl⟩
    | cons i is ih =>
      intro out pos hok
      have hok' : pos + strata[i]!.need ≤ src.size ∧
          (∃ D, Flock.Draw.subset strata[i]!.k strata[i]!.size
            (src.extract pos (pos + strata[i]!.need)) = some D) ∧
          FirstOk (is.map fun i => stage strata[i]! (miss i)) src (pos + strata[i]!.need) := hok
      obtain ⟨hle, ⟨D, hD⟩, hrest⟩ := hok'
      have ho := onWith_take_first src strata[i]!.need (Flock.Draw.subset strata[i]!.k strata[i]!.size) pos D hle hD
      obtain ⟨out', pos₁, hih⟩ := ih (out ++ Array.map strata[i]!.unit D) (pos + strata[i]!.need) hrest
      refine ⟨out', pos₁, ?_⟩
      rw [List.forIn_cons, StateT.run_bind, StateT.run_bind, ho, bind_ok]
      dsimp only
      rw [StateT.run_pure, pure_bind]
      dsimp only
      exact hih
  obtain ⟨out', pos₁, hl⟩ := loop (List.range' 0 strata.size) (#[] : Array ℕ) p h
  have key : ∀ r, (Flock.Draw.stratifiedWith (fun _ need => _root_.Spot.Draw.take src need) strata).run p = r →
      ∃ D p', r = .ok (some D, p') := by
    intro r hr
    unfold Flock.Draw.stratifiedWith at hr
    dsimp only at hr
    rw [Std.Legacy.Range.forIn_eq_forIn_range'] at hr
    have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
      show (strata.size - 0 + 1 - 1) / 1 = _
      omega
    rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
      show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at hr
    rw [StateT.run_bind] at hr
    cases hl' : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
        ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run p with
    | error e =>
      rw [hl'] at hr
      exact absurd (hl'.symm.trans hl) (by simp)
    | ok y =>
      rw [hl', bind_ok] at hr
      have hy : y = ((none, out'), pos₁) := Except.ok.inj (hl'.symm.trans hl)
      subst hy
      dsimp only at hr
      rw [StateT.run_pure] at hr
      exact ⟨_, _, hr.symm⟩
  exact key _ rfl

/-- **`drawUnits` returns a sample** where every stage's first read samples. -/
theorem units_ok_of_firstOk (law : Flock.Draw.Law) (n : ℕ) (src : ByteArray) (miss : ℕ → Array ℕ → Prop)
    (h : FirstOk (lawStages law n miss) src 0) :
    ∃ units read, _root_.Spot.Draw.drawUnits law n src = .ok (units, read) := by
  have key : ∀ r, (Flock.Draw.drawWith (fun _ need => _root_.Spot.Draw.take src need) law n).run 0 = r →
      ∃ d read, r = .ok (some d, read) := by
    intro r hr
    cases law with
    | subset k =>
      have h' : 0 + (2 * k * Flock.Draw.width n + 64) ≤ src.size ∧
          (∃ D, Flock.Draw.subset k n (src.extract 0 (0 + (2 * k * Flock.Draw.width n + 64))) = some D) ∧
          True := h
      obtain ⟨hle, ⟨D, hD⟩, -⟩ := h'
      have ho := onWith_take_first src (2 * k * Flock.Draw.width n + 64) (Flock.Draw.subset k n) 0 D hle hD
      unfold Flock.Draw.drawWith at hr
      dsimp only at hr
      rw [StateT.run_bind, ho, bind_ok, StateT.run_pure] at hr
      exact ⟨_, _, hr.symm⟩
    | bernoulli a b =>
      have h' : 0 + (2 * n * Flock.Draw.width b + 64) ≤ src.size ∧
          (∃ D, Flock.Draw.bernoulli a b n (src.extract 0 (0 + (2 * n * Flock.Draw.width b + 64))) = some D) ∧
          True := h
      obtain ⟨hle, ⟨D, hD⟩, -⟩ := h'
      have ho := onWith_take_first src (2 * n * Flock.Draw.width b + 64) (Flock.Draw.bernoulli a b n) 0 D hle hD
      unfold Flock.Draw.drawWith at hr
      dsimp only at hr
      rw [StateT.run_bind, ho, bind_ok, StateT.run_pure] at hr
      exact ⟨_, _, hr.symm⟩
    | stratified K St =>
      obtain ⟨_, _, ho⟩ := stratifiedWith_take_first src St 0 miss h
      unfold Flock.Draw.drawWith at hr
      dsimp only at hr
      rw [StateT.run_bind, ho, bind_ok, StateT.run_pure] at hr
      exact ⟨_, _, hr.symm⟩
    | work K ws =>
      obtain ⟨_, _, ho⟩ := stratifiedWith_take_first src (ws.map fun w : Flock.Draw.WorkStratum => w.toStratum) 0
        miss h
      unfold Flock.Draw.drawWith at hr
      dsimp only at hr
      rw [StateT.run_bind, ho, bind_ok, StateT.run_pure] at hr
      exact ⟨_, _, hr.symm⟩
  obtain ⟨d, read, hd⟩ := key _ rfl
  refine ⟨d.units, read, ?_⟩
  unfold _root_.Spot.Draw.drawUnits
  rw [hd]
  rfl

end Properties.Spot.SecurityProofs

namespace Flock.Bridge

open FlockSoundness FlockSoundness.Game Finset
open scoped ENNReal

/-! ## The deployed sample at a sampling design on strata -/

/-- **The bytes the deployed sample's stages read first, in all**: `2·k·width n + 64` for a subset sample, each
stratum's `need` added for a stratified or work sample. -/
def firstRead (law : Flock.Draw.Law) (n : ℕ) : ℕ :=
  ((Properties.Spot.SecurityProofs.lawStages law n fun _ _ => True).map (·.need)).sum

/-- The strata's `k`s by index are the strata's `k`s. -/
theorem range_map_k (St : Array Flock.Draw.Stratum) :
    (List.range' 0 St.size).map (fun i => St[i]!.k) = St.toList.map (·.k) := by
  refine List.ext_getElem (by simp) fun i h₁ _ => ?_
  have hi : i < St.size := by simpa using h₁
  simp [getElem!_pos St i hi]

/-- A stratum by an index of the strata is one of them. -/
theorem stratum_mem_toList (St : Array Flock.Draw.Stratum) {i : ℕ} (hi : i ∈ List.range' 0 St.size) :
    St[i]! ∈ St.toList := by
  have hlt : i < St.size := by
    have := List.mem_range'_1.1 hi
    omega
  rw [getElem!_pos St i hlt]
  exact Array.mem_toList_iff.2 (Array.getElem_mem hlt)

/-- **Some stratum's first read fails with probability at most `(Σ_s k_s)·2⁻⁶⁴`**, on a uniform stream that holds the
strata's first reads, each stratum sampling at most its size. -/
theorem strata_fail_le (St : Array Flock.Draw.Stratum) (hks : ∀ s ∈ St.toList, s.k ≤ s.size) (L : ℕ)
    (hL : (((List.range' 0 St.size).map fun i => Properties.Spot.SecurityProofs.stage St[i]! fun _ => True).map
      (·.need)).sum ≤ L) :
    prCoin (fun f : Fin L → Fin 256 => ¬ Properties.Spot.SecurityProofs.FirstOk
      ((List.range' 0 St.size).map fun i => Properties.Spot.SecurityProofs.stage St[i]! fun _ => True)
      (Audit.Law.streamOf f) 0) ≤ (((St.toList.map (·.k)).sum : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  have hb := Properties.Spot.SecurityProofs.firstOk_fail_le
    (fun i : ℕ => Properties.Spot.SecurityProofs.stage St[i]! fun _ => True) (fun i : ℕ => St[i]!.k)
    (List.range' 0 St.size)
    (fun i hi => Properties.Spot.SecurityProofs.stage_firstBounded St[i]! (hks _ (stratum_mem_toList St hi)) _)
    L hL
  rw [range_map_k St] at hb
  exact hb

/-- **The deployed sample returns no sample with probability at most `kd·2⁻⁶⁴`**, under a sampling design on strata
that samples `kd` units, on a uniform stream that holds its stages' first reads: it ends where every stage's first read
samples (`units_ok_of_firstOk`), and stage `s`'s fails with probability at most `k_s·2⁻⁶⁴`. -/
theorem units_error_le {law : Flock.Draw.Law} {ns : List ℕ} {kd n : ℕ} (hlaw : LawOn law ns kd)
    (hns : ns.sum = n) (L : ℕ) (hL : firstRead law n ≤ L) :
    prCoin (fun f : Fin L → Fin 256 => ¬ ∃ units read,
      Spot.Draw.drawUnits law n (Properties.Spot.bytesOf f) = .ok (units, read)) ≤
      (kd : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  refine le_trans (prCoin_mono (q := fun f : Fin L → Fin 256 =>
    ¬ Properties.Spot.SecurityProofs.FirstOk (Properties.Spot.SecurityProofs.lawStages law n fun _ _ => True)
      (Audit.Law.streamOf f) 0) fun f hf hok =>
        hf (Properties.Spot.SecurityProofs.units_ok_of_firstOk law n _ _ hok)) ?_
  cases law with
  | subset k =>
    obtain ⟨hkk, hk⟩ := hlaw
    have hk' : k ≤ n := by omega
    have hb := Properties.Spot.SecurityProofs.firstOk_fail_le
      (fun _ : ℕ => (⟨2 * k * Flock.Draw.width n + 64, Flock.Draw.subset k n, fun _ => True⟩ :
        Properties.Spot.SecurityProofs.Stage))
      (fun _ => k) [0]
      (fun _ _ => by
        intro L' hL'
        exact Audit.Law.subset_prefix_none_le n k hk' (2 * k * Flock.Draw.width n + 64) L'
          (by rw [Nat.mul_assoc]; omega) hL') L hL
    have hsum1 : ([0].map fun _ : ℕ => k).sum = k := by simp
    rw [hsum1] at hb
    rw [← hkk]
    exact hb
  | bernoulli a b => exact hlaw.elim
  | stratified K St =>
    obtain ⟨⟨-, hsz⟩, hK, hn1, hk, hsum⟩ := hlaw
    have hks : ∀ s ∈ St.toList, s.k ≤ s.size := fun s hs => by
      have hm : s.size ∈ ns := hsz ▸ List.mem_map_of_mem hs
      rw [hk s hs]
      exact stratumK_le hK (hn1 _ hm) (List.le_sum_of_mem hm)
    have hb := strata_fail_le St hks L hL
    rw [hsum] at hb
    exact hb
  | work K ws =>
    obtain ⟨-, hk, hsum⟩ := hlaw
    have hks : ∀ s ∈ (ws.map fun w : Flock.Draw.WorkStratum => w.toStratum).toList, s.k ≤ s.size := fun s hs => by
      rw [Array.toList_map] at hs
      obtain ⟨w, hw, rfl⟩ := List.mem_map.1 hs
      show w.k ≤ w.size
      rw [hk w hw]
      exact min_le_left _ _
    have hsum' : ((ws.map fun w : Flock.Draw.WorkStratum => w.toStratum).toList.map (·.k)).sum = kd := by
      rw [Array.toList_map, List.map_map]
      exact hsum
    have hb := strata_fail_le (ws.map fun w : Flock.Draw.WorkStratum => w.toStratum) hks L hL
    rw [hsum'] at hb
    exact hb

/-- **The deployed sample runs out with probability at most `kd·2⁻⁶⁴`** on a uniform byte source (A3) that holds its
stages' first reads. -/
theorem execSel_error_le {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) (ht : Assumptions.UniformRandomBytes L t) {law : Flock.Draw.Law}
    {ns : List ℕ} {kd : ℕ} (hlaw : LawOn law ns kd) (hns : ns.sum = n) (hL : firstRead law n ≤ L 0) :
    prCoin (fun ω => ¬ ∃ units read,
      Spot.Draw.drawUnits law n (Properties.Spot.bytesOf (t ω 0)) = .ok (units, read)) ≤
      (kd : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹ := by
  classical
  let E : (i : Fin 1) → (Fin (L i) → Fin 256) → Prop := fun _ g => ¬ ∃ units read,
    Spot.Draw.drawUnits law n (Properties.Spot.bytesOf g) = .ok (units, read)
  calc prCoin (fun ω => E 0 (t ω 0))
      = prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => E 0 (f 0)) := ht fun f => E 0 (f 0)
    _ ≤ prCoin (fun f : (i : Fin 1) → Fin (L i) → Fin 256 => ∀ s ∈ (univ : Finset (Fin 1)), E s (f s)) :=
        prCoin_mono fun f h s _ => by obtain rfl : s = 0 := Subsingleton.elim _ _; exact h
    _ = ∏ s : Fin 1, prCoin (E s) := prCoin_pi_forall (Ω := fun s => Fin (L s) → Fin 256) univ E
    _ = prCoin (E 0) := Fin.prod_univ_one _
    _ ≤ _ := units_error_le hlaw hns (L 0) hL

/-- **A deployed sample that ends holds at most `kd` proof blocks**: the units of `[0, n)` among the `kd` it returns
(`drawUnits_size`). -/
theorem execSel_card_le {n : ℕ} {Ω : Type} [Fintype Ω] [Nonempty Ω] {L : Fin 1 → ℕ}
    (t : Ω → (i : Fin 1) → Fin (L i) → Fin 256) {law : Flock.Draw.Law} {ns : List ℕ} {kd : ℕ}
    (hlaw : LawOn law ns kd) (ω : Ω)
    (h : ∃ units read, Spot.Draw.drawUnits law n (Properties.Spot.bytesOf (t ω 0)) = .ok (units, read)) :
    ((execSel (n := n) t law n).draw ω).card ≤ kd := by
  obtain ⟨units, read, hu⟩ := h
  have hd : (execSel (n := n) t law n).draw ω = univ.filter fun u : Fin n => u.val ∈ units := by
    change (match Spot.Draw.drawUnits law n (Properties.Spot.bytesOf (t ω 0)) with
      | .ok r => univ.filter fun u : Fin n => u.val ∈ r.1
      | .error _ => univ) = univ.filter fun u : Fin n => u.val ∈ units
    rw [hu]
  rw [hd]
  exact (Audit.Law.card_filter_mem_le n units).trans (le_of_eq (drawUnits_size hlaw hu))

end Flock.Bridge
