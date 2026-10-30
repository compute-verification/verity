import Proofs.Flock.EndToEnd.Served.Draw
import Proofs.Flock.Claims.Sampling.Select.Take

/-!
# The served stratified draw is the live draw's on the served tapes

`SampledProofs.Draw.drawUnits` reads one stream of coins, every stratum from where the one before stopped;
`Law.runOS` reads stratum `i` from tape `i`. On the served tapes (`Served.tapes`) the two draw the same units.

* `Sim`: a code run (`Draw.take`, one position) that ends at `p'` is the model run on stream `i` (`Law.tapeGet`) whenever
  tape `i`'s first bytes are the code's bytes from `base` to `p'` (`Agree`). `sim_onWith` is `onWith`'s.
* `used_of_stopsAt`: a stratum that stops at its `j`-th read (`StopsAt`) reads `j · need` bytes, so the code's
  strata start at `Served.starts`.
* `stratified_sim`: a stratified code run that draws `D` from position `0` is a model run that draws `D`, on tapes whose
  first `seg i` bytes are each stratum's segment.
* `drawUnits_runOS`: on the served tapes, `runOS` draws the units `drawUnits` draws on the coins.
-/

namespace FlockSoundness.Audit.Served

open Classical
open SampledProofs Properties.SampledProofs.SecurityProofs

theorem used_le (s : Flock.Draw.Stratum) (a : ByteArray) : used s a ≤ a.size := by
  unfold used
  split
  · rename_i h; exact (Nat.find_spec h).2.1
  · exact le_rfl

theorem extract_from (src : ByteArray) (p q : ℕ) (hq : p + q ≤ src.size) :
    (src.extract p src.size).extract 0 q = src.extract p (p + q) := by
  rw [ByteArray.extract_extract, Nat.add_zero, Nat.min_eq_left hq]

theorem size_extract_from (src : ByteArray) (p : ℕ) : (src.extract p src.size).size = src.size - p := by
  rw [ByteArray.size_extract, Nat.min_self]

/-- **A stratum that stops at its `j`-th read reads `j · need` bytes.** -/
theorem used_of_stopsAt (s : Flock.Draw.Stratum) (src : ByteArray) (p j : ℕ) (D : Array ℕ)
    (h : StopsAt s.need (Flock.Draw.subset s.k s.size) src p j D) :
    used s (src.extract p src.size) = j * s.need := by
  obtain ⟨hj, hle, hrun, hnone⟩ := h
  have hsz := size_extract_from src p
  have hd : Draws s (src.extract p src.size) j := ⟨hj, by omega, by rw [extract_from src p _ hle, hrun]; rfl⟩
  have hex : ∃ j, Draws s (src.extract p src.size) j := ⟨j, hd⟩
  unfold used
  rw [dite_eq_left hex]
  congr 1
  rw [Nat.find_eq_iff]
  refine ⟨hd, fun i hi hdi => ?_⟩
  obtain ⟨hi1, hi2, hi3⟩ := hdi
  have hle' : p + i * s.need ≤ src.size := by omega
  rw [extract_from src p _ hle', hnone i hi1 hi] at hi3
  exact absurd hi3 (by simp)

theorem starts_succ (strata : Array Flock.Draw.Stratum) (src : ByteArray) (i : ℕ) :
    starts strata src (i + 1) = starts strata src i + seg strata src i := rfl

theorem seg_le (strata : Array Flock.Draw.Stratum) (src : ByteArray) (i : ℕ) :
    seg strata src i ≤ src.size - starts strata src i := by
  have h := used_le strata[i]! (src.extract (starts strata src i) src.size)
  rw [size_extract_from] at h
  exact h

theorem starts_le (strata : Array Flock.Draw.Stratum) (src : ByteArray) : ∀ i, starts strata src i ≤ src.size
  | 0 => Nat.zero_le _
  | i + 1 => by
    have h1 := starts_le strata src i
    have h2 := seg_le strata src i
    rw [starts_succ]
    omega

/-- `src`'s bytes from `base` to `q` are the first bytes of tape `i`. -/
def Agree (src : ByteArray) (T : ℕ → ByteArray) (i base q : ℕ) : Prop :=
  base ≤ q ∧ q ≤ src.size ∧ q - base ≤ (T i).size ∧ (T i).extract 0 (q - base) = src.extract base q

theorem Agree.mono {src : ByteArray} {T : ℕ → ByteArray} {i base q : ℕ} (h : Agree src T i base q) {q' : ℕ}
    (h1 : base ≤ q') (h2 : q' ≤ q) : Agree src T i base q' := by
  obtain ⟨hb, hq, hT, he⟩ := h
  refine ⟨h1, by omega, by omega, ?_⟩
  have := congrArg (fun a : ByteArray => a.extract 0 (q' - base)) he
  simp only [ByteArray.extract_extract, Nat.zero_add, Nat.add_zero] at this
  rw [Nat.min_eq_left (by omega : q' - base ≤ q - base), show base + (q' - base) = q' by omega,
    Nat.min_eq_left h2] at this
  exact this

/-- **A code run is a model run on stream `i`**: a code run (`Draw.take` on `src`) from `p` that ends at `p'`, reading
no byte before `p`, is the model's from any state whose stream `i` sits at `p − base`, ending with stream `i` at
`p' − base`, whenever tape `i` begins with `src`'s bytes from `base` to `p'`. -/
def Sim (src : ByteArray) (T : ℕ → ByteArray) (i base : ℕ) {α : Type} (x₁ : StateT ℕ (Except String) α)
    (x₂ : StateT (ℕ → ℕ) Option α) : Prop :=
  ∀ p a p', x₁.run p = .ok (a, p') → p ≤ p' ∧ ∀ pos : ℕ → ℕ, pos i + base = p → Agree src T i base p' →
    x₂.run pos = some (a, Function.update pos i (p' - base))

section Sim

variable {src : ByteArray} {T : ℕ → ByteArray} {i base : ℕ}

theorem sim_pure {α : Type} (a : α) :
    Sim src T i base (pure a : StateT ℕ (Except String) α) (pure a : StateT (ℕ → ℕ) Option α) := by
  intro p b p' h
  rw [StateT.run_pure] at h
  obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
  refine ⟨le_rfl, fun pos hpos _ => ?_⟩
  rw [StateT.run_pure, show p - base = pos i by omega, Function.update_eq_self]
  rfl

theorem sim_bind {α β : Type} {x₁ : StateT ℕ (Except String) α} {x₂ : StateT (ℕ → ℕ) Option α}
    {g₁ : α → StateT ℕ (Except String) β} {g₂ : α → StateT (ℕ → ℕ) Option β}
    (hx : Sim src T i base x₁ x₂) (hg : ∀ a, Sim src T i base (g₁ a) (g₂ a)) :
    Sim src T i base (x₁ >>= g₁) (x₂ >>= g₂) := by
  intro p b p' h
  rw [StateT.run_bind] at h
  cases h₁ : x₁.run p with
  | error e => rw [h₁] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨a, q⟩ := y
    rw [h₁, bind_ok] at h
    dsimp only at h
    obtain ⟨hpq, hx'⟩ := hx p a q h₁
    obtain ⟨hqp', hg'⟩ := hg a q b p' h
    refine ⟨le_trans hpq hqp', fun pos hpos hag => ?_⟩
    rw [StateT.run_bind, hx' pos hpos (hag.mono (by omega) hqp')]
    show (g₂ a).run (Function.update pos i (q - base)) = _
    rw [hg' _ (by rw [Function.update_self]; omega) hag, Function.update_idem]

theorem sim_take (need : ℕ) : Sim src T i base (Draw.take src need) (Law.tapeGet T i need) := by
  intro p a p' h
  rw [take_run] at h
  split at h
  · exact absurd h (by simp)
  · obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
    refine ⟨Nat.le_add_right _ _, fun pos hpos hag => ?_⟩
    obtain ⟨hb, hq, hTs, he⟩ := hag
    show (if pos i + need ≤ (T i).size then
      some ((T i).extract (pos i) (pos i + need), Function.update pos i (pos i + need)) else none) = _
    rw [ite_eq_left (by omega)]
    have hpi : pos i = p - base := by omega
    have hx := congrArg (fun a : ByteArray => a.extract (p - base) (p + need - base)) he
    simp only [ByteArray.extract_extract, Nat.zero_add] at hx
    rw [Nat.min_self, show base + (p - base) = p by omega, show base + (p + need - base) = p + need by omega,
      Nat.min_self] at hx
    rw [hpi, show p - base + need = p + need - base by omega, hx]

/-- A `repeat` loop whose bodies simulate, each code `yield` lowering a measure of the position, simulates. -/
theorem sim_loop {β : Type} (f₁ : Unit → β → StateT ℕ (Except String) (ForInStep β))
    (f₂ : Unit → β → StateT (ℕ → ℕ) Option (ForInStep β)) (μ : ℕ → ℕ) (hf : ∀ b, Sim src T i base (f₁ () b) (f₂ () b))
    (hμ : ∀ b p b' p', (f₁ () b).run p = .ok (.yield b', p') → μ p' < μ p) (b : β) :
    Sim src T i base (Lean.Loop.forIn Lean.Loop.mk b f₁) (Lean.Loop.forIn Lean.Loop.mk b f₂) := by
  suffices h : ∀ N b p, μ p < N → ∀ a p', (Lean.Loop.forIn Lean.Loop.mk b f₁).run p = .ok (a, p') →
      p ≤ p' ∧ ∀ pos : ℕ → ℕ, pos i + base = p → Agree src T i base p' →
        (Lean.Loop.forIn Lean.Loop.mk b f₂).run pos = some (a, Function.update pos i (p' - base)) from
    fun p a p' hr => h _ b p (Nat.lt_succ_self _) a p' hr
  intro N
  induction N with
  | zero => intro b p h; omega
  | succ N ih =>
    intro b p hN a p' h
    rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind] at h
    cases h₁ : (f₁ () b).run p with
    | error e => rw [h₁] at h; exact absurd h (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨step, q⟩ := y
      rw [h₁, bind_ok] at h
      obtain ⟨hpq, hsim⟩ := hf b p step q h₁
      cases step with
      | done b' =>
        change (pure b' : StateT ℕ (Except String) β).run q = .ok (a, p') at h
        rw [StateT.run_pure] at h
        obtain ⟨rfl, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
        refine ⟨hpq, fun pos hpos hag => ?_⟩
        rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind, hsim pos hpos hag]
        rfl
      | yield b' =>
        change (Lean.Loop.forIn Lean.Loop.mk b' f₁).run q = .ok (a, p') at h
        have hq := hμ b p b' q h₁
        obtain ⟨hqp', hrest⟩ := ih b' q (by omega) a p' h
        refine ⟨le_trans hpq hqp', fun pos hpos hag => ?_⟩
        rw [Lean.Loop.forIn_eq_of_monadTail, StateT.run_bind, hsim pos hpos (hag.mono (by omega) hqp')]
        show (Lean.Loop.forIn Lean.Loop.mk b' f₂).run (Function.update pos i (q - base)) = _
        rw [hrest _ (by rw [Function.update_self]; omega) hag, Function.update_idem]

/-- **`onWith` on the code's reader is `onWith` on stream `i` of the tapes**, up to where the code stopped. -/
theorem sim_onWith (N : ℕ) (hN : 0 < N) (run : ByteArray → Option (Array ℕ)) :
    Sim src T i base (Flock.Draw.onWith (fun need => Draw.take src need) N run)
      (Flock.Draw.onWith (Law.tapeGet T i) N run) := by
  unfold Flock.Draw.onWith
  refine sim_bind (sim_take N) fun s₀ =>
    sim_bind (sim_loop _ _ (fun p => src.size + 1 - p) (fun b => ?_) ?_ _) fun r => ?_
  · obtain ⟨r, s⟩ := b
    try dsimp only
    cases hrun : run s with
    | some units => exact sim_pure _
    | none => exact sim_bind (sim_take N) fun x => sim_pure _
  · intro b p b' p' h
    obtain ⟨r, s⟩ := b
    cases hrun : run s with
    | some units =>
      simp only [hrun, StateT.run_pure] at h
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj h)
      exact absurd h1 (by simp)
    | none =>
      simp only [hrun] at h
      rw [StateT.run_bind, take_run] at h
      split at h
      · exact absurd h (by rw [bind_error]; simp)
      · rw [bind_ok, StateT.run_pure] at h
        obtain ⟨_, rfl⟩ := Prod.mk.inj (Except.ok.inj h)
        omega
  · obtain ⟨r, s⟩ := r
    cases r <;> exact sim_pure _

end Sim

/-- **A stratified code run is a model run on tapes that begin with each stratum's segment**: a run of `stratifiedWith`
on the code's reader from position `0` that draws `D` draws `D` on the tapes `T`, from unread streams, when tape `i`
begins with the `seg i` bytes stratum `i` reads from `starts i`. -/
theorem stratified_sim (strata : Array Flock.Draw.Stratum) (src : ByteArray) (T : ℕ → ByteArray)
    (hT : ∀ x < strata.size, seg strata src x ≤ (T x).size ∧
      (T x).extract 0 (seg strata src x) =
        src.extract (starts strata src x) (starts strata src x + seg strata src x))
    (D : Array ℕ) (p' : ℕ)
    (h : (Flock.Draw.stratifiedWith (fun _ need => Draw.take src need) strata).run 0 = .ok (some D, p')) :
    ∃ pos', (Flock.Draw.stratifiedWith (Law.tapeGet T) strata).run (fun _ => 0) = some (some D, pos') := by
  unfold Flock.Draw.stratifiedWith at h ⊢
  dsimp only at h ⊢
  rw [Std.Legacy.Range.forIn_eq_forIn_range'] at h ⊢
  have hsize : ([:strata.size] : Std.Legacy.Range).size = strata.size := by
    show (strata.size - 0 + 1 - 1) / 1 = _
    omega
  rw [hsize, show ([:strata.size] : Std.Legacy.Range).start = 0 from rfl,
    show ([:strata.size] : Std.Legacy.Range).step = 1 from rfl] at h ⊢
  have loop : ∀ (n i : ℕ), i + n ≤ strata.size → ∀ (out : Array ℕ) (pos : ℕ → ℕ), (∀ x, i ≤ x → pos x = 0) →
      ∀ (r : Option (Option (Array ℕ))) (out' : Array ℕ) (p₁ : ℕ),
      (forIn (m := StateT ℕ (Except String)) (List.range' i n) ((none : Option (Option (Array ℕ))), out)
        (fun i __s => do
          let __x ← Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run (starts strata src i) = .ok ((r, out'), p₁) →
      r = none ∧ ∃ pos₁, (forIn (m := StateT (ℕ → ℕ) Option) (List.range' i n)
        ((none : Option (Option (Array ℕ))), out) (fun i __s => do
          let __x ← Flock.Draw.onWith (Law.tapeGet T i) strata[i]!.need
            (Flock.Draw.subset strata[i]!.k strata[i]!.size)
          match __x with
          | some local_ => pure (ForInStep.yield (none, __s.2 ++ Array.map strata[i]!.unit local_))
          | _ => pure (ForInStep.done (some none, __s.2)))).run pos = some ((none, out'), pos₁) := by
    intro n
    induction n with
    | zero =>
      intro i _ out pos _ r out' p₁ hl
      simp only [List.range'_zero, List.forIn_nil, StateT.run_pure] at hl ⊢
      obtain ⟨h1, _⟩ := Prod.mk.inj (Except.ok.inj hl)
      obtain ⟨rfl, rfl⟩ := Prod.mk.inj h1
      exact ⟨rfl, pos, rfl⟩
    | succ n ih =>
      intro i hin out pos hpos r out' p₁ hl
      rw [List.range'_succ, List.forIn_cons, StateT.run_bind, StateT.run_bind] at hl
      cases ho : (Flock.Draw.onWith (fun need => Draw.take src need) strata[i]!.need
          (Flock.Draw.subset strata[i]!.k strata[i]!.size)).run (starts strata src i) with
      | error e => rw [ho] at hl; exact absurd hl (by rw [bind_error, bind_error]; simp)
      | ok x =>
        obtain ⟨v, p₂⟩ := x
        obtain ⟨j, Dl, rfl, rfl, hst⟩ := onWith_take src _ (Law.need_pos _) _ _ _ _ ho
        have hseg : seg strata src i = j * strata[i]!.need :=
          used_of_stopsAt strata[i]! src _ j Dl hst
        obtain ⟨hTs, hTe⟩ := hT i (by omega)
        have hag : Agree src T i (starts strata src i) (starts strata src i + j * strata[i]!.need) := by
          refine ⟨Nat.le_add_right _ _, hst.2.1, ?_, ?_⟩
          · rw [Nat.add_sub_cancel_left, ← hseg]; exact hTs
          · rw [Nat.add_sub_cancel_left, ← hseg]; exact hTe
        have hm := (sim_onWith (src := src) (T := T) (i := i) (base := starts strata src i) _ (Law.need_pos _) _ _ _ _
          ho).2 pos (by rw [hpos i le_rfl, Nat.zero_add]) hag
        rw [ho, bind_ok] at hl
        dsimp only at hl
        rw [StateT.run_pure, pure_bind] at hl
        dsimp only at hl
        have hnext : starts strata src i + j * strata[i]!.need = starts strata src (i + 1) := by
          rw [starts_succ, hseg]
        rw [hnext] at hl
        obtain ⟨hr, pos₁, hpos₁⟩ := ih (i + 1) (by omega) _
          (Function.update pos i (starts strata src i + j * strata[i]!.need - starts strata src i))
          (fun x hx => by rw [Function.update_of_ne (by omega)]; exact hpos x (by omega)) r out' p₁ hl
        refine ⟨hr, pos₁, ?_⟩
        rw [List.range'_succ, List.forIn_cons, StateT.run_bind, StateT.run_bind, hm]
        exact hpos₁
  rw [StateT.run_bind] at h ⊢
  cases hl : (forIn (m := StateT ℕ (Except String)) (List.range' 0 strata.size 1)
      ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run 0 with
  | error e => rw [hl] at h; exact absurd h (by rw [bind_error]; simp)
  | ok y =>
    obtain ⟨⟨r, out'⟩, p₁⟩ := y
    rw [hl, bind_ok] at h
    dsimp only at h
    obtain ⟨rfl, pos₁, hm⟩ := loop strata.size 0 (by omega) #[] (fun _ => 0) (fun _ _ => rfl) _ out' p₁ hl
    rw [StateT.run_pure] at h
    obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj h)
    cases hl2 : (forIn (m := StateT (ℕ → ℕ) Option) (List.range' 0 strata.size 1)
        ((none : Option (Option (Array ℕ))), (#[] : Array ℕ)) _).run (fun _ => 0) with
    | none => exact absurd (hl2.symm.trans hm) (by simp)
    | some z =>
      obtain rfl := Option.some.inj (hl2.symm.trans hm)
      exact ⟨pos₁, by rw [← h1]; rfl⟩

theorem tapesOf_lt {m : ℕ} {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256) (x : ℕ) (hx : x < m) :
    Law.tapesOf f x = Law.streamOf (f ⟨x, hx⟩) := by
  simp [Law.tapesOf, Array.getD_eq_getD_getElem?, hx]

theorem streamOf_getElem {L : ℕ} (f : Fin L → Fin 256) (k : ℕ) (h : k < (Law.streamOf f).size) :
    (Law.streamOf f)[k] = UInt8.ofFin (f ⟨k, by rwa [DrawExec.size_streamOf] at h⟩) := by
  simp [Law.streamOf, ByteArray.getElem_eq_getElem_data]

/-- **Tape `i` begins with stratum `i`'s segment of the coins.** -/
theorem tapes_extract {m L : ℕ} (St : Fin m → Flock.Draw.Stratum) (c : Fin L → Fin 256) (pad : Pads m L)
    (i : Fin m) :
    (Law.streamOf (tapes St c pad i)).extract 0 (seg (Array.ofFn St) (Law.streamOf c) i) =
      (Law.streamOf c).extract (starts (Array.ofFn St) (Law.streamOf c) i)
        (starts (Array.ofFn St) (Law.streamOf c) i + seg (Array.ofFn St) (Law.streamOf c) i) := by
  have hs := starts_le (Array.ofFn St) (Law.streamOf c) (i.val + 1)
  rw [starts_succ, DrawExec.size_streamOf] at hs
  have h := (ByteArray.extract_eq_extract_iff_getElem (as := Law.streamOf (tapes St c pad i)) (bs := Law.streamOf c)
    (i := 0) (j := starts (Array.ofFn St) (Law.streamOf c) i) (len := seg (Array.ofFn St) (Law.streamOf c) i)
    (by rw [DrawExec.size_streamOf]; omega) (by rw [DrawExec.size_streamOf]; omega)).2 ?_
  · rwa [Nat.zero_add] at h
  · intro k hk
    rw [streamOf_getElem, streamOf_getElem]
    congr 1
    show (if (0 + k) < seg (Array.ofFn St) (Law.streamOf c) i then
      byteAt c (starts (Array.ofFn St) (Law.streamOf c) i + (0 + k)) else pad i ⟨0 + k, _⟩) = _
    rw [ite_eq_left (by omega), byteAt, dite_eq_left (by omega)]
    exact congrArg c (Fin.ext (by simp))

/-- **The served draw is the live draw on the served tapes**: a stratified draw by `drawUnits` on the coins `c` that
ends with `units` is `Law.runOS`'s draw of `units` on `tapes St c pad`, whatever the spare bytes `pad`. -/
theorem drawUnits_runOS {m L : ℕ} (St : Fin m → Flock.Draw.Stratum) (K pop : ℕ) (c : Fin L → Fin 256)
    (pad : Pads m L) (units : Array ℕ) (read : ℕ)
    (h : Draw.drawUnits (.stratified K (Array.ofFn St)) pop (Law.streamOf c) = .ok (units, read)) :
    Law.runOS (L := fun _ => L) (tapes St c pad) (.stratified K (Array.ofFn St)) pop =
      some { law := .stratified K (Array.ofFn St), population := pop, units } := by
  unfold Draw.drawUnits at h
  cases hw : (Flock.Draw.drawWith (fun _ need => Draw.take (Law.streamOf c) need) (.stratified K (Array.ofFn St))
      pop).run 0 with
  | error e => rw [hw] at h; exact absurd h (by rw [bind_error]; simp)
  | ok x =>
    obtain ⟨r, read'⟩ := x
    rw [hw, bind_ok] at h
    obtain ⟨d, rfl⟩ : ∃ d, r = some d := by
      cases r with
      | none => exact absurd h (by simp [throw, throwThe, MonadExceptOf.throw])
      | some d => exact ⟨d, rfl⟩
    dsimp only at h
    obtain ⟨hu, -⟩ := Prod.mk.inj (Except.ok.inj h)
    subst hu
    unfold Flock.Draw.drawWith at hw
    dsimp only at hw
    rw [StateT.run_bind] at hw
    cases ho : (Flock.Draw.stratifiedWith (fun _ need => Draw.take (Law.streamOf c) need) (Array.ofFn St)).run 0 with
    | error e => rw [ho] at hw; exact absurd hw (by rw [bind_error]; simp)
    | ok y =>
      obtain ⟨v, p⟩ := y
      rw [ho, bind_ok, StateT.run_pure] at hw
      obtain ⟨h1, -⟩ := Prod.mk.inj (Except.ok.inj hw)
      cases v with
      | none => exact absurd h1 (by simp)
      | some D =>
        obtain rfl := (Option.some.inj h1).symm
        have hT : ∀ x < (Array.ofFn St).size, seg (Array.ofFn St) (Law.streamOf c) x ≤
            (Law.tapesOf (L := fun _ => L) (tapes St c pad) x).size ∧
            (Law.tapesOf (L := fun _ => L) (tapes St c pad) x).extract 0 (seg (Array.ofFn St) (Law.streamOf c) x) =
              (Law.streamOf c).extract (starts (Array.ofFn St) (Law.streamOf c) x)
                (starts (Array.ofFn St) (Law.streamOf c) x + seg (Array.ofFn St) (Law.streamOf c) x) := by
          intro x hx
          rw [Array.size_ofFn] at hx
          rw [tapesOf_lt _ x hx]
          refine ⟨?_, tapes_extract St c pad ⟨x, hx⟩⟩
          have h1 := seg_le (Array.ofFn St) (Law.streamOf c) x
          rw [DrawExec.size_streamOf] at h1 ⊢
          omega
        obtain ⟨pos', hm⟩ := stratified_sim (Array.ofFn St) (Law.streamOf c) _ hT D p ho
        unfold Law.runOS
        rw [Law.drawWith_stratified_run, hm]
        rfl

end FlockSoundness.Audit.Served
