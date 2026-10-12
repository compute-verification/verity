import Mathlib
import Proofs.Accounting.Actual.Confidence

/-!
# A location as a transducer over what reaches it

In a strategy causal from every tick (v2.4 of `note:console/20261012T0645Z-draft-actual-circuit-model`), what a location
`X` does during `[t, t + n)` is a function of its state's content at `t` and of what crosses into it at each tick of
`[t, t + n)` (`transducer`), read through any per-tick observation that fixes what crosses in. What crosses in at a tick
is fixed by the signals the links deliver to `X`'s leaves then (`crossing_of_signals`). So `X` is a transducer whose state
is its content and whose input at each tick is its links' signals: the step "`X` is the prover of the protocol's model"
of B2 and B3, and the split of I. `Recall` is the case where nothing that crosses in depends on the coins.

Counting (`acts_encard_le`): with `mem X t ≤ M` in every run and at most `2^β` observations a tick, what `X` does during
`[t, t + n)` takes at most `2^{M + 1 + β n}` values across the coins. The `+ 1` is the content's length: a state of at
most `M` bits is one of `2^{M + 1} − 1` strings, since holding nothing differs from holding a 0 (`lists_encard_le`), so a
location of memory `M` carries `M + 1` bits across a tick, not `M`. With the links as alphabets (`Alphabet`) and every wire
into `X` on one of its ports, the observation is the ports' signals (`portSignals`), and what `X` does during
`[t, t + n)` takes at most `2^{M + 1 + β_X n}` values, `β_X` the ports' capacities summed (`acts_encard_le_links`).
-/

namespace Accounting.SecurityProofs.Actual

variable {Ω L : Type*}

/-- A wire read during `I` is one read at its own arrival tick. -/
theorem mem_crossing_iff (R : Actual.Run L) (X Y : Set L) (I : Set ℕ) (p : ℕ × L × ℕ × L × ℕ × ℕ) :
    p ∈ R.crossing X Y I ↔ p.1 ∈ I ∧ p ∈ R.crossing X Y {p.1} := by
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, rfl⟩
    exact ⟨ht, g, a, hr, ha, hg, rfl, rfl⟩
  · rintro ⟨hI, g, a, hr, ha, hg, -, rfl⟩
    exact ⟨g, a, hr, ha, hg, hI, rfl⟩

/-- Runs alike in what crosses at each tick of `[t, t + n)` are alike in what crosses during it. -/
theorem crossing_Ico_congr {R R' : Actual.Run L} {X Y : Set L} {t n : ℕ}
    (h : ∀ i < n, R.crossing X Y {t + i} = R'.crossing X Y {t + i}) :
    R.crossing X Y (Set.Ico t (t + n)) = R'.crossing X Y (Set.Ico t (t + n)) := by
  have key : ∀ u ∈ Set.Ico t (t + n), R.crossing X Y {u} = R'.crossing X Y {u} := by
    intro u hu
    obtain ⟨i, rfl⟩ := Nat.exists_eq_add_of_le hu.1
    exact h i (by have := hu.2; omega)
  ext p
  rw [mem_crossing_iff, mem_crossing_iff R']
  constructor
  · rintro ⟨hI, hp⟩
    exact ⟨hI, by rwa [← key _ hI]⟩
  · rintro ⟨hI, hp⟩
    exact ⟨hI, by rwa [key _ hI]⟩

/-- What crosses into `X` at tick `u` is what the links deliver to `X`'s leaves then, from outside `X`. -/
theorem mem_crossing_signal {E : Type*} (R : Actual.Run L) (link : Fin R.A.N → Fin R.A.N → E) (X : Finset L) (u : ℕ)
    (p : ℕ × L × ℕ × L × ℕ × ℕ) :
    p ∈ R.crossing (↑X)ᶜ ↑X {u} ↔ p.1 = u ∧ p.2.1 ∈ X ∧ p.2.2.2.1 ∉ X ∧ ∃ e, p.2.2 ∈ R.signal link e p.2.1 u := by
  obtain ⟨u', y, s⟩ := p
  constructor
  · rintro ⟨g, a, hr, ha, hg, ht, he⟩
    simp only [Prod.mk.injEq] at he
    obtain ⟨rfl, rfl, rfl⟩ := he
    refine ⟨ht, hg, ha, link g a, g, a, hr, fun hx => ha (hx ▸ hg), rfl, rfl, ht, rfl⟩
  · rintro ⟨rfl, hy, hx, e, g, a, hr, -, -, hxg, htg, rfl⟩
    exact ⟨g, a, hr, hx, hxg ▸ hy, htg, by rw [hxg, htg]⟩

/-- **What crosses in is fixed by the links' signals**: runs alike in what each link delivers to each of `X`'s leaves at
`u` are alike in what crosses into `X` at `u`. -/
theorem crossing_of_signals {E : Type*} {R R' : Actual.Run L} (link : Fin R.A.N → Fin R.A.N → E)
    (link' : Fin R'.A.N → Fin R'.A.N → E) (X : Finset L) (u : ℕ)
    (h : ∀ e, ∀ y ∈ X, R.signal link e y u = R'.signal link' e y u) :
    R.crossing (↑X)ᶜ ↑X {u} = R'.crossing (↑X)ᶜ ↑X {u} := by
  ext p
  rw [mem_crossing_signal R link, mem_crossing_signal R' link']
  constructor
  · rintro ⟨h1, h2, h3, e, he⟩
    exact ⟨h1, h2, h3, e, by rwa [← h e _ h2]⟩
  · rintro ⟨h1, h2, h3, e, he⟩
    exact ⟨h1, h2, h3, e, by rwa [h e _ h2]⟩

/-- The signals a list of ports (a link and the leaf of `X` it delivers to) carries at tick `u`. -/
def portSignals {E : Type*} (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (ports : List (E × L)) (r : Ω) (u : ℕ) : List (Set (ℕ × L × ℕ × ℕ)) :=
  ports.map fun p => (S r).signal (link r) p.1 p.2 u

/-- When every wire into `X` runs on one of the ports, the ports' signals at `u` fix what crosses into `X` at `u`. -/
theorem crossing_of_ports {E : Type*} (S : Ω → Actual.Run L) (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E)
    (X : Finset L) (ports : List (E × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X → (link r g a, (S r).A.x g) ∈ ports)
    (u : ℕ) (r r' : Ω) (h : portSignals S link ports r u = portSignals S link ports r' u) :
    (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u} := by
  have hp : ∀ p ∈ ports, (S r).signal (link r) p.1 p.2 u = (S r').signal (link r') p.1 p.2 u :=
    List.map_inj_left.1 h
  have key : ∀ {r r' : Ω}, (∀ p ∈ ports, (S r).signal (link r) p.1 p.2 u = (S r').signal (link r') p.1 p.2 u) →
      (S r).crossing (↑X)ᶜ ↑X {u} ⊆ (S r').crossing (↑X)ᶜ ↑X {u} := by
    intro r r' hp q hq
    have hq' := hq
    obtain ⟨g, a, hr, ha, hg, -, rfl⟩ := hq'
    rw [mem_crossing_signal (S r) (link r)] at hq
    rw [mem_crossing_signal (S r') (link r')]
    refine ⟨hq.1, hq.2.1, hq.2.2.1, link r g a, ?_⟩
    rw [← hp _ (hports r g a hr ha hg)]
    exact ⟨g, a, hr, fun hx => ha (hx ▸ hg), rfl, rfl, hq.1, rfl⟩
  exact (key hp).antisymm (key fun p hm => (hp p hm).symm)

/-- With each link an alphabet, the ports' signals at a tick take at most `2^{∑ β}` values. -/
theorem portSignals_encard_le {E : Type*} (S : Ω → Actual.Run L) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (hα : Alphabet S β link) (u : ℕ) :
    ∀ ports : List (E × L), (Set.range fun r => portSignals S link ports r u).encard ≤
      ((2 ^ (ports.map fun p => β p.1).sum : ℕ) : ℕ∞)
  | [] => by
    have : (Set.range fun r => portSignals S link [] r u) ⊆ {[]} := by
      rintro _ ⟨r, rfl⟩
      rfl
    exact (Set.encard_le_encard this).trans (by simp)
  | p :: ps => by
    have hsub : (Set.range fun r => portSignals S link (p :: ps) r u) ⊆
        (fun q : Set (ℕ × L × ℕ × ℕ) × List (Set (ℕ × L × ℕ × ℕ)) => q.1 :: q.2) ''
          ((Set.range fun r => (S r).signal (link r) p.1 p.2 u) ×ˢ Set.range fun r => portSignals S link ps r u) := by
      rintro _ ⟨r, rfl⟩
      exact ⟨(_, _), ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩, rfl⟩
    refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
    rw [Set.encard_prod]
    calc _ ≤ ((2 ^ β p.1 : ℕ) : ℕ∞) * ((2 ^ (ps.map fun p => β p.1).sum : ℕ) : ℕ∞) :=
          mul_le_mul' (hα p.1 p.2 u) (portSignals_encard_le S β link hα u ps)
      _ = _ := by simp [List.map_cons, List.sum_cons, pow_add]

variable [LinearOrder L]

/-- **A location is a transducer over what reaches it**: in a strategy causal from every tick, if what crosses into `X`
at each tick `u` is fixed by an observation `obs r u`, then what `X` does during `[t, t + n)` is a function `F n` of its
state's content at `t` (at most `mem X t` bits) and of its first `n` observations from `t`. -/
theorem transducer {Ob : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L) (t : ℕ) (obs : Ω → ℕ → Ob)
    (hobs : ∀ u r r', obs r u = obs r' u → (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u}) :
    ∃ F : ℕ → List Bool → List Ob →
        Set (ℕ × L × ℕ × ℕ) × Set (ℕ × L × ℕ × L × ℕ × ℕ) × Set (ℕ × ℕ × L × ℕ × ℕ),
      ∀ r n, (S r).acts ↑X (Set.Ico t (t + n)) =
        F n ((S r).content X t) ((List.range n).map fun i => obs r (t + i)) := by
  have hf : ∀ n, Function.FactorsThrough (fun r => (S r).acts ↑X (Set.Ico t (t + n)))
      (fun r => ((S r).content X t, (List.range n).map fun i => obs r (t + i))) := by
    intro n r r' h
    simp only [Prod.mk.injEq] at h
    exact hc X t (t + n) r r' h.1
      (crossing_Ico_congr fun i hi => hobs _ r r' (List.map_inj_left.1 h.2 i (List.mem_range.2 hi)))
  exact ⟨fun n c os => Function.extend (fun r => ((S r).content X t, (List.range n).map fun i => obs r (t + i)))
    (fun r => (S r).acts ↑X (Set.Ico t (t + n))) (fun _ => (∅, ∅, ∅)) (c, os),
    fun r n => ((hf n).extend_apply _ r).symm⟩

/-- Bit strings of length at most `M` number `2^{M + 1} − 1`. -/
theorem lists_encard_le (M : ℕ) : {l : List Bool | l.length ≤ M}.encard + 1 ≤ 2 ^ (M + 1) := by
  induction M with
  | zero =>
    have : {l : List Bool | l.length ≤ 0} = {[]} := by
      ext l
      simp
    rw [this, Set.encard_singleton]
    norm_num
  | succ M ih =>
    have hsub : {l : List Bool | l.length ≤ M + 1} ⊆
        {[]} ∪ (fun p : Bool × List Bool => p.1 :: p.2) '' (Set.univ ×ˢ {l | l.length ≤ M}) := by
      intro l hl
      cases l with
      | nil => exact Or.inl rfl
      | cons x l => exact Or.inr ⟨(x, l), ⟨trivial, by simpa using hl⟩, rfl⟩
    have h1 := (Set.encard_le_encard hsub).trans ((Set.encard_union_le _ _).trans
      (add_le_add le_rfl (Set.encard_image_le (fun p : Bool × List Bool => p.1 :: p.2)
        (Set.univ ×ˢ {l : List Bool | l.length ≤ M}))))
    rw [Set.encard_prod, Set.encard_univ, ENat.card_eq_coe_fintype_card, Fintype.card_bool, Set.encard_singleton]
      at h1
    calc {l : List Bool | l.length ≤ M + 1}.encard + 1
        ≤ (1 + ((2 : ℕ) : ℕ∞) * {l : List Bool | l.length ≤ M}.encard) + 1 := by gcongr
      _ = 2 * ({l : List Bool | l.length ≤ M}.encard + 1) := by push_cast; ring
      _ ≤ 2 * 2 ^ (M + 1) := by gcongr
      _ = 2 ^ (M + 1 + 1) := by ring

/-- `n` observations, at most `2^β` a tick, are one of at most `2^{β n}` lists. -/
theorem obs_encard_le {Ob : Type*} (obs : Ω → ℕ → Ob) (t β : ℕ)
    (hα : ∀ u, (Set.range fun r => obs r u).encard ≤ ((2 ^ β : ℕ) : ℕ∞)) (n : ℕ) :
    (Set.range fun r => (List.range n).map fun i => obs r (t + i)).encard ≤ ((2 ^ (β * n) : ℕ) : ℕ∞) := by
  induction n with
  | zero =>
    have : (Set.range fun r : Ω => (List.range 0).map fun i => obs r (t + i)) ⊆ {[]} := by
      rintro _ ⟨r, rfl⟩
      simp
    exact (Set.encard_le_encard this).trans (by simp)
  | succ n ih =>
    have hsub : (Set.range fun r => (List.range (n + 1)).map fun i => obs r (t + i)) ⊆
        (fun p : List Ob × Ob => p.1 ++ [p.2]) ''
          ((Set.range fun r => (List.range n).map fun i => obs r (t + i)) ×ˢ Set.range fun r => obs r (t + n)) := by
      rintro _ ⟨r, rfl⟩
      exact ⟨(_, _), ⟨⟨r, rfl⟩, ⟨r, rfl⟩⟩, by simp [List.range_succ]⟩
    refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
    rw [Set.encard_prod]
    calc _ ≤ ((2 ^ (β * n) : ℕ) : ℕ∞) * ((2 ^ β : ℕ) : ℕ∞) := mul_le_mul' ih (hα _)
      _ = _ := by push_cast; ring

/-- **What a location does takes at most `2^{M + 1 + β n}` values**: in a strategy causal from every tick, with what
crosses into `X` at each tick fixed by an observation of at most `2^β` values and `X`'s state at `t` at most `M` bits in
every run, what `X` does during `[t, t + n)` takes at most `2^{M + 1 + β n}` values across the coins. -/
theorem acts_encard_le {Ob : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (X : Finset L) (t : ℕ)
    (obs : Ω → ℕ → Ob)
    (hobs : ∀ u r r', obs r u = obs r' u → (S r).crossing (↑X)ᶜ ↑X {u} = (S r').crossing (↑X)ᶜ ↑X {u})
    (M β : ℕ) (hM : ∀ r, (S r).mem X t ≤ M) (hα : ∀ u, (Set.range fun r => obs r u).encard ≤ ((2 ^ β : ℕ) : ℕ∞))
    (n : ℕ) :
    (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))).encard ≤ ((2 ^ (M + 1 + β * n) : ℕ) : ℕ∞) := by
  obtain ⟨F, hF⟩ := transducer hc X t obs hobs
  have hsub : (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))) ⊆
      (fun p : List Bool × List Ob => F n p.1 p.2) ''
        ({l | l.length ≤ M} ×ˢ Set.range fun r => (List.range n).map fun i => obs r (t + i)) := by
    rintro _ ⟨r, rfl⟩
    exact ⟨(_, _), ⟨(content_length_le _ _ _).trans (hM r), ⟨r, rfl⟩⟩, (hF r n).symm⟩
  refine (Set.encard_le_encard hsub).trans ((Set.encard_image_le _ _).trans ?_)
  rw [Set.encard_prod]
  have h1 : {l : List Bool | l.length ≤ M}.encard ≤ 2 ^ (M + 1) := le_self_add.trans (lists_encard_le M)
  calc _ ≤ (2 : ℕ∞) ^ (M + 1) * ((2 ^ (β * n) : ℕ) : ℕ∞) := mul_le_mul' h1 (obs_encard_le obs t β hα n)
    _ = _ := by push_cast; ring

/-- **What a location does over `n` ticks carries at most `M + 1 + β_X n` bits**: in a strategy causal from every tick
whose links are alphabets, with every wire into `X` on one of its ports (`β_X` their links' capacities summed) and `X`'s
state at `t` at most `M` bits in every run, what `X` does during `[t, t + n)` takes at most `2^{M + 1 + β_X n}` values. -/
theorem acts_encard_le_links {E : Type*} {S : Ω → Actual.Run L} (hc : CausalFrom S) (β : E → ℕ)
    (link : ∀ r, Fin (S r).A.N → Fin (S r).A.N → E) (hα : Alphabet S β link) (X : Finset L) (ports : List (E × L))
    (hports : ∀ r g, ∀ a ∈ (S r).A.reads g, (S r).A.x a ∉ X → (S r).A.x g ∈ X → (link r g a, (S r).A.x g) ∈ ports)
    (t M : ℕ) (hM : ∀ r, (S r).mem X t ≤ M) (n : ℕ) :
    (Set.range fun r => (S r).acts ↑X (Set.Ico t (t + n))).encard ≤
      ((2 ^ (M + 1 + (ports.map fun p => β p.1).sum * n) : ℕ) : ℕ∞) :=
  acts_encard_le hc X t (portSignals S link ports) (fun u r r' => crossing_of_ports S link X ports hports u r r') M _
    hM (fun u => portSignals_encard_le S β link hα u ports) n

end Accounting.SecurityProofs.Actual
