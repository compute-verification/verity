import Proofs.Flock.Recursive.HoloExec

/-!
# Recursion: the inner fork with the last round's message unbound

`ForkInner`'s leaf lemmas with the simulated inner prover's last message free. At the last round it sends a message
that the recursive prover's own leaf at some last coin decodes and the inner verifier accepts (`refStratL`), not one a
commitment fixed before the coin. An inner verifier that ignores the last round's coin (`Inner.LastFree`, every
`schedInner`) accepts that message at every last coin, and a pick that ignores it (`PickLastFree`: `clashPickCL`,
`clashPickHL`) reads a fork of the simulated prover as one of the recursive prover's, at its leaves' own last coins.

* `relabelL`, `refStratL`, `AtLeafL`: `relabel`, `refStrat` and `AtLeaf` with the last message free.
* `leafAvg_relabelL_le`, `outcome_contF_refL`, `finds_forkG_refL`: `leafAvg_relabel_at`, `outcome_contF_ref` and
  `finds_forkG_ref` with the last message free.
* `InnerAtLeavesL`, `innerAtLeavesL_of_forkOn`: `InnerAtLeaves` with the decoded last message free, from
  `InnerSoundForkOn` at a verifier and a pick that ignore the last coin, unless the recursive prover has an inner fork.
* `schedInner_lastFree`, `clashPickCL_lastFree`, `clashPickHL_lastFree`: the three for C-Flock's inner protocols.
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit StrictCR
open scoped ENNReal

section Last

variable {M C : Type}

/-- **`u` and `u'` are one transcript but for the last round's coin.** -/
def LastCoin {N : Type} (u u' : List (N × C)) : Prop :=
  u = u' ∨ ∃ p m ρ ρ', u = p ++ [(m, ρ)] ∧ u' = p ++ [(m, ρ')]

theorem LastCoin.cons {N : Type} {u u' : List (N × C)} (e : N × C) (h : LastCoin u u') :
    LastCoin (e :: u) (e :: u') := by
  rcases h with rfl | ⟨p, m, ρ, ρ', rfl, rfl⟩
  · exact Or.inl rfl
  · exact Or.inr ⟨e :: p, m, ρ, ρ', rfl, rfl⟩

/-- **A pick that ignores the last round's coin** of transcripts of `r` rounds. -/
def PickLastFree {N : Type} (r : ℕ) (pk : List (N × C) → List (N × C) → StrictCR.Pair) : Prop :=
  ∀ u₁ u₂ u₁' u₂', u₁.length = r → u₂.length = r → LastCoin u₁ u₁' → LastCoin u₂ u₂' → pk u₁ u₂ = pk u₁' u₂'

/-- **An inner verifier that ignores the last round's coin.** -/
def Inner.LastFree (I : Inner) : Prop :=
  ∀ x (u : List (I.Msg × I.Coin)) m ρ ρ', u.length + 1 = I.r → I.accepts x (u ++ [(m, ρ)]) →
    I.accepts x (u ++ [(m, ρ')])

/-- **`relabel ref t` with `m` as the last round's message.** -/
def relabelL {N : Type} : (List (M × C) → M → N) → N → List (M × C) → List (N × C)
  | _, _, [] => []
  | _, m, [(_, ρ)] => [(m, ρ)]
  | ref, m, (c, ρ) :: e :: t => (ref [] c, ρ) :: relabelL (fun p c' => ref ((c, ρ) :: p) c') m (e :: t)

variable {β : Type} [Fintype C] [Nonempty C]

open Classical in
/-- **The simulated inner prover's last message** at the last node of `s`: a message `m` such that, at some last coin
`ρ`, `Q` holds of the transcript `[(m, ρ)]` and `V` of the leaf there with it, when there is one. -/
noncomputable def lastMsg {N δ : Type} (f : List (M × C) → Game β) (ref : List (M × C) → M → N) (Q : δ → Prop)
    (h : List (N × C) → δ) (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (s : Strategy (rounds M C 1 f)) : N :=
  if hm : ∃ ρ m, Q (h [(m, ρ)]) ∧ V [(s.1, ρ)] (s.2 ρ) [(m, ρ)] then hm.choose_spec.choose else ref [] s.1

theorem lastMsg_spec {N δ : Type} (f : List (M × C) → Game β) (ref : List (M × C) → M → N) (Q : δ → Prop)
    (h : List (N × C) → δ) (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (s : Strategy (rounds M C 1 f)) (hm : ∃ ρ m, Q (h [(m, ρ)]) ∧ V [(s.1, ρ)] (s.2 ρ) [(m, ρ)]) :
    ∃ ρ, Q (h [(lastMsg f ref Q h V s, ρ)]) ∧ V [(s.1, ρ)] (s.2 ρ) [(lastMsg f ref Q h V s, ρ)] := by
  unfold lastMsg
  rw [dite_eq_left hm]
  exact ⟨hm.choose, hm.choose_spec.choose_spec⟩

/-- **The simulated inner prover, last message free**: `refStrat`, but at the last round it sends `lastMsg`. -/
noncomputable def refStratL {N δ : Type} (Q : δ → Prop) : (r : ℕ) → (f : List (M × C) → Game β) →
    (List (M × C) → M → N) → (h : List (N × C) → δ) →
    ((t : List (M × C)) → Strategy (f t) → List (N × C) → Prop) → Strategy (rounds M C r f) →
    Strategy (rounds N C r fun t => .ret (h t))
  | 0, _, _, _, _, _ => PUnit.unit
  | 1, f, ref, h, V, s => ⟨lastMsg f ref Q h V s, fun _ => PUnit.unit⟩
  | r + 2, f, ref, h, V, s => ⟨ref [] s.1, fun ρ =>
      refStratL Q (r + 1) (fun t => f ((s.1, ρ) :: t)) (fun p c => ref ((s.1, ρ) :: p) c)
        (fun t => h ((ref [] s.1, ρ) :: t)) (fun t s' u => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u)) (s.2 ρ)⟩

/-- **`V` at a leaf of `s` that the coins of `u` reach but for the last**, read at `u` with that leaf's last coin. -/
def AtLeafL {N : Type} : (r : ℕ) → (f : List (M × C) → Game β) →
    ((t : List (M × C)) → Strategy (f t) → List (N × C) → Prop) → Strategy (rounds M C r f) → List (N × C) → Prop
  | 0, _, V, s, u => V [] s u
  | 1, _, V, s, (a, _) :: u => ∃ ρ, V [(s.1, ρ)] (s.2 ρ) ((a, ρ) :: u)
  | r + 2, f, V, s, (a, ρ) :: u =>
      AtLeafL (r + 1) (fun t => f ((s.1, ρ) :: t)) (fun t s' u' => V ((s.1, ρ) :: t) s' ((a, ρ) :: u')) (s.2 ρ) u
  | _ + 1, _, _, _, [] => False

open Classical in
/-- **The simulation with the last message free**: averaged over the round coins, the indicator that some last message
makes the relabelled transcript satisfy `Q` and the leaf's `V` is at most the average, against `refStratL`, of the
indicator that its transcript satisfies `Q` and `AtLeafL`, when `Q` ignores the last coin. -/
theorem leafAvg_relabelL_le {N δ : Type} (Q : δ → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop) (s : Strategy (rounds M C r f)),
    (∀ u m ρ ρ', u.length + 1 = r → Q (h (u ++ [(m, ρ)])) → Q (h (u ++ [(m, ρ')]))) →
    leafAvg r f (fun t s' => if ∃ m, Q (h (relabelL ref m t)) ∧ V t s' (relabelL ref m t) then 1 else 0) s ≤
      leafAvg r (fun t => .ret (h t)) (fun u _ => if Q (h u) ∧ AtLeafL r f V s u then 1 else 0)
        (refStratL Q r f ref h V s)
  | 0, _, _, _, _, _, _ => ite_one_zero_le fun ⟨_, hm⟩ => hm
  | 1, f, ref, h, V, s, hQ => by
      show avg _ ≤ avg _
      refine avg_mono fun ρ => ite_one_zero_le fun ⟨m, hm⟩ => ?_
      obtain ⟨ρ', hq, hv⟩ := lastMsg_spec f ref Q h V s ⟨ρ, m, hm⟩
      exact ⟨hQ [] _ ρ' ρ rfl hq, ρ', hv⟩
  | r + 2, f, ref, h, V, s, hQ => by
      show avg _ ≤ avg _
      refine avg_mono fun ρ => Eq.trans_le (leafAvg_congr_len (G := fun t s' =>
        if ∃ m, Q (h ((ref [] s.1, ρ) :: relabelL (fun p c => ref ((s.1, ρ) :: p) c) m t)) ∧
          V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: relabelL (fun p c => ref ((s.1, ρ) :: p) c) m t)
        then 1 else 0) (fun t s' ht => ?_) _) ?_
      · cases t with
        | nil => simp at ht
        | cons _ _ => rfl
      · exact leafAvg_relabelL_le Q (r + 1) (fun t => f ((s.1, ρ) :: t)) (fun p c => ref ((s.1, ρ) :: p) c)
          (fun t => h ((ref [] s.1, ρ) :: t)) (fun t s' u => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u)) (s.2 ρ)
          fun u m ρ₁ ρ₂ hu hq => hQ ((ref [] s.1, ρ) :: u) m ρ₁ ρ₂ (by simp only [List.length_cons]; omega) hq

/-- **An outcome of `refStratL`'s run that satisfies `G` is one of the recursive prover's but for the last coin**, when
`G` of a transcript gives `AtLeafL`: the trial at the leaf it names reaches the transcript at that leaf's last coin. -/
theorem outcome_contF_refL {N δ : Type} (Q G : δ → Prop) : ∀ (r : ℕ) (f : List (M × C) → Game β)
    (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (T : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) (s : Strategy (rounds M C r f)),
    (∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (h u)) →
    (∀ u, G (h u) → AtLeafL r f V s u) → ∀ d,
    Game.Outcome (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) (refStratL Q r f ref h V s)).g
      (contF r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩) (refStratL Q r f ref h V s)).s d →
    G d → ∃ u u', d = h u ∧ u.length = r ∧ LastCoin u u' ∧ Game.Outcome (contF r f T s).g (contF r f T s).s (h u')
  | 0, _, _, h, _, _, s, hVT, hG, d, hd, hGd => by
      have hd' : h [] = d := hd
      subst hd'
      exact ⟨[], [], rfl, rfl, Or.inl rfl, hVT [] s [] (hG [] hGd)⟩
  | 1, f, ref, h, V, T, s, hVT, hG, d, hd, hGd => by
      obtain ⟨ρ, hd⟩ := hd
      have hd' : h [(lastMsg f ref Q h V s, ρ)] = d := hd
      subst hd'
      obtain ⟨ρ', hv⟩ := hG _ hGd
      exact ⟨_, [(lastMsg f ref Q h V s, ρ')], rfl, rfl, Or.inr ⟨[], _, ρ, ρ', rfl, rfl⟩, ρ', hVT _ _ _ hv⟩
  | r + 2, f, ref, h, V, T, s, hVT, hG, d, hd, hGd => by
      obtain ⟨ρ, hd⟩ := hd
      obtain ⟨u, u', rfl, hl, hc, ho⟩ := outcome_contF_refL Q G (r + 1) (fun t => f ((s.1, ρ) :: t))
        (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (fun t s' u' => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u')) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)
        (fun t s' u hv => hVT _ s' _ hv) (fun u hu => hG _ hu) d hd hGd
      exact ⟨(ref [] s.1, ρ) :: u, (ref [] s.1, ρ) :: u', rfl, (by simp only [List.length_cons]; omega), hc.cons _,
        ρ, ho⟩

/-- **A fork of `refStratL` on transcripts satisfying `G` is a fork of the recursive prover**, when the pick ignores the
last coin: at the same round and coins but for the last, its two trials reach the two transcripts at their leaves' last
coins (`outcome_contF_refL`), so `pick` reads the same collision. -/
theorem finds_forkG_refL {D N δ : Type} (H : List UInt8 → D) (pk : δ → δ → StrictCR.Pair) (Q G : δ → Prop) :
    ∀ (k r : ℕ) (f : List (M × C) → Game β) (ref : List (M × C) → M → N) (h : List (N × C) → δ)
    (V : (t : List (M × C)) → Strategy (f t) → List (N × C) → Prop)
    (T : (t : List (M × C)) → Strategy (f t) → StrictCR.Finder δ) (s : Strategy (rounds M C r f)),
    (∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (h u)) →
    (∀ u, G (h u) → AtLeafL r f V s u) →
    (∀ u₁ u₂ u₁' u₂', u₁.length = r → u₂.length = r → LastCoin u₁ u₁' → LastCoin u₂ u₂' →
      pk (h u₁) (h u₂) = pk (h u₁') (h u₂')) →
    (forkG (pickOn G pk) k r (fun t => .ret (h t)) (fun t _ => ⟨.ret (h t), PUnit.unit⟩)
      (refStratL Q r f ref h V s)).Finds H →
    (forkG pk k r f T s).Finds H
  | 0, 0, _, _, _, _, _, _, _, _, _, hF => hF
  | _ + 1, 0, _, _, _, _, _, _, _, _, _, hF => hF
  | 0, r + 1, f, ref, h, V, T, s, hVT, hG, hpk, hF => by
      classical
      obtain ⟨p, hp, hc⟩ := hF
      obtain ⟨d, d', hd, hd', rfl⟩ := outcome_of_twoRuns _ _ _ hp
      have hGG : G d ∧ G d' := Classical.byContradiction fun hn => by
        have hnone : pickOn G pk d d' = none := ite_eq_right hn
        rw [hnone] at hc
        obtain ⟨_, _, he, _⟩ := hc
        exact absurd he (by simp)
      rw [pickOn_of hGG.1 hGG.2] at hc
      obtain ⟨u₁, u₁', rfl, hl₁, hc₁, ho₁⟩ := outcome_contF_refL Q G (r + 1) f ref h V T s hVT hG d hd hGG.1
      obtain ⟨u₂, u₂', rfl, hl₂, hc₂, ho₂⟩ := outcome_contF_refL Q G (r + 1) f ref h V T s hVT hG d' hd' hGG.2
      rw [hpk u₁ u₂ u₁' u₂' hl₁ hl₂ hc₁ hc₂] at hc
      exact ⟨_, outcome_twoRuns _ _ pk ho₁ ho₂, hc⟩
  | k + 1, 1, _, _, _, _, _, _, _, _, _, hF => by
      obtain ⟨p, ⟨_, hp⟩, hc⟩ := hF
      have hp' : (none : StrictCR.Pair) = p := by cases k <;> exact hp
      subst hp'
      obtain ⟨_, _, he, _⟩ := hc
      exact absurd he (by simp)
  | k + 1, r + 2, f, ref, h, V, T, s, hVT, hG, hpk, hF => by
      obtain ⟨p, ⟨ρ, hp⟩, hc⟩ := hF
      obtain ⟨p', hp', hc'⟩ := finds_forkG_refL H pk Q G k (r + 1) (fun t => f ((s.1, ρ) :: t))
        (fun p c => ref ((s.1, ρ) :: p) c) (fun t => h ((ref [] s.1, ρ) :: t))
        (fun t s' u' => V ((s.1, ρ) :: t) s' ((ref [] s.1, ρ) :: u')) (fun t => T ((s.1, ρ) :: t)) (s.2 ρ)
        (fun t s' u hv => hVT _ s' _ hv) (fun u hu => hG _ hu)
        (fun u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂ => hpk _ _ _ _ (by simp only [List.length_cons]; omega)
          (by simp only [List.length_cons]; omega) (c₁.cons _) (c₂.cons _))
        ⟨p, hp, hc⟩
      exact ⟨p', ⟨ρ, hp'⟩, hc'⟩

end Last

section Outer

variable {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {β : Type}
  {outer : Reg → L.Ω → List (Cm × I.Coin) → Game β}

open Classical in
/-- **`InnerAtLeaves` with the decoded last message free**: the outcome decodes to the relabelled transcript but for the
last round's message (`relabelL`), and the inner verifier accepts that. -/
def InnerAtLeavesL (σ : Strategy (recGame L Reg I Cm outer)) (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop)
    (x : Reg → L.Ω → I.Stmt) {W : Type} (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wmsg : W → List I.Msg)
    (ref : L.Ω → List (Cm × I.Coin) → Cm → I.Msg) (εin : ℝ≥0∞) : Prop :=
  ∀ ω, leafAvg I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
    (fun t s => if ¬ I.holds (x σ.1 ω) ∧ ∃ m b,
        Game.Outcome (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b ∧
          Acc σ.1 ω t b ∧ (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip
            (t.map Prod.snd) = relabelL (ref ω) m t ∧ I.accepts (x σ.1 ω) (relabelL (ref ω) m t) then 1 else 0)
    (σ.2 ω) ≤ εin

/-- **The inner protocol's soundness in fork form bounds the inner term with the last message free**, at every choice of
reference openings, when the inner verifier and the pick ignore the last coin, unless the recursive prover has an inner
fork: `innerAtLeavesV_of_forkOn` at `refStratL` and `G := AtLeafL`. -/
theorem innerAtLeavesL_of_forkOn (σ : Strategy (recGame L Reg I Cm outer))
    (Acc : Reg → L.Ω → List (Cm × I.Coin) → β → Prop) (x : Reg → L.Ω → I.Stmt) {W : Type}
    (dec : ∀ R ω t, Strategy (outer R ω t) → β → W) (wmsg : W → List I.Msg) {D : Type} (H : List UInt8 → D)
    (pick : I.Stmt → List (I.Msg × I.Coin) → List (I.Msg × I.Coin) → StrictCR.Pair) {εin : ℝ≥0∞}
    (hin : InnerSoundForkOn H I pick εin) (hlast : I.LastFree) (hpick : ∀ x, PickLastFree I.r (pick x))
    (hno : ¬ ∃ ω, ∃ i < I.r,
      (forkG (pick (x σ.1 ω)) i I.r (fun t => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b))
        (fun t s => StrictCR.oneRun (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s)
          (fun b => (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip
            (t.map Prod.snd)))
        (σ.2 ω)).Finds H) :
    ∀ ref, InnerAtLeavesL σ Acc x dec wmsg ref εin := by
  classical
  intro ref ω
  by_cases hx : I.holds (x σ.1 ω)
  · refine le_trans (leafAvg_mono (G := fun _ _ => 0) (fun t s => ?_) _) ?_
    · exact le_of_eq (ite_eq_right fun h => h.1 hx)
    · exact (leafAvg_const 0 _).le.trans zero_le
  let f := fun t : List (Cm × I.Coin) => (outer σ.1 ω t).map fun b => (σ.1, ω, t, b)
  let dz : (t : List (Cm × I.Coin)) → Strategy (f t) → β → List (I.Msg × I.Coin) := fun t s b =>
    (wmsg (dec σ.1 ω t (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b)).zip (t.map Prod.snd)
  let V : (t : List (Cm × I.Coin)) → Strategy (f t) → List (I.Msg × I.Coin) → Prop := fun t s u =>
    ∃ b, Game.Outcome (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) b ∧
      Acc σ.1 ω t b ∧ dz t s b = u
  let T : (t : List (Cm × I.Coin)) → Strategy (f t) → StrictCR.Finder (List (I.Msg × I.Coin)) := fun t s =>
    StrictCR.oneRun (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s) (dz t s)
  let Q : List (I.Msg × I.Coin) → Prop := I.accepts (x σ.1 ω)
  let G : List (I.Msg × I.Coin) → Prop := AtLeafL I.r f V (σ.2 ω)
  have hVT : ∀ t s' u, V t s' u → Game.Outcome (T t s').g (T t s').s (id u) := by
    rintro t s' u ⟨b, hb, -, rfl⟩
    exact outcome_oneRun (outer σ.1 ω t) (Strategy.ofMap (fun b => (σ.1, ω, t, b)) (outer σ.1 ω t) s') (dz t s') hb
  rcases hin (x σ.1 ω) hx G (refStratL Q I.r f (ref ω) id V (σ.2 ω)) with ⟨i, hi, hF⟩ | hle
  · exact (hno ⟨ω, i, hi, finds_forkG_refL H (pick (x σ.1 ω)) Q G i I.r f (ref ω) id V T (σ.2 ω) hVT
      (fun _ hu => hu) (hpick (x σ.1 ω)) hF⟩).elim
  refine le_trans ?_ hle
  refine le_trans (leafAvg_mono (G := fun t s =>
    if ∃ m, Q (id (relabelL (ref ω) m t)) ∧ V t s (relabelL (ref ω) m t) then 1 else 0)
    (fun t s => ite_one_zero_le ?_) _) ?_
  · rintro ⟨-, m, b, hb, ha, hd, hq⟩
    exact ⟨m, hq, b, hb, ha, hd⟩
  refine (leafAvg_relabelL_le Q I.r f (ref ω) id V (σ.2 ω) fun u m ρ ρ' hu hq =>
    hlast (x σ.1 ω) u m ρ ρ' hu hq).trans (le_of_eq ?_)
  refine Eq.trans (leafAvg_congr_len (G := fun u s' => prob (fun u => Q u ∧ G u) (.ret (id u)) s')
    (fun u s' _ => ?_) _) ?_
  · exact ite_one_zero_iff Iff.rfl
  exact (prob_rounds _ _ _ _).symm

end Outer

section Schedule

variable {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

theorem schedSteps_last : ∀ (j : ℕ) (t : List (List (SMsg T) × ((k : Fin r) → CT k))) (ms : List (SMsg T))
    (ρ ρ' : (k : Fin r) → CT k), j + t.length = r →
    schedSteps T CT j (t ++ [(ms, ρ)]) = schedSteps T CT j (t ++ [(ms, ρ')])
  | j, [], ms, ρ, ρ', hj => by
      have hr : ¬ j < r := by simp at hj; omega
      simp [schedSteps, hr]
  | j, (ms₀, ρ₀) :: t, ms, ρ, ρ', hj => by
      have ih := schedSteps_last (j + 1) t ms ρ ρ' (by simp at hj; omega)
      simp only [List.cons_append, schedSteps, ih]

theorem schedSteps_lastCoin {t t' : List (List (SMsg T) × ((k : Fin r) → CT k))} (hl : t.length = r + 1)
    (h : LastCoin t t') : schedSteps T CT 0 t = schedSteps T CT 0 t' := by
  rcases h with rfl | ⟨p, m, ρ, ρ', rfl, rfl⟩
  · rfl
  · exact schedSteps_last T CT 0 p m ρ ρ' (by simp at hl; omega)

/-- **`schedInner` ignores the last round's coin**: `schedSteps` reads round `j`'s coin only at `j < r`, and the last
round is `j = r`. -/
theorem schedInner_lastFree [∀ k, Fintype (CT k)] [∀ k, Nonempty (CT k)] {α : Type} (Stmt : Type)
    (g : Stmt → Game α) (p : Stmt → α → Prop) (holds : Stmt → Prop) : (schedInner T CT Stmt g p holds).LastFree := by
  rintro x u m ρ ρ' hu ⟨a, ha, hp⟩
  have hu' : u.length + 1 = r + 1 := hu
  exact ⟨a, Eq.mp (congrArg (fun l => PlaysT (g x) l a)
    (schedSteps_last T CT 0 u m ρ ρ' ((Nat.zero_add _).trans (Nat.add_right_cancel hu')))) ha, hp⟩

end Schedule

section Picks

open Model Accounting Exec

/-- **`collOf` reads its condition only up to `↔`.** -/
theorem collOf_congr {D : Type} {H : List UInt8 → D} {p q : Prop} {h : p → Merkle.Collision H}
    {h' : q → Merkle.Collision H} (e : p ↔ q) : collOf H p h = collOf H q h' := by
  cases propext e
  rfl

variable {F K : Type} [Field F] [Fintype F] [DecidableEq F] [Field K] [Fintype K] [DecidableEq K] {D Sl : Type}
  {nT r : ℕ} (T : Fin nT → Type) (CT : Fin r → Type)

omit [Fintype K] in
/-- **`clashPickCL` ignores the last round's coin**: `outCL` reads a transcript only through `schedSteps`. -/
theorem clashPickCL_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (cls : HClass) (regs : List (Region cls.kLog (cls.m - cls.kLog)))
    (sch : Schedule) (M : Hidden cls) : PickLastFree (r + 1) (clashPickCL T CT A H E Lf cls regs sch M) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outCL T CT A H E Lf cls regs sch M t = outCL T CT A H E Lf cls regs sch M t' := fun _ _ h => by
    unfold outCL
    rw [h]
  unfold clashPickCL
  exact collOf_congr (by rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)])

omit [Fintype K] in
/-- **`clashPickHL` ignores the last round's coin**: `outHL` reads a transcript only through `schedSteps`. -/
theorem clashPickHL_lastFree (A : Arith F K) (H : List UInt8 → D) (E : Merkle.Enc (List F) D)
    (Lf : Merkle.Leaf (List F) Sl D H) (e : ExecHolo) (x : HoloReg F D) :
    PickLastFree (r + 1) (clashPickHL T CT A H E Lf e x) := by
  intro u₁ u₂ u₁' u₂' h₁ h₂ c₁ c₂
  have hout : ∀ t t' : List (List (SMsg T) × ((k : Fin r) → CT k)), schedSteps T CT 0 t = schedSteps T CT 0 t' →
      outHL T CT A H E Lf e x t = outHL T CT A H E Lf e x t' := fun _ _ h => by
    unfold outHL
    rw [h]
  unfold clashPickHL
  exact collOf_congr (by rw [hout _ _ (schedSteps_lastCoin T CT h₁ c₁), hout _ _ (schedSteps_lastCoin T CT h₂ c₂)])

end Picks

end FlockSoundness.Discharge.Recursion
