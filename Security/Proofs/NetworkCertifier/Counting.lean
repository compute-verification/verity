import Models.NetworkCertifier.Assumptions
import Mathlib.Data.Set.Card
import Mathlib.Data.Fintype.Pi
import Mathlib.Algebra.BigOperators.Fin

/-!
# Theorem 1(b): the counting bound on accepted runs

`docs/network-transparency/timing-channel.md` §2.2. The operator reacts to everything it has observed, and the sender
knows the secret and everything observed. The bound is on the observation sequences of runs accepted up to the
horizon; nothing is said about runs with a violation (decision 1 stays abstract).
-/

namespace NetTiming

open Finset

variable {L K S : ℕ} {Z In Act Wire Adv : Type}

/-! ### Encard through an injection into a finset -/

theorem encard_le_card_of_injOn {α β : Type} {s : Set α} (f : α → β) (hf : Set.InjOn f s) (t : Finset β)
    (hmaps : ∀ a ∈ s, f a ∈ t) : s.encard ≤ (t.card : ℕ∞) := by
  rw [← hf.encard_image, ← Set.encard_coe_eq_coe_finsetCard]
  exact Set.encard_le_encard (by rintro _ ⟨a, ha, rfl⟩; exact hmaps a ha)

/-! ### The tree lemma -/

/-- **The tree lemma.** Take a set of length-`n` sequences in which, after any prefix of length `t`, the next element
takes at most `N t` values. It has at most `∏_{t<n} N t` elements. -/
theorem encard_le_prod_of_branching {α : Type} (n : ℕ) (Sq : Set (List α)) (N : ℕ → ℕ)
    (hlen : ∀ o ∈ Sq, o.length = n)
    (hN : ∀ t < n, ∀ h : List α, {x | ∃ o ∈ Sq, o.take (t + 1) = h ++ [x]}.encard ≤ N t) :
    Sq.encard ≤ ((∏ t ∈ Finset.range n, N t : ℕ) : ℕ∞) := by
  classical
  let Br : ℕ → List α → Set α := fun t h => {x | ∃ o ∈ Sq, o.take (t + 1) = h ++ [x]}
  have hfin : ∀ t < n, ∀ h, (Br t h).Finite ∧ (Br t h).ncard ≤ N t := fun t ht h =>
    Set.encard_le_coe_iff_finite_ncard_le.mp (hN t ht h)
  let idx : ℕ → List α → α → ℕ := fun t h x =>
    if hf : (Br t h).Finite then (if hx : x ∈ hf.toFinset then (hf.toFinset.equivFin ⟨x, hx⟩ : ℕ) else 0) else 0
  have idx_lt : ∀ t < n, ∀ h, ∀ x ∈ Br t h, idx t h x < N t := by
    intro t ht h x hx
    obtain ⟨hf, hle⟩ := hfin t ht h
    have hx' : x ∈ hf.toFinset := hf.mem_toFinset.mpr hx
    simp only [idx, hf, hx', ↓reduceDIte]
    have := (hf.toFinset.equivFin ⟨x, hx'⟩).2
    rw [Set.ncard_eq_toFinset_card _ hf] at hle
    omega
  have idx_inj : ∀ t < n, ∀ h, Set.InjOn (idx t h) (Br t h) := by
    intro t ht h x hx y hy hxy
    obtain ⟨hf, -⟩ := hfin t ht h
    have hx' : x ∈ hf.toFinset := hf.mem_toFinset.mpr hx
    have hy' : y ∈ hf.toFinset := hf.mem_toFinset.mpr hy
    simp only [idx, hf, hx', hy', ↓reduceDIte] at hxy
    have := hf.toFinset.equivFin.injective (Fin.ext hxy)
    exact congrArg Subtype.val this
  let code : List α → Fin n → ℕ := fun o t => match o[(t : ℕ)]? with
    | some x => idx t (o.take t) x
    | none => 0
  have mem_br : ∀ o ∈ Sq, ∀ t (ht : t < o.length), o[t] ∈ Br t (o.take t) := by
    intro o ho t ht
    exact ⟨o, ho, (List.take_concat_get' o t ht).symm⟩
  have code_eq : ∀ o (ho : o ∈ Sq) (t : Fin n),
      code o t = idx t (o.take t) (o[(t : ℕ)]'(by rw [hlen o ho]; exact t.2)) := by
    intro o ho t
    have ht : (t : ℕ) < o.length := by rw [hlen o ho]; exact t.2
    simp only [code, List.getElem?_eq_getElem ht]
  have hinj : Set.InjOn code Sq := by
    intro o ho o' ho' hcode
    have hl : o.length = n := hlen o ho
    have hl' : o'.length = n := hlen o' ho'
    have key : ∀ t ≤ n, o.take t = o'.take t := by
      intro t
      induction t with
      | zero => simp
      | succ t ih =>
        intro ht
        have ht1 : t < o.length := by omega
        have ht2 : t < o'.length := by omega
        have hpre := ih (by omega)
        rw [← List.take_concat_get' o t ht1, ← List.take_concat_get' o' t ht2, hpre]
        congr 1
        have hc := congrFun hcode ⟨t, by omega⟩
        rw [code_eq o ho, code_eq o' ho'] at hc
        simp only at hc
        rw [hpre] at hc
        have hm := mem_br o ho t ht1
        have hm' := mem_br o' ho' t ht2
        rw [hpre] at hm
        rw [idx_inj t (by omega) _ hm hm' hc]
    have := key n le_rfl
    rwa [← hl, List.take_length, hl, ← hl', List.take_length] at this
  calc Sq.encard ≤ ((Fintype.piFinset fun t : Fin n => range (N t)).card : ℕ∞) := by
        apply encard_le_card_of_injOn code hinj
        intro o ho
        rw [Fintype.mem_piFinset]
        intro t
        rw [code_eq o ho, mem_range]
        exact idx_lt t t.2 _ _ (mem_br o ho t _)
    _ = ((∏ t ∈ Finset.range n, N t : ℕ) : ℕ∞) := by
        rw [Fintype.card_piFinset]
        simp only [card_range]
        rw [Fin.prod_univ_eq_prod_range (fun t => N t) n]

/-! ### Runs -/

namespace Interaction

variable (I : Interaction L K S Z In Act Wire)

theorem trace_succ (z : Z) (k : ℕ) :
    I.trace z (k + 1) =
      ((I.trace z k).1 ++ [I.certifier.obs k ((I.trace z k).2.1 ++ [I.op k (I.trace z k).1])
          ((I.trace z k).2.2.1 ++ [I.snd z k (I.trace z k).1])],
        (I.trace z k).2.1 ++ [I.op k (I.trace z k).1],
        (I.trace z k).2.2.1 ++ [I.snd z k (I.trace z k).1],
        (I.trace z k).2.2.2 ++ [I.certifier.wire k ((I.trace z k).2.1 ++ [I.op k (I.trace z k).1])
          ((I.trace z k).2.2.1 ++ [I.snd z k (I.trace z k).1])]) := rfl

theorem trace_length (z : Z) (n : ℕ) :
    (I.trace z n).1.length = n ∧ (I.trace z n).2.1.length = n ∧ (I.trace z n).2.2.1.length = n ∧
      (I.trace z n).2.2.2.length = n := by
  induction n with
  | zero => simp [trace]
  | succ k ih => rw [trace_succ]; simp [ih.1, ih.2.1, ih.2.2.1, ih.2.2.2]

theorem length_obsRun (z : Z) (n : ℕ) : (I.obsRun z n).length = n := (I.trace_length z n).1

theorem length_actRun (z : Z) (n : ℕ) : (I.actRun z n).length = n := (I.trace_length z n).2.2.1

theorem obsRun_succ (z : Z) (k : ℕ) : ∃ x, I.obsRun z (k + 1) = I.obsRun z k ++ [x] := ⟨_, rfl⟩

theorem actRun_succ (z : Z) (k : ℕ) : ∃ x, I.actRun z (k + 1) = I.actRun z k ++ [x] := ⟨_, rfl⟩

theorem obsRun_take (z : Z) {m n : ℕ} (h : m ≤ n) : (I.obsRun z n).take m = I.obsRun z m := by
  induction n with
  | zero => have : m = 0 := by omega
            subst this; simp [obsRun, trace]
  | succ k ih =>
    rcases Nat.lt_or_ge m (k + 1) with hm | hm
    · obtain ⟨x, hx⟩ := I.obsRun_succ z k
      rw [hx, List.take_append_of_le_length (by rw [length_obsRun]; omega), ih (by omega)]
    · have : m = k + 1 := by omega
      subst this; rw [List.take_of_length_le (by rw [length_obsRun])]

theorem actRun_take (z : Z) {m n : ℕ} (h : m ≤ n) : (I.actRun z n).take m = I.actRun z m := by
  induction n with
  | zero => have : m = 0 := by omega
            subst this; simp [actRun, trace]
  | succ k ih =>
    rcases Nat.lt_or_ge m (k + 1) with hm | hm
    · obtain ⟨x, hx⟩ := I.actRun_succ z k
      rw [hx, List.take_append_of_le_length (by rw [length_actRun]; omega), ih (by omega)]
    · have : m = k + 1 := by omega
      subst this; rw [List.take_of_length_le (by rw [length_actRun])]

/-- The observation at step `k` is the certifiers' record at `k`, over the ingress and the actions through `k`. -/
theorem obsRun_getElem (z : Z) {n k : ℕ} (hk : k < n) :
    (I.obsRun z n)[k]'(by rw [length_obsRun]; exact hk) =
      I.certifier.obs k (I.trace z (k + 1)).2.1 (I.actRun z (k + 1)) := by
  have h1 : (I.obsRun z n)[k]'(by rw [length_obsRun]; exact hk) =
      ((I.obsRun z n).take (k + 1))[k]'(by rw [List.length_take, length_obsRun]; omega) := by
    rw [List.getElem_take]
  rw [h1]
  simp only [I.obsRun_take z (show k + 1 ≤ n by omega)]
  simp only [obsRun, actRun, trace_succ]
  rw [List.getElem_append_right (by rw [(I.trace_length z k).1])]
  simp [(I.trace_length z k).1]

theorem mem_obsRun (z : Z) (n : ℕ) (x : Obs L K S) (hx : x ∈ I.obsRun z n) :
    ∃ k xs as, x = I.certifier.obs k xs as := by
  obtain ⟨k, hk, rfl⟩ := List.getElem_of_mem hx
  have hk' : k < n := by rwa [length_obsRun] at hk
  exact ⟨k, _, _, I.obsRun_getElem z hk'⟩

/-- Under W2 the wire is the observations, mapped step by step through `view`. -/
theorem wireRun_eq (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view) (z : Z) (n : ℕ) :
    I.wireRun z n = (I.obsRun z n).mapIdx view := by
  induction n with
  | zero => simp [wireRun, obsRun, trace]
  | succ k ih =>
    simp only [wireRun, obsRun] at ih ⊢
    rw [trace_succ]
    simp only
    rw [ih, List.mapIdx_append, hW2]
    simp [(I.trace_length z k).1]

end Interaction

/-! ### Acceptance is closed under prefixes -/

theorem ingressUpTo_congr {In : Type} (op : ℕ → List (Obs L K S) → In) {o o' : List (Obs L K S)} {t : ℕ}
    (h : o.take t = o'.take t) : ingressUpTo op o t = ingressUpTo op o' t := by
  unfold ingressUpTo
  apply List.map_congr_left
  intro i hi
  rw [List.mem_range] at hi
  have e1 : o.take i = (o.take t).take i := by rw [List.take_take, Nat.min_eq_left (by omega)]
  have e2 : o'.take i = (o'.take t).take i := by rw [List.take_take, Nat.min_eq_left (by omega)]
  rw [e1, e2, h]

theorem accepted_take {In Adv : Type} (P : Program L S In Adv) (op : ℕ → List (Obs L K S) → In) {k m : ℕ}
    (hm : m ≤ k) {o : List (Obs L K S)} {a : Adv} (h : Accepted P op k o a) : Accepted P op m (o.take m) a := by
  intro t ht
  obtain ⟨hlen, hrows, hstat⟩ := h t (by omega)
  have hlen' : t < (o.take m).length := by rw [List.length_take]; omega
  refine ⟨hlen', ?_, ?_⟩
  · rw [List.getElem_take, hrows]
    congr 1
    apply ingressUpTo_congr
    rw [List.take_take, Nat.min_eq_left (by omega)]
  · intro κ; rw [List.getElem_take]; exact hstat κ

/-- **Theorem 1(b), accepted runs.** Take the observation sequences of the runs accepted up to the horizon `n`,
whatever the sender's strategy (over all secrets) and however the operator reacts. Suppose that after any prefix of
length `t`, the runs accepted up to `t + 1` continue with at most `N t` distinct observations; this is the doc's
`N_acc`, defined by prefix acceptance. Then there are at most `∏_{t<n} N t` such sequences. -/
theorem encard_acceptedRuns_le (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ) (N : ℕ → ℕ)
    (hN : ∀ t < n, ∀ h : List (Obs L K S),
      {x | ∃ z, AcceptedUpTo P I.op (t + 1) (I.obsRun z (t + 1)) ∧ I.obsRun z (t + 1) = h ++ [x]}.encard ≤ N t) :
    {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard ≤ ((∏ t ∈ Finset.range n, N t : ℕ) : ℕ∞) := by
  apply encard_le_prod_of_branching n _ N
  · rintro o ⟨z, rfl, -⟩; exact I.length_obsRun z n
  · intro t ht h
    refine le_trans (Set.encard_le_encard ?_) (hN t ht h)
    rintro x ⟨o, ⟨z, rfl, a, ha⟩, hx⟩
    refine ⟨z, ⟨a, ?_⟩, ?_⟩
    · rw [← I.obsRun_take z (show t + 1 ≤ n by omega)]
      exact accepted_take P I.op (by omega) ha
    · rw [← I.obsRun_take z (show t + 1 ≤ n by omega)]; exact hx

/-- **Zero error, from the observations.** A decoder reads the observations of the first `n` steps (the published
records, jitter and statuses) and recovers every secret in `Zs`, all of whose runs are accepted up to `n`. Then `Zs`
has at most as many elements as there are accepted observation sequences. This needs no W2, and the wire version below
follows from it. -/
theorem encard_obsDecodable_le (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List (Obs L K S) → Z) (hdec : ∀ z ∈ Zs, dec (I.obsRun z n) = z) :
    Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard := by
  have hinj : Set.InjOn (fun z => I.obsRun z n) Zs := by
    intro z hz z' hz' h
    simp only at h
    rw [← hdec z hz, ← hdec z' hz', h]
  rw [← hinj.encard_image]
  exact Set.encard_le_encard (by rintro _ ⟨z, hz, rfl⟩; exact ⟨z, rfl, hacc z hz⟩)

/-- **Zero error, through the wire (uses W2).** A decoder reads the wire of the first `n` steps and recovers every
secret in `Zs`, all of whose runs are accepted up to `n`. Then `Zs` has at most as many elements as there are accepted
observation sequences. -/
theorem encard_decodable_le (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (n : ℕ)
    (view : ℕ → Obs L K S → Wire) (hW2 : Regenerates I.certifier view)
    (Zs : Set Z) (hacc : ∀ z ∈ Zs, AcceptedUpTo P I.op n (I.obsRun z n))
    (dec : List Wire → Z) (hdec : ∀ z ∈ Zs, dec (I.wireRun z n) = z) :
    Zs.encard ≤ {o | ∃ z, o = I.obsRun z n ∧ AcceptedUpTo P I.op n o}.encard :=
  encard_obsDecodable_le I P n Zs hacc (fun o => dec (o.mapIdx view))
    (fun z hz => by rw [← I.wireRun_eq view hW2]; exact hdec z hz)

/-- **W6 on accepted runs.** Under fail-closed certification, a run accepted up to `n` takes no fault action at any step
`i` whose window closes before `n`. (`hwin`: every step lies in its own window, which ends no earlier.) -/
theorem no_fault_of_accepted (I : Interaction L K S Z In Act Wire) (P : Program L S In Adv) (winEnd : ℕ → ℕ)
    (hwin : ∀ i, i ≤ winEnd i)
    (hW6 : FailClosed I.certifier winEnd) (z : Z) (n : ℕ) (hacc : AcceptedUpTo P I.op n (I.obsRun z n))
    (i : ℕ) (hi : i < (I.actRun z n).length) (hend : winEnd i < n) :
    ¬ I.certifier.fault (I.actRun z n)[i] := by
  intro hfault
  obtain ⟨a, ha⟩ := hacc
  set e := winEnd i with he
  have hie : i ≤ e := hwin i
  have hlen : (I.actRun z (e + 1)).length = e + 1 := I.length_actRun z (e + 1)
  have hi' : i < (I.actRun z (e + 1)).length := by omega
  have hsame : (I.actRun z (e + 1))[i] = (I.actRun z n)[i] := by
    have : (I.actRun z n).take (e + 1) = I.actRun z (e + 1) := I.actRun_take z (by omega)
    simp only [← this, List.getElem_take]
  obtain ⟨-, hbad⟩ := hW6 i e (I.trace z (e + 1)).2.1 (I.actRun z (e + 1)) hi' hlen hie le_rfl (by rw [hsame]; exact hfault)
  obtain ⟨κ, hκ⟩ := hbad rfl
  obtain ⟨hlt, -, hstat⟩ := ha e hend
  apply hκ
  rw [← I.obsRun_getElem z hend]
  exact hstat κ

end NetTiming
