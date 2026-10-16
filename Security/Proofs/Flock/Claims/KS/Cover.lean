import Proofs.Flock.Claims.KS.Range

/-!
# Extraction from a filled word, and the coverage `nMin` asks

The old soundness theorems bound the probability that the table the extractor fills from `k` accepted reruns'
openings, `0` where no rerun opened a position (`extractTable`, `extractTableS`), is not within the level-0 radius `θ`
of a codeword whose message satisfies the statement (`Committed`, `failProbAccB`, `joint_le_accB`). `Extracts` asks
something else of the same columns: at least `nMin M` distinct opened positions, a fraction `agree` of them agreeing
with `enc M`.

* **The extraction lemma** (`agree_of_fill`, `extracts_of_fill`): when the opened columns, each position once, filled
  with a column `z` elsewhere, differ from `enc M` at `e` positions at most, they agree with `enc M` at all but `e` of
  their positions. So `Extracts` follows from the old bound (`e = θ·N₀`) once the reruns' columns cover `nMin M`
  positions and `θ·N₀ ≤ (1 − agree)·#opened`. That coverage is what the old theorems do not give: their filled table
  is close to the codeword however few positions the reruns opened.
* **Coverage is not implied** (`E_none_of_few`, `PrBad_of_few`, `acc_le_of_KS`): the waits gather only accepted
  reruns, so a prover whose accepted runs open columns at fewer than `nMin` positions in all leaves every wait short
  of `nMin`. At a decoder that meets `ListComplete` and lists only traces whose `nMin` the distinct positions reach
  (`refuseFew`), `E^rep` then returns nothing on every judged run, and `Flock.KS` bounds that prover's acceptance by
  `κ + √(γ·Pr[B])`.
* **At the deployed records** (`plain_acc_le_of_KS`, `zk_acc_le_of_KS`): every plain trace has `nMin ≥ 1012`, every
  masked one `nMin ≥ 7991`, so `Flock.KS` bounds by `κ + √(γ·Pr[B])` the acceptance of every prover whose accepted
  runs open table 0's level-0 columns at `1011` (masked: `7990`) positions at most. At `m = 22`, where the plain
  level-0 codeword has `1024` positions, the honest prover that sends a failing proof whenever one of its level-0
  queries lands in `13` fixed positions is such a prover, accepted with probability `≈ 2^{−8}`
  (`Numbers.miss_ge`: `≥ 1/64` per rep).
-/

namespace Claims

/-- **The waits return nothing when every wait fails.** -/
theorem firstWait_none_of {R W Op : Type} (H : Msg → Msg) (run : R → Trace) (k : ℕ) (opens : Seen → List Op)
    (assemble : List Op → Option W) (check : W → Bool) :
    ∀ rss : List (List R), (∀ rs ∈ rss, ¬ WaitOk H run k opens assemble check rs) →
      ((firstWait k opens assemble check rss).eval H run).1 = none := by
  intro rss
  induction rss with
  | nil => intro _; simp [firstWait, RP.eval]
  | cons rs rss ih =>
    intro h
    have ih' := ih fun rs' h' => h rs' (List.mem_cons_of_mem _ h')
    simp only [firstWait]
    rw [(RP.eval_bind H run _ _).1]
    split
    · rename_i w hw
      split
      · rename_i hc
        exact absurd ⟨w, hw, hc⟩ (h rs (by simp))
      · exact ih'
    · exact ih'

/-- A wait's runs are accepted reruns: every run it collects satisfies what every accepted run does. -/
theorem waitRes_acc {R : Type} (run : R → Trace) (k : ℕ) (G : Seen → Prop)
    (hG : ∀ r, (run r).out = .acc → G (run r).seen) :
    ∀ (rs : List R) (acc : List Seen), (∀ v ∈ acc, G v) → ∀ v ∈ waitRes run k rs acc, G v := by
  intro rs
  induction rs with
  | nil => intro acc h; simpa [waitRes] using h
  | cons r rs ih =>
    intro acc h
    simp only [waitRes]
    split_ifs with h1 h2
    · exact h
    · refine ih _ fun v hv => ?_
      rcases List.mem_append.1 hv with hv | hv
      · exact h v hv
      · rw [List.mem_singleton] at hv
        subst hv
        exact hG r h2
    · exact ih _ h

namespace Flock

variable {Col T : Type} [DecidableEq Col]

/-! ## The extraction lemma -/

/-- **The opened columns as a word**: the column opened at `q`, and `z` where none is. -/
def fillAt (z : Col) (os : List (ℕ × Col)) (q : ℕ) : Col := ((os.find? fun p => decide (p.1 = q)).map Prod.snd).getD z

omit [DecidableEq Col] in
theorem fillAt_of_mem (z : Col) {os : List (ℕ × Col)} (hnd : (os.map Prod.fst).Nodup) {p : ℕ × Col} (hp : p ∈ os) :
    fillAt z os p.1 = p.2 := by
  unfold fillAt
  obtain ⟨p', hp'⟩ : ∃ p', os.find? (fun p' => decide (p'.1 = p.1)) = some p' :=
    Option.isSome_iff_exists.1 (List.find?_isSome.2 ⟨p, hp, by simp⟩)
  have h1 : p'.1 = p.1 := by simpa using List.find?_some hp'
  have h2 := List.mem_of_find?_eq_some hp'
  have : p' = p := List.inj_on_of_nodup_map hnd h2 hp h1
  rw [hp', this]
  rfl

omit [DecidableEq Col] in
/-- At most `A.card` distinct positions in `A`. -/
theorem nodup_length_le {os : List (ℕ × Col)} (hnd : (os.map Prod.fst).Nodup) {A : Finset ℕ}
    (h : ∀ p ∈ os, p.1 ∈ A) : os.length ≤ A.card := by
  have hl : os.length = ((os.map Prod.fst).toFinset).card := by
    rw [List.toFinset_card_of_nodup hnd, List.length_map]
  rw [hl]
  refine Finset.card_le_card fun q hq => ?_
  obtain ⟨p, hp, rfl⟩ := List.mem_map.1 (List.mem_toFinset.1 hq)
  exact h p hp

/-- **Agreement on the opened columns from a close filled word**: opened columns at distinct positions below `N` agree
with `enc M` at all but as many positions as their filled word differs from `enc M` below `N`. -/
theorem agree_of_fill (enc : T → ℕ → Col) (M : T) (z : Col) (N : ℕ) (os : List (ℕ × Col))
    (hnd : (os.map Prod.fst).Nodup) (hlt : ∀ p ∈ os, p.1 < N) :
    os.length ≤ (os.filter fun p => decide (enc M p.1 = p.2)).length +
      ((Finset.range N).filter fun q => fillAt z os q ≠ enc M q).card := by
  set bad := os.filter fun p => !decide (enc M p.1 = p.2)
  have hsplit : os.length = (os.filter fun p => decide (enc M p.1 = p.2)).length + bad.length :=
    List.length_eq_length_filter_add _
  have hbnd : (bad.map Prod.fst).Nodup := (hnd.sublist ((List.filter_sublist).map _))
  have hb : bad.length ≤ ((Finset.range N).filter fun q => fillAt z os q ≠ enc M q).card := by
    refine nodup_length_le hbnd fun p hp => ?_
    have hp' := List.mem_filter.1 hp
    rw [Finset.mem_filter, Finset.mem_range, fillAt_of_mem z hnd hp'.1]
    refine ⟨hlt p hp'.1, fun h => ?_⟩
    simp [h] at hp'
  omega

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **The extraction lemma**: on accepted runs `vs`, `Extracts` holds at a trace `M` that reads as a witness when the
filled word of their opened columns (each position once, all below `N`) differs from `enc M` at `e` positions below
`N` at most, the columns cover `nMin M` positions, and `e ≤ (1 − agree)·#positions`. -/
theorem extracts_of_fill (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) (M : T) (s : Stmt)
    (hs : D.parse x = some s) (hrel : RelS H s (D.read M)) (z : Col) (N e : ℕ)
    (hlt : ∀ p ∈ vs.flatMap D.cols, p.1 < N)
    (hclose : ((Finset.range N).filter fun q => fillAt z (dedupPos (vs.flatMap D.cols)) q ≠ D.enc M q).card ≤ e)
    (hn : D.nMin M ≤ (dedupPos (vs.flatMap D.cols)).length)
    (he : (e : ℝ) ≤ (1 - D.agree) * (dedupPos (vs.flatMap D.cols)).length) : Extracts H D x vs := by
  refine ⟨M, s, hs, hrel, hn, ?_⟩
  have h := agree_of_fill D.enc M z N (dedupPos (vs.flatMap D.cols)) (dedupPos_nodup _)
    (fun p hp => hlt p ((List.pwFilter_sublist _).subset hp))
  have h' : ((dedupPos (vs.flatMap D.cols)).length : ℝ) ≤
      ((dedupPos (vs.flatMap D.cols)).filter fun p => decide (D.enc M p.1 = p.2)).length + e := by
    exact_mod_cast h.trans (Nat.add_le_add_left hclose _)
  linarith

/-! ## Coverage is not implied -/

/-- **A decoder that lists only traces whose `nMin` the distinct opened positions reach**: `d`'s list, filtered. -/
def refuseFew (nMin : T → ℕ) (d : List (ℕ × Col) → List T) (os : List (ℕ × Col)) : List T :=
  (d os).filter fun M => decide (nMin M ≤ (dedupPos os).length)

/-- Filtering a decoder that meets `ListComplete` by `nMin` still meets it. -/
theorem listComplete_refuseFew {enc : T → ℕ → Col} {agree : ℝ} {nMin : T → ℕ} {d : List (ℕ × Col) → List T}
    (h : ListComplete enc agree nMin d) : ListComplete enc agree nMin (refuseFew nMin d) := by
  intro os M hnd hn hag
  unfold refuseFew
  rw [List.mem_filter, dedupPos_of_nodup os hnd]
  exact ⟨h os M hnd hn hag, decide_eq_true hn⟩

omit [DecidableEq Col] in
/-- **`E^rep` fails on a prover whose accepted runs open few positions**: if every accepted run at `x` opens columns
only at positions in `A`, and every trace that reads as a witness of `x`'s statement has `nMin` above `A`'s size, then
`E^rep` at `refuseFew` returns nothing, at every tape, judged run and extractor coin. -/
theorem E_none_of_few (H : Msg → Msg) (D : Deploy Coin Col T) (d : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    (P : Prover Ω) (A : Finset ℕ)
    (hA : ∀ ω r, (run H P D.V x ω r).out = .acc → ∀ p ∈ D.cols (run H P D.V x ω r).seen, p.1 ∈ A)
    (hfew : ∀ M s, D.parse x = some s → RelS H s (D.read M) → A.card < D.nMin M)
    (ω : Ω) (r₀ : Fin (D.V.k x) → Coin) (c : (E H D (refuseFew D.nMin d)).C x) :
    ((E H D (refuseFew D.nMin d)).eval H P x ω r₀ c).1 = none := by
  simp only [Extractor.eval, E, ERep, RP.eval]
  split
  · refine firstWait_none_of H _ _ _ _ _ _ fun rs _ hok => ?_
    obtain ⟨w, hw, -⟩ := hok
    rw [waitAcc_res] at hw
    have hall : ∀ v ∈ waitRes (run H P D.V x ω) (D.k x) rs [], ∀ p ∈ D.cols v, p.1 ∈ A :=
      waitRes_acc _ _ _ (hA ω) rs [] (by simp)
    set os := (waitRes (run H P D.V x ω) (D.k x) rs []).flatMap D.cols
    have hos : (dedupPos os).length ≤ A.card := by
      refine nodup_length_le (dedupPos_nodup os) fun p hp => ?_
      obtain ⟨v, hv, hpv⟩ := List.mem_flatMap.1 ((List.pwFilter_sublist _).subset hp)
      exact hall v hv p hpv
    unfold assemble at hw
    split at hw
    · cases hw
    · rename_i s hs
      have hrel : RelS H s w := by simpa using List.find?_some hw
      obtain ⟨M, hM, rfl⟩ := List.mem_map.1 (List.mem_of_find?_eq_some hw)
      have hn := of_decide_eq_true (List.mem_filter.1 hM).2
      have := hfew M s hs hrel
      omega
  · simp [RP.eval]

omit [DecidableEq Col] in
/-- **`Pr[acc ∧ ¬R(x, E^rep)]` is the acceptance** of such a prover, at `refuseFew`. -/
theorem PrBad_of_few (H : Msg → Msg) (D : Deploy Coin Col T) (d : List (ℕ × Col) → List T) (x : Msg) {Ω : Type}
    [Fintype Ω] (P : Prover Ω) (A : Finset ℕ)
    (hA : ∀ ω r, (run H P D.V x ω r).out = .acc → ∀ p ∈ D.cols (run H P D.V x ω r).seen, p.1 ∈ A)
    (hfew : ∀ M s, D.parse x = some s → RelS H s (D.read M) → A.card < D.nMin M) :
    PrBad H D.V (R H D.parse) (E H D (refuseFew D.nMin d)) x P =
      Pr fun s : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x s.1 s.2).out = .acc := by
  unfold PrBad
  rw [← Pr_drop (γ := (E H D (refuseFew D.nMin d)).C x)
    (fun (ω : Ω) (r : Fin (D.V.k x) → Coin) => (run H P D.V x ω r).out = .acc)]
  refine Pr_congr fun s => ?_
  rw [E_none_of_few H D d x P A hA hfew s.1 s.2.1 s.2.2]
  simp [Rel.ext]

/-- **What `Flock.KS` asks of a prover whose accepted runs open few positions**: its acceptance is at most
`κ + √(γ·Pr[B])`. -/
theorem acc_le_of_KS (H : Msg → Msg) (D : Deploy Coin Col T) (hKS : Flock.KS H D) (d : List (ℕ × Col) → List T)
    (hd : ListComplete D.enc D.agree D.nMin d) (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω)
    (A : Finset ℕ)
    (hA : ∀ ω r, (run H P D.V x ω r).out = .acc → ∀ p ∈ D.cols (run H P D.V x ω r).seen, p.1 ∈ A)
    (hfew : ∀ M s, D.parse x = some s → RelS H s (D.read M) → A.card < D.nMin M) :
    Pr (fun s : Ω × (Fin (D.V.k x) → Coin) => (run H P D.V x s.1 s.2).out = .acc) ≤
      κ D x + Real.sqrt (γ D x * PrB H D.V P x) := by
  rw [← PrBad_of_few H D d x P A hA hfew]
  exact hKS _ (listComplete_refuseFew hd) x Ω P

end Flock

end Claims

namespace FlockDeploy.KS

open Flock

theorem plainCols_mem : ∀ i : Claims.Flock.PlainIdx,
    Claims.Flock.plainCols i ∈ [9, 10, 11, 12, 13, 14, 17, 18, 19, 20, 21, 22] := by
  decide +kernel

/-- **Every plain trace has `nMin ≥ 1012`**: `1012` at `m = 22`, more at every larger `m`. -/
theorem plain_nMin_ge (rel : Readable.Parsed → Readable.Relation) (M : Claims.Flock.PlainTrace) :
    1012 ≤ (deployPlain rel).nMin M := by
  show 1012 ≤ Claims.Flock.plainNMinD Claims.Flock.Numbers.agree₀ M
  rw [Claims.Flock.plainNMinD_eq, Claims.Flock.Numbers.johnsonCount_agree₀ (plainCols_mem M.1)]
  have h := plainCols_mem M.1
  simp only [List.mem_cons, List.not_mem_nil, or_false] at h
  rcases h with h | h | h | h | h | h | h | h | h | h | h | h <;> rw [h] <;> decide

theorem zkAgree_sq_le : Claims.Flock.zkAgree ^ 2 ≤ 4651 / 7990 := by
  rw [Claims.Flock.zkAgree_eq]
  have h0 : 0 ≤ Real.sqrt (4650 / 8192) := Real.sqrt_nonneg _
  have h1 : Real.sqrt (4650 / 8192) ≤ 0.7575 := by
    rw [Real.sqrt_le_left (by norm_num)]
    norm_num
  nlinarith

/-- **Every masked trace has `nMin ≥ 7991`**: its code's dimension is at least `4650` (`zkDim_ge`), and
`zkAgree² ≤ 4651/7990`. -/
theorem zk_nMin_ge (rel : Readable.Parsed → Readable.Relation) (M : Claims.Flock.ZkTrace) :
    7991 ≤ (deployZk rel).nMin M := by
  show 7991 ≤ Claims.Flock.zkNMinD Claims.Flock.zkAgree M
  rw [Claims.Flock.zkNMinD_eq]
  unfold Claims.johnsonCount
  have hk : (4651 : ℝ) ≤ (2 ^ Claims.Flock.zkCols M.1 + Claims.Flock.zkPad M.1 : ℕ) + 1 := by
    have := Claims.Flock.zkDim_ge M.1
    have : (4650 : ℝ) ≤ (2 ^ Claims.Flock.zkCols M.1 + Claims.Flock.zkPad M.1 : ℕ) := by exact_mod_cast this
    linarith
  have ha : 0 < Claims.Flock.zkAgree ^ 2 := pow_pos Claims.Flock.zkAgree_pos 2
  have hq : (7990 : ℝ) ≤ (((2 ^ Claims.Flock.zkCols M.1 + Claims.Flock.zkPad M.1 : ℕ) : ℝ) + 1) /
      Claims.Flock.zkAgree ^ 2 := by
    rw [le_div_iff₀ ha]
    have := zkAgree_sq_le
    nlinarith
  have : 7990 ≤ ⌊(((2 ^ Claims.Flock.zkCols M.1 + Claims.Flock.zkPad M.1 : ℕ) : ℝ) + 1) /
      Claims.Flock.zkAgree ^ 2⌋₊ := Nat.le_floor (by exact_mod_cast hq)
  push_cast at this ⊢
  omega

/-- **What `Flock.KS` asks at the deployed plain record**: every prover whose accepted runs open table 0's level-0
columns at `1011` positions at most, in all, is accepted with probability at most `κ + √(γ·Pr[B])`. -/
theorem plain_acc_le_of_KS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hKS : Claims.Flock.KS H (deployPlain rel)) (x : Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (A : Finset ℕ) (hcard : A.card ≤ 1011)
    (hA : ∀ ω r, (Claims.run H P (deployPlain rel).V x ω r).out = .acc →
      ∀ p ∈ (deployPlain rel).cols (Claims.run H P (deployPlain rel).V x ω r).seen, p.1 ∈ A) :
    Claims.Pr (fun s : Ω × (Fin ((deployPlain rel).V.k x) → Coin) =>
        (Claims.run H P (deployPlain rel).V x s.1 s.2).out = .acc) ≤
      Claims.Flock.κ (deployPlain rel) x + Real.sqrt (Claims.Flock.γ (deployPlain rel) x *
        Claims.PrB H (deployPlain rel).V P x) :=
  Claims.Flock.acc_le_of_KS H (deployPlain rel) hKS _ (listComplete_plain rel) x Ω P A hA
    fun M _ _ _ => by have := plain_nMin_ge rel M; omega

/-- **What `Flock.KS` asks at the deployed masked record**: every prover whose accepted runs open table 0's level-0
columns at `7990` positions at most, in all, is accepted with probability at most `κ + √(γ·Pr[B])`. -/
theorem zk_acc_le_of_KS (H : Claims.Msg → Claims.Msg) (rel : Readable.Parsed → Readable.Relation)
    (hKS : Claims.Flock.KS H (deployZk rel)) (x : Claims.Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω]
    (P : Claims.Prover Ω) (A : Finset ℕ) (hcard : A.card ≤ 7990)
    (hA : ∀ ω r, (Claims.run H P (deployZk rel).V x ω r).out = .acc →
      ∀ p ∈ (deployZk rel).cols (Claims.run H P (deployZk rel).V x ω r).seen, p.1 ∈ A) :
    Claims.Pr (fun s : Ω × (Fin ((deployZk rel).V.k x) → Coin) =>
        (Claims.run H P (deployZk rel).V x s.1 s.2).out = .acc) ≤
      Claims.Flock.κ (deployZk rel) x + Real.sqrt (Claims.Flock.γ (deployZk rel) x *
        Claims.PrB H (deployZk rel).V P x) :=
  Claims.Flock.acc_le_of_KS H (deployZk rel) hKS _ (listComplete_zk rel) x Ω P A hA
    fun M _ _ _ => by have := zk_nMin_ge rel M; omega

end FlockDeploy.KS
