module

public import Proofs.Properties.Verity.SampledTwo.Accepts

/-!
# The form's accept is the run's accept: the proof

`acceptsRun` proves `AcceptsRunOf`: `AcceptsRun` at the two-stage form's reading, from `PlayAccepts`, `SessionMsgs`,
`SourcesFit`, `ReadsOnly` and `CoinsCover`. At a play the form's check accepts and coins `r` agreeing with what it read:

* round 0's coin of `r` starts with the first stage's bytes, and `r`'s second-stage coins with the interior's: the
  coins `coinOf` and `stageCoinsOf` chose are those bytes padded with zeros (`CoinsCover`, `SourcesFit`), and `r`
  agrees with them on the bytes the run read (`starts_of_prefixAgree`, `stage_starts`). So `r`'s rounds 0 to `stage`
  and its sessions' rounds are coins `PlayAccepts` speaks of, and `proverSeq` on them is accepted;
* `proverSeq`'s run on `r` is that run: the two coin vectors agree on every round `readsAt` names (`ReadsOnly`), the
  sessions leaving no round between them (`slot_of_lt`);
* that run's messages are those the check read, round by round: `proverSeq`'s first messages (`[]`, the interior
  record, `[]`, the session statements), `SessionMsgs` in the sessions' rounds and `[]` in the verdict's. Each round
  has one: `SpotFlock.V` accepts only at its last round (`next_ne_acc`), so an accepted run has all its messages.

So `ProverAt` holds, and `acceptsRun_of_proverAt` gives `AcceptsRun`.
-/

@[expose] public section

namespace Verity.SampledTwo

open Claims

/-! ## Bytes -/

theorem size_bytesOf {n : ℕ} (f : Fin n → Fin 256) : (Properties.Spot.bytesOf f).size = n := by
  rw [← ByteArray.size_data]
  simp [Properties.Spot.bytesOf]

theorem getElem_bytesOf {n : ℕ} (f : Fin n → Fin 256) {k : ℕ} (hk : k < (Properties.Spot.bytesOf f).size) :
    (Properties.Spot.bytesOf f)[k] = UInt8.ofFin (f ⟨k, by rwa [size_bytesOf] at hk⟩) := by
  rw [ByteArray.getElem_eq_getElem_data]
  simp [Properties.Spot.bytesOf]

theorem size_src (c : FlockDeploy.Coin) : (SpotFlock.src c).size = SpotFlock.L := size_bytesOf _

theorem size_stageSrc {R : ℕ} (cs : Fin R → FlockDeploy.Coin) : (SpotFlock.stageSrc cs).size = R * SpotFlock.L :=
  size_bytesOf _

theorem L_pos : 0 < SpotFlock.L := by
  unfold SpotFlock.L
  omega

theorem stageWidth_le (b : ByteArray) (j : ℕ) : stageWidth b j ≤ SpotFlock.L := Nat.min_le_left _ _

/-- **A byte string whose first `b.size` bytes are `b` is `b` and then the rest.** -/
theorem eq_append_of_extract {a b : ByteArray} (hb : a.extract 0 b.size = b) (hn : b.size ≤ a.size) :
    a = b ++ a.extract b.size a.size :=
  calc a = a.extract 0 a.size := ByteArray.extract_zero_size.symm
    _ = a.extract 0 b.size ++ a.extract b.size a.size :=
        ByteArray.extract_eq_extract_append_extract b.size (Nat.zero_le _) hn
    _ = b ++ a.extract b.size a.size := by rw [hb]

theorem getElem_of_eq_append {a b u : ByteArray} (h : a = b ++ u) {k : ℕ} (hk : k < b.size) {ha : k < a.size} :
    a[k] = b[k] := by
  subst h
  exact ByteArray.getElem_append_left hk

/-- **Coins agreeing on their first `w` bytes have the same byte at each place before `w`.** -/
theorem getElem_of_prefixAgree {c c' : FlockDeploy.Coin} {w : ℕ} (hw : w ≤ SpotFlock.L) (h : prefixAgree w c c')
    {m : ℕ} (hm : m < w) :
    (SpotFlock.src c)[m]'(by rw [size_src]; omega) = (SpotFlock.src c')[m]'(by rw [size_src]; omega) := by
  have h' : (SpotFlock.src c).extract 0 (0 + w) = (SpotFlock.src c').extract 0 (0 + w) := by
    rw [Nat.zero_add]
    exact h
  rw [ByteArray.extract_eq_extract_iff_getElem (by rw [size_src]; omega) (by rw [size_src]; omega)] at h'
  simpa using h' m hm

/-- **Round 0's coin starts with the bytes the run read**: a coin agreeing on `b.size` bytes with one whose bytes start
with `b` starts with `b`. -/
theorem starts_of_prefixAgree {b : ByteArray} {c c' : FlockDeploy.Coin} (hb : ∃ u, SpotFlock.src c = b ++ u)
    (h : prefixAgree b.size c c') : ∃ u, SpotFlock.src c' = b ++ u := by
  obtain ⟨u, hu⟩ := hb
  have hn : b.size ≤ SpotFlock.L := by
    have := congrArg ByteArray.size hu
    rw [size_src, ByteArray.size_append] at this
    omega
  have h' : (SpotFlock.src c').extract 0 b.size = b := by
    have e := h
    unfold prefixAgree at e
    rw [hu, ByteArray.extract_append_eq_left rfl] at e
    exact e.symm
  exact ⟨_, eq_append_of_extract h' (by rw [size_src]; exact hn)⟩

/-- **Byte `k` of `R` rounds' coins is byte `k % L` of coin `k / L`.** -/
theorem getElem_stageSrc {R : ℕ} (cs : Fin R → FlockDeploy.Coin) {k : ℕ} (hk : k < R * SpotFlock.L) :
    (SpotFlock.stageSrc cs)[k]'(by rw [size_stageSrc]; exact hk) =
      (SpotFlock.src (cs ⟨k / SpotFlock.L, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hk)⟩))[k % SpotFlock.L]'
        (by rw [size_src]; exact Nat.mod_lt _ L_pos) := by
  have e1 := getElem_bytesOf (SpotFlock.stageByte cs) (k := k) (by rw [size_bytesOf]; exact hk)
  have e2 := getElem_bytesOf
    (SpotFlock.coinByte (cs ⟨k / SpotFlock.L, Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hk)⟩))
    (k := k % SpotFlock.L) (by rw [size_bytesOf]; exact Nat.mod_lt _ L_pos)
  exact e1.trans e2.symm

/-- **The second stage's coins start with the bytes the run read**: coins agreeing coin by coin, on the bytes `b`
carries of each (`stageWidth`), with coins whose bytes start with `b` start with `b`. -/
theorem stage_starts {R : ℕ} {b : ByteArray} {cs c₁ : Fin R → FlockDeploy.Coin}
    (hb : ∃ u, SpotFlock.stageSrc cs = b ++ u)
    (hj : ∀ j : Fin R, prefixAgree (stageWidth b j.val) (cs j) (c₁ j)) :
    ∃ u, SpotFlock.stageSrc c₁ = b ++ u := by
  obtain ⟨u, hu⟩ := hb
  have hn : b.size ≤ R * SpotFlock.L := by
    have := congrArg ByteArray.size hu
    rw [size_stageSrc, ByteArray.size_append] at this
    omega
  have e : ∀ k (hk : k < b.size), (SpotFlock.stageSrc c₁)[k]'(by rw [size_stageSrc]; omega) = b[k] := by
    intro k hk
    have hkR : k < R * SpotFlock.L := by omega
    have hq : k / SpotFlock.L < R := Nat.div_lt_of_lt_mul (by rw [Nat.mul_comm]; exact hkR)
    have hρ : k % SpotFlock.L < stageWidth b (k / SpotFlock.L) := by
      have h1 := Nat.div_add_mod k SpotFlock.L
      have h2 := Nat.mod_lt k L_pos
      unfold stageWidth
      refine lt_min_iff.2 ⟨h2, ?_⟩
      rw [Nat.mul_comm] at h1
      generalize k / SpotFlock.L * SpotFlock.L = Q at h1 ⊢
      generalize k % SpotFlock.L = ρ at h1 ⊢
      omega
    rw [getElem_stageSrc c₁ hkR, ← getElem_of_prefixAgree (stageWidth_le b _) (hj ⟨k / SpotFlock.L, hq⟩) hρ,
      ← getElem_stageSrc cs hkR]
    exact getElem_of_eq_append hu hk
  have h1 : (SpotFlock.stageSrc c₁).extract 0 (0 + b.size) = b.extract 0 (0 + b.size) := by
    rw [ByteArray.extract_eq_extract_iff_getElem (by rw [size_stageSrc]; omega) (by omega)]
    intro k hk
    simpa using e k hk
  rw [Nat.zero_add, ByteArray.extract_zero_size] at h1
  exact ⟨_, eq_append_of_extract h1 (by rw [size_stageSrc]; exact hn)⟩

/-- **The coin `coinOf` chose starts with the run's bytes.** -/
theorem coinOf_starts {h : SpotFlock.Held} (hcov : CoinsCover h) {b : ByteArray} {c : FlockDeploy.Coin}
    (hc : coinOf b = some c) : ∃ u, SpotFlock.src c = b ++ u := by
  unfold coinOf at hc
  split at hc
  · rename_i hn
    obtain rfl := Option.some.inj hc
    obtain ⟨c', hc'⟩ := hcov.1 (b ++ ⟨Array.replicate (SpotFlock.L - b.size) 0⟩) (by
      rw [ByteArray.size_append]
      show b.size + (Array.replicate (SpotFlock.L - b.size) (0 : UInt8)).size = SpotFlock.L
      rw [Array.size_replicate]
      omega)
    have hsp := Classical.epsilon_spec (p := fun c : FlockDeploy.Coin =>
      SpotFlock.src c = b ++ ByteArray.mk (Array.replicate (SpotFlock.L - b.size) 0)) ⟨c', hc'⟩
    exact ⟨_, hsp⟩
  · cases hc

/-- **The coins `stageCoinsOf` chose start with the run's bytes.** -/
theorem stageCoinsOf_starts {h : SpotFlock.Held} (hcov : CoinsCover h) {b : ByteArray} {cs : SpotFlock.Stage h}
    (hc : stageCoinsOf h b = some cs) : ∃ u, SpotFlock.stageSrc cs = b ++ u := by
  unfold stageCoinsOf at hc
  split at hc
  · rename_i hn
    obtain rfl := Option.some.inj hc
    obtain ⟨c', hc'⟩ := hcov.2 (b ++ ⟨Array.replicate (h.stage * SpotFlock.L - b.size) 0⟩) (by
      rw [ByteArray.size_append]
      show b.size + (Array.replicate (h.stage * SpotFlock.L - b.size) (0 : UInt8)).size = h.stage * SpotFlock.L
      rw [Array.size_replicate]
      omega)
    have hsp := Classical.epsilon_spec (p := fun cs : SpotFlock.Stage h =>
      SpotFlock.stageSrc cs = b ++ ByteArray.mk (Array.replicate (h.stage * SpotFlock.L - b.size) 0)) ⟨c', hc'⟩
    exact ⟨_, hsp⟩
  · cases hc

/-! ## Runs -/

/-- **SPOT[Flock] accepts only at its verdict's round.** -/
theorem next_ne_acc (h : SpotFlock.Held) (H : Msg → Msg) (x : Msg) (as : List Msg) (cs : List FlockDeploy.Coin)
    (hne : as.length ≠ SpotFlock.off h h.ms.length + 1) : ((SpotFlock.next h x as cs).eval H).1 ≠ .acc := by
  unfold SpotFlock.next
  split
  · nofun
  split
  · split <;> nofun
  split
  · split <;> nofun
  split
  · exact SpotFlock.sessStep_ne_acc h H _ _ _ _
  · split
    · rename_i hc
      simp [hne] at hc
    · nofun

/-- **An accepted run of SPOT[Flock] has all its rounds' messages.** -/
theorem length_of_acc (h : SpotFlock.Held) {Ω : Type} (H : Msg → Msg) (P : Prover Ω) (x : Msg) (ω : Ω)
    (r : Fin ((SpotFlock.V h).k x) → FlockDeploy.Coin) (hacc : (Claims.run H P (SpotFlock.V h) x ω r).out = .acc) :
    (Claims.run H P (SpotFlock.V h) x ω r).msgs.length = SpotFlock.off h h.ms.length + 1 := by
  obtain ⟨as', hlen, hacc'⟩ := RecDeploy.stepWith_acc_length H P x ω
    (fun as => (SpotFlock.V h).next x as ((List.range as.length).map (coinSeq r))) ((SpotFlock.V h).k x) [] []
    ⟨[], [], [], .rej⟩ rfl hacc
  have e : (Claims.run H P (SpotFlock.V h) x ω r).msgs.length = as'.length := hlen.symm
  rw [e]
  by_contra hne
  exact next_ne_acc h H x as' _ hne hacc'

/-- **A round's message is the prover's next one on a history as long as the rounds before it.** -/
theorem msgAt_run {Coin Ω : Type} [Inhabited Coin] (H : Msg → Msg) (P : Prover Ω) (V : Verifier Coin)
    (x : Msg) (ω : Ω) (r : Fin (V.k x) → Coin) {n : ℕ} (hn : n < (Claims.run H P V x ω r).msgs.length) :
    ∃ hist : List (Msg × Msg), msgAt (Claims.run H P V x ω r) n = some ((P.next x hist ω).eval H).1 ∧
      hist.length = n := by
  obtain ⟨new, hnew, -, hget⟩ := RecDeploy.stepWith_get H P x ω
    (fun as => V.next x as ((List.range as.length).map (coinSeq r))) n (V.k x) [] [] ⟨[], [], [], .rej⟩
    (by rw [List.length_nil, Nat.zero_add]; exact hn)
  simp only [List.length_nil, Nat.zero_add, List.nil_append] at hget
  have hget' : (Claims.run H P V x ω r).msgs[n]? = _ := hget
  refine ⟨new.map RecDeploy.histOf, ?_, ?_⟩
  · unfold msgAt
    rw [hget']
    rfl
  · rw [List.length_map, hnew]

/-- **The sessions leave no round between them**: every round from `stage + 2` before template `m`'s session is in
some earlier template's. -/
theorem slot_of_lt (h : SpotFlock.Held) {n : ℕ} (h2 : h.stage + 2 ≤ n) :
    ∀ m, n < SpotFlock.off h m → ∃ t < m, SpotFlock.off h t ≤ n ∧ n < SpotFlock.off h t + SpotFlock.k₀ h t
  | 0, hn => absurd hn (Nat.not_lt.2 h2)
  | m + 1, hn => by
    by_cases hm : n < SpotFlock.off h m
    · obtain ⟨t, ht, h1, h3⟩ := slot_of_lt h h2 m hm
      exact ⟨t, by omega, h1, h3⟩
    · exact ⟨m, by omega, by omega, hn⟩

theorem slotOf_some {h : SpotFlock.Held} {n t : ℕ} (hs : SpotFlock.slotOf h n = some t) :
    t < h.ms.length ∧ SpotFlock.off h t < n ∧ n ≤ SpotFlock.off h t + SpotFlock.k₀ h t := by
  unfold SpotFlock.slotOf at hs
  have hm := List.mem_of_find?_eq_some hs
  have hp := List.find?_some hs
  simp only [List.mem_range, decide_eq_true_eq] at hm hp
  exact ⟨hm, hp⟩

/-! ## The form's reading, round by round -/

section Form

variable (A : Live₂) (C : CaseOf) (S : SessionReading) (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin)
  (o : Verity.Exchange)

theorem form_reads {n : ℕ} (h0 : n ≠ A.h.stage + 1) (h1 : n < SpotFlock.off A.h A.h.ms.length) :
    (formReading A C S dev).reads n :=
  And.intro h0 h1

theorem form_coin_zero : (formReading A C S dev).coin o 0 = (inputOn C o).bind fun i => coinOf i.source := by
  simp [formReading]

theorem form_coin_stage {j : ℕ} (hj : j < A.h.stage) :
    (formReading A C S dev).coin o (j + 1) =
      (inputOn C o).bind fun i => i.interior.bind fun it =>
        (stageCoinsOf A.h it.source).map fun cs => cs ⟨j, hj⟩ := by
  have hj' : j ≤ A.h.more := Nat.le_of_lt_succ hj
  simp [formReading, hj']

theorem form_coin_slot {n t : ℕ} (h0 : n ≠ 0) (h1 : ¬n - 1 < A.h.stage) (h2 : n ≠ A.h.stage + 1)
    (hs : SpotFlock.slotOf A.h (n + 1) = some t) :
    (formReading A C S dev).coin o n = S.coin o t (n - SpotFlock.off A.h t) := by
  simp only [formReading]
  rw [if_neg h0, dif_neg h1, if_neg h2, hs]

theorem form_agree_zero (c c' : FlockDeploy.Coin) :
    (formReading A C S dev).agree o 0 c c' ↔ ∃ i, inputOn C o = some i ∧ prefixAgree i.source.size c c' := by
  simp [formReading]

theorem form_agree_stage {j : ℕ} (hj : j < A.h.stage) (c c' : FlockDeploy.Coin) :
    (formReading A C S dev).agree o (j + 1) c c' ↔
      ∃ i it, inputOn C o = some i ∧ i.interior = some it ∧ prefixAgree (stageWidth it.source j) c c' := by
  simp [formReading, Nat.succ_le_of_lt hj]

theorem form_agree_slot {n t : ℕ} (h0 : n ≠ 0) (h1 : ¬n ≤ A.h.stage) (hs : SpotFlock.slotOf A.h (n + 1) = some t)
    (c c' : FlockDeploy.Coin) :
    (formReading A C S dev).agree o n c c' = S.agree o t (n - SpotFlock.off A.h t) c c' := by
  simp only [formReading]
  rw [if_neg h0, if_neg h1, hs]

theorem form_msg_zero : (formReading A C S dev).msg o 0 = some [] := by
  simp [formReading]

theorem form_msg_one :
    (formReading A C S dev).msg o 1 = (inputOn C o).bind fun i => i.interior.bind fun it => recordMsg it.record := by
  simp [formReading]

theorem form_msg_mid {n : ℕ} (h2 : 2 ≤ n) (h3 : n ≤ A.h.stage) : (formReading A C S dev).msg o n = some [] := by
  simp only [formReading]
  rw [if_neg (show ¬n = 0 by omega), if_neg (show ¬n = 1 by omega), if_pos h3]

theorem form_msg_stmt {n : ℕ} (hn : n = A.h.stage + 1) : (formReading A C S dev).msg o n = S.stmts o := by
  have hst : A.h.stage = A.h.more + 1 := rfl
  simp only [formReading]
  rw [if_neg (show ¬n = 0 by omega), if_neg (show ¬n = 1 by omega), if_neg (show ¬n ≤ A.h.stage by omega),
    if_pos hn]

theorem form_msg_slot {n t : ℕ} (h0 : n ≠ 0) (h1 : n ≠ 1) (h3 : ¬n ≤ A.h.stage) (h4 : n ≠ A.h.stage + 1)
    (hs : SpotFlock.slotOf A.h (n + 1) = some t) :
    (formReading A C S dev).msg o n = S.msg o t (n - SpotFlock.off A.h t) := by
  simp only [formReading]
  rw [if_neg h0, if_neg h1, if_neg h3, if_neg h4, hs]

theorem form_msg_none {n : ℕ} (h0 : n ≠ 0) (h1 : n ≠ 1) (h3 : ¬n ≤ A.h.stage) (h4 : n ≠ A.h.stage + 1)
    (hs : SpotFlock.slotOf A.h (n + 1) = none) : (formReading A C S dev).msg o n = some [] := by
  simp only [formReading]
  rw [if_neg h0, if_neg h1, if_neg h3, if_neg h4, hs]

end Form

/-! ## The accept -/

/-- **`proverSeq` is the accepted prover sending the check's messages** (`ProverAt` at the form): at the sessions'
provers `P'`, under `AcceptsRunOf`'s premises, at every play the form's check accepts and every coin vector agreeing
with what it read, `SpotFlock.V` accepts `proverSeq` at the run of record's records, and its messages are those the
check read. -/
theorem proverAt (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) (P' : Verity.Exchange → ℕ → Claims.Prover Unit)
    (hcov : CoinsCover A.h) (hfit : SourcesFit A C) (hplay : PlayAccepts A C S P') (hsess : SessionMsgs A C S P')
    (hreads : ReadsOnly A.h) : ProverAt A x (formReading A C S dev) (FormAccepts C) := by
  intro o hA hs r hc
  have hst : A.h.stage = A.h.more + 1 := rfl
  have hT3 := SpotFlock.stage_le_off A.h A.h.ms.length
  have hA' := hA
  obtain ⟨k, hk, hacc⟩ := hA'
  obtain ⟨cm, i, -, hi', -, it, m₁, hit, hm₁⟩ := input_of_accepts hacc
  have hi : inputOn C o = some i := by
    unfold inputOn
    rw [hk]
    exact hi'
  have hx : recordMsg i.record = some x := by
    have hs' : (inputOn C o).bind (fun i => recordMsg i.record) = some x := hs
    rw [hi] at hs'
    exact hs'
  obtain ⟨m₂, hm₂, hk₀, hacc₂⟩ := hplay o i it x m₁ hA hi hit hx hm₁
  obtain ⟨hsz₀, hsz₁⟩ := hfit o i it hA hi hit
  have h0k : 0 < A.L.V.k x := by
    show 0 < SpotFlock.off A.h A.h.ms.length + 1
    omega
  let c₀ : FlockDeploy.Coin := r ⟨0, h0k⟩
  let c₁ : SpotFlock.Stage A.h := fun j => r ⟨j.val + 1, by
    show j.val + 1 < SpotFlock.off A.h A.h.ms.length + 1
    have := j.isLt
    omega⟩
  let r' : ∀ t, Fin ((FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t)) → FlockDeploy.Coin := fun t j =>
    if hn : SpotFlock.off A.h t + j.val < SpotFlock.off A.h A.h.ms.length + 1 then r ⟨_, hn⟩ else default
  have hread : SeqCoinsRead A S o i it m₂ c₀ c₁ r' := by
    refine ⟨?_, ?_, ?_⟩
    · obtain ⟨cc, hcc⟩ : ∃ cc, coinOf i.source = some cc := by
        unfold coinOf
        rw [if_pos hsz₀]
        exact ⟨_, rfl⟩
      have hcoin : (formReading A C S dev).coin o 0 = some cc := by
        rw [form_coin_zero, hi]
        exact hcc
      have hag := hc ⟨0, h0k⟩ (form_reads A C S dev (n := 0) (by omega) (by omega)) cc hcoin
      obtain ⟨i', hi'', hpa⟩ := (form_agree_zero A C S dev o cc _).1 hag
      rw [hi] at hi''
      obtain rfl := Option.some.inj hi''
      exact starts_of_prefixAgree (coinOf_starts hcov hcc) hpa
    · obtain ⟨cs, hcs⟩ : ∃ cs, stageCoinsOf A.h it.source = some cs := by
        unfold stageCoinsOf
        rw [if_pos hsz₁]
        exact ⟨_, rfl⟩
      refine stage_starts (stageCoinsOf_starts hcov hcs) fun j => ?_
      have hj := j.isLt
      have hn1 : j.val + 1 < A.L.V.k x := by
        show j.val + 1 < SpotFlock.off A.h A.h.ms.length + 1
        omega
      have hcoin : (formReading A C S dev).coin o (j.val + 1) = some (cs j) := by
        rw [form_coin_stage A C S dev o j.isLt, hi]
        simp [hit, hcs]
      have hag := hc ⟨j.val + 1, hn1⟩ (form_reads A C S dev (n := j.val + 1) (by omega) (by omega)) (cs j) hcoin
      obtain ⟨i', it', hi'', hit', hpa⟩ := (form_agree_stage A C S dev o j.isLt _ _).1 hag
      rw [hi] at hi''
      obtain rfl := Option.some.inj hi''
      rw [hit] at hit'
      obtain rfl := Option.some.inj hit'
      exact hpa
    · intro t ht j hj c hcj
      have hjk : j < SpotFlock.k₀ A.h t := by
        rw [← hk₀ t ht]
        exact hj
      have hle : SpotFlock.off A.h t + SpotFlock.k₀ A.h t ≤ SpotFlock.off A.h A.h.ms.length :=
        Claims.Sampling.sessOff_add_le (A.h.stage + 2) (SpotFlock.k₀ A.h) ht
      have h3 := SpotFlock.stage_le_off A.h t
      have hn : SpotFlock.off A.h t + j < SpotFlock.off A.h A.h.ms.length + 1 := by omega
      have hs := SpotFlock.slotOf_eq A.h ht (n := SpotFlock.off A.h t + j + 1) (by omega) (by omega)
      have hcoin : (formReading A C S dev).coin o (SpotFlock.off A.h t + j) = some c := by
        rw [form_coin_slot A C S dev o (by omega) (by omega) (by omega) hs, Nat.add_sub_cancel_left]
        exact hcj
      have hag := hc ⟨SpotFlock.off A.h t + j, hn⟩
        (form_reads A C S dev (n := SpotFlock.off A.h t + j) (by omega) (by omega)) c hcoin
      have hag' : (formReading A C S dev).agree o (SpotFlock.off A.h t + j) c (r ⟨SpotFlock.off A.h t + j, hn⟩) :=
        hag
      rw [form_agree_slot A C S dev o (by omega) (by omega) hs, Nat.add_sub_cancel_left] at hag'
      show S.agree o t j c
        (if hn' : SpotFlock.off A.h t + j < SpotFlock.off A.h A.h.ms.length + 1 then r ⟨_, hn'⟩ else default)
      rw [dif_pos hn]
      exact hag'
  have hagree : ∀ n : Fin ((SpotFlock.V A.h).k x), readsAt A.h n.val →
      r n = SpotFlock.seqCoins A.h (x := x) c₀ c₁ r' n := by
    intro n hn
    obtain ⟨hn1, hn2⟩ := hn
    simp only [SpotFlock.seqCoins]
    by_cases h0 : n.val = 0
    · rw [if_pos h0]
      exact congrArg r (Fin.ext h0)
    rw [if_neg h0]
    by_cases hs1 : n.val - 1 < A.h.stage
    · rw [dif_pos hs1]
      exact congrArg r (Fin.ext (show n.val = n.val - 1 + 1 by omega))
    rw [dif_neg hs1]
    obtain ⟨t, ht, h1, h2⟩ := slot_of_lt A.h (n := n.val) (by omega) A.h.ms.length hn2
    have hs := SpotFlock.slotOf_eq A.h ht (n := n.val + 1) (by omega) (by omega)
    have hj : n.val - SpotFlock.off A.h t < (FlockDeploy.V true).k (SpotFlock.stmtAt A.h m₂ t) := by
      rw [hk₀ t ht]
      omega
    rw [hs]
    dsimp only
    rw [dif_pos hj]
    show r n = if hn' : SpotFlock.off A.h t + (n.val - SpotFlock.off A.h t) < SpotFlock.off A.h A.h.ms.length + 1
      then r ⟨_, hn'⟩ else default
    have hc' : SpotFlock.off A.h t + (n.val - SpotFlock.off A.h t) < SpotFlock.off A.h A.h.ms.length + 1 := by
      omega
    rw [dif_pos hc']
    exact congrArg r (Fin.ext (show n.val = SpotFlock.off A.h t + (n.val - SpotFlock.off A.h t) by omega))
  have hrun := hreads FlockDeploy.sha512 _ (SpotFlock.proverSeq A.h [] m₁ m₂ (P' o)) x (fun _ => ()) r _ hagree
  have hacc' : (seqRun A P' o x m₁ m₂ c₀ c₁ r').out = .acc := hacc₂ c₀ c₁ r' hread
  have hlen := length_of_acc A.h FlockDeploy.sha512 (SpotFlock.proverSeq A.h [] m₁ m₂ (P' o)) x (fun _ => ())
    (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r') hacc'
  have hT : ∀ n, msgAt (seqRun A P' o x m₁ m₂ c₀ c₁ r') n =
      msgAt (Claims.run FlockDeploy.sha512 (SpotFlock.proverSeq A.h [] m₁ m₂ (P' o)) (SpotFlock.V A.h) x
        (fun _ => ()) (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r')) n := fun _ => rfl
  refine ⟨ℕ → Unit, SpotFlock.proverSeq A.h [] m₁ m₂ (P' o), fun _ => (), ?_, ?_⟩
  · exact (congrArg Claims.Transcript.out hrun).trans hacc'
  intro n hn
  have hn' : n < SpotFlock.off A.h A.h.ms.length + 1 := hn
  suffices hsuf : (formReading A C S dev).msg o n = msgAt (seqRun A P' o x m₁ m₂ c₀ c₁ r') n from
    hsuf.trans (congrArg (fun T => msgAt T n) hrun).symm
  rw [hT]
  obtain ⟨hist, hm, hhl⟩ := msgAt_run FlockDeploy.sha512 (SpotFlock.proverSeq A.h [] m₁ m₂ (P' o)) (SpotFlock.V A.h)
    x (fun _ => ()) (SpotFlock.seqCoins A.h (x := x) c₀ c₁ r') (n := n) (by rw [hlen]; exact hn')
  by_cases hpre : n < A.h.stage + 2
  · rw [hm, SpotFlock.proverSeq_pre A.h [] m₁ m₂ (P' o) x hist _ (by omega), hhl]
    show _ = some (SpotFlock.preMsg A.h [] m₁ m₂ n)
    rcases Nat.lt_or_ge n 2 with h2 | h2
    · rcases Nat.lt_or_ge n 1 with h1 | h1
      · obtain rfl : n = 0 := by omega
        rw [form_msg_zero]
        simp [SpotFlock.preMsg]
      · obtain rfl : n = 1 := by omega
        rw [form_msg_one, hi, SpotFlock.preMsg_one]
        simp [hit, hm₁]
    · rcases Nat.lt_or_ge n (A.h.stage + 1) with h3 | h3
      · rw [form_msg_mid A C S dev o h2 (by omega), SpotFlock.preMsg_mid A.h [] m₁ m₂ h2 (by omega)]
      · obtain rfl : n = A.h.stage + 1 := by omega
        rw [form_msg_stmt A C S dev o rfl, SpotFlock.preMsg_last, hm₂]
  · rcases hs : SpotFlock.slotOf A.h (n + 1) with _ | t
    · rw [form_msg_none A C S dev o (by omega) (by omega) (by omega) (by omega) hs, hm,
        SpotFlock.proverSeq_next A.h [] m₁ m₂ (P' o) x hist _ (by omega), hhl, hs] <;> rfl
    · obtain ⟨ht, h1, h2⟩ := slotOf_some hs
      rw [form_msg_slot A C S dev o (by omega) (by omega) (by omega) (by omega) hs]
      have hsm := hsess o i it x m₁ m₂ hA hi hit hx hm₁ hm₂ c₀ c₁ r' hread t ht (n - SpotFlock.off A.h t) (by omega)
      rw [Nat.add_sub_cancel' (show SpotFlock.off A.h t ≤ n by omega), hT] at hsm
      exact hsm.symm

/-- **The form's accept is the run's accept** (`AcceptsRunOf`): at the sessions' provers `P'`, where every coins' byte
string is some coins' (`CoinsCover`), an accepted run read at most the coins it has (`SourcesFit`), its run of record is
a play SPOT[Flock] accepts (`PlayAccepts`) whose sessions' messages are those the check read (`SessionMsgs`), and
SPOT[Flock] reads a coin only where `readsAt` says (`ReadsOnly`), every run whose coins agree with what the form's
check read and whose messages are what it read accepts. -/
theorem acceptsRun (A : Live₂) (x : Msg) (C : CaseOf) (S : SessionReading)
    (dev : List Verity.Release → ℕ → Option FlockDeploy.Coin) : AcceptsRunOf A x C S dev := by
  intro P' hcov hfit hplay hsess hreads
  exact acceptsRun_of_proverAt (proverAt A x C S dev P' hcov hfit hplay hsess hreads)

end Verity.SampledTwo
