import Proofs.Flock.Claims.Complete.Ready

/-!
# The masked prover as `flock-circuit --zk` runs it

`honestP` runs the masked session on each table's share of the witness as given. `flock-circuit --zk` differs from it in
two places, and agrees with it in its other refusals; `honestPR` is `honestP` with the two differences made.

* **The mask slot's words are the tape's** (`verity/ml/flock/live/src/bin/flock-circuit.rs:2738`–`2747`): table `i`'s
  block `b`'s mask words are `h.rng.f128s(STREAM_MASK, b·mask_words, mask_words)` (`STREAM_MASK`,
  `verity/ml/flock/live/src/zk_hooks.rs:35`), so word `k` of the slot in `maskWordsOf` order (`mask_words`,
  `flock-circuit.rs:993`) is `STREAM_MASK`'s index `k`, the tape's `mask i k`. `maskW` puts them in the witness: table
  `c`'s bit `a` is hidden value `a·J + c` (`Readable.tableWit`), and where `a` is in the slot's word `k` (`maskIdx`) the
  bit is `mask c k`'s bit `a mod 128`. `maskWOf x` does it at `x`'s session.
* **An outer coin `1` stops the prover** (`Masking::refuse_outer`, `verity/ml/flock/live/src/zk_veil.rs:329`, called on
  every coin as it is sampled, `zk_veil.rs:393`, `403`): the zerocheck's outer coins (`outerSlotsZk`) read from the
  verifier's replies so far (`outerHit`); at one equal to `1` the prover sends nothing more of the session (`[]`).
* **A degenerate mask rank stops the prover** (`mask_rank_ok`, `zk_veil.rs:1087`, at the opening's label,
  `flock-circuit.rs:1026`): already `honestP`'s, in its session (`MaskRank.rankCheckP`, in `sessionPZk`). So are the
  other refused coins: `ρ₀, ρ₁` dependent over F128 (`ligeritoPZk`) and an inner `β = 0` (`innerP`), each a throw
  that ends `sessionPZk` with the empty message.

`honestPR w` is `honestP (maskWOf x w ω.mask)` at every history no outer coin of which is `1` (`honestPR_next`).
-/

namespace FlockComplete.ZkTables

open Flock FlockDeploy

/-- **The mask slot's word that bit `a` is in**: its index in `maskWordsOf st`, as `Composed.zWith` reads it. -/
def maskIdx (st : Flock.Stmt) (a : ℕ) : Option (Fin (FlockSoundness.Discharge.Composed.maskWordsOf st).length) :=
  (List.finRange _).find? fun x => (FlockSoundness.Discharge.Composed.maskWordsOf st).getD x.val 0 = a / 128

/-- **Hidden value `p` with the mask words `μ`**: table `c = p mod J`'s bit `a = p / J` (`Readable.tableWit`), which is
`μ c k`'s bit `a mod 128` where `a` is in the slot's word `k`, and `w`'s elsewhere. -/
def maskBit (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) (p : ℕ) : Bool :=
  match sts[p % sts.size]? with
  | some st => match maskIdx st (p / sts.size) with
    | some k => (μ (p % sts.size) k.val).bit (p / sts.size % 128)
    | none => w.hid.getD p false
  | none => w.hid.getD p false

/-- **The witness with every table's mask slot set to `μ`**: `maskBit` at each hidden value, long enough for every
table's `2^m` bits. -/
def maskW (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) : Claims.Flock.Wit :=
  ⟨(List.range (max w.hid.length (sts.size * (sts.toList.map fun st => 2 ^ st.m).sum))).map (maskBit sts w μ)⟩

/-- **The witness with the mask slots set to `μ`** at `x`'s session, and `w` where `x` does not set up. -/
def maskWOf (x : Claims.Msg) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) : Claims.Flock.Wit :=
  match setupOf true x with
  | some (_, _, sts) => maskW sts w μ
  | none => w

/-- **Some zerocheck outer coin among the replies `bs` is `1`** (`refuse_outer`'s test). -/
def outerHit (x : Claims.Msg) (bs : List Claims.Msg) : Bool :=
  (outerSlotsZk x).any fun p => decide (p.1 < bs.length) && decide (coinOf (bs.getD p.1 []) p.2 = 1)

/-- **Flock's masked prover as `flock-circuit --zk` runs it**: `honestP` with the tape's mask words in every table's
mask slot (`maskWOf`), stopped (`[]`) once an outer coin is `1` (`outerHit`). -/
def honestPR (w : Claims.Flock.Wit) : Claims.Prover TapeZk :=
  ⟨fun x hist ω => if outerHit x (hist.map Prod.snd) then .ret [] else (honestP (maskWOf x w ω.mask)).next x hist ω⟩

theorem honestPR_next {w : Claims.Flock.Wit} {x : Claims.Msg} {hist : List (Claims.Msg × Claims.Msg)} {ω : TapeZk}
    (h : outerHit x (hist.map Prod.snd) = false) :
    (honestPR w).next x hist ω = (honestP (maskWOf x w ω.mask)).next x hist ω := by
  simp only [honestPR, h, Bool.false_eq_true, ↓reduceIte]

theorem honestPR_stop {w : Claims.Flock.Wit} {x : Claims.Msg} {hist : List (Claims.Msg × Claims.Msg)} {ω : TapeZk}
    (h : outerHit x (hist.map Prod.snd) = true) : (honestPR w).next x hist ω = .ret [] := by
  simp only [honestPR, h, ↓reduceIte]

/-! ## The mask witness's bits -/

theorem maskW_getD (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) {p : ℕ}
    (hp : p < sts.size * (sts.toList.map fun st => 2 ^ st.m).sum) :
    (maskW sts w μ).hid.getD p false = maskBit sts w μ p := by
  simp only [maskW, List.getD_eq_getElem?_getD, List.getElem?_map,
    List.getElem?_range (lt_of_lt_of_le hp (le_max_right _ _)), Option.map_some, Option.getD_some]

/-- **Off the mask slots the mask witness is `w`**: at a hidden value whose bit is in no table's mask word. -/
theorem maskW_getD_off (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) {p : ℕ}
    (hp : maskBit sts w μ p = w.hid.getD p false) : (maskW sts w μ).hid.getD p false = w.hid.getD p false := by
  by_cases hl : p < max w.hid.length (sts.size * (sts.toList.map fun st => 2 ^ st.m).sum)
  · simp only [maskW, List.getD_eq_getElem?_getD, List.getElem?_map, List.getElem?_range hl, Option.map_some,
      Option.getD_some, hp]
  · rw [List.getD_eq_default _ _ (by simp only [maskW, List.length_map, List.length_range]; omega),
      List.getD_eq_default _ _ (by omega)]

theorem maskW_lt (sts : Array Flock.Stmt) {c : ℕ} (hc : c < sts.size) {a : ℕ} (ha : a < 2 ^ sts[c].m) :
    a * sts.size + c < sts.size * (sts.toList.map fun st => 2 ^ st.m).sum := by
  have hle : 2 ^ sts[c].m ≤ (sts.toList.map fun st => 2 ^ st.m).sum :=
    List.le_sum_of_mem (List.mem_map.2 ⟨sts[c], Array.getElem_mem_toList hc, rfl⟩)
  calc a * sts.size + c < (a + 1) * sts.size := by rw [Nat.add_mul, Nat.one_mul]; omega
    _ ≤ 2 ^ sts[c].m * sts.size := Nat.mul_le_mul_right _ ha
    _ ≤ sts.size * (sts.toList.map fun st => 2 ^ st.m).sum := by rw [Nat.mul_comm]; exact Nat.mul_le_mul_left _ hle

/-- **Table `c`'s bit `a` of the mask witness**: `μ c k`'s bit `a mod 128` where `a` is in the slot's word `k`, and
`w`'s elsewhere (`maskW_table_off`), at a bit of the table (`a < 2^m`). -/
theorem maskW_table (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) {c : ℕ} (hc : c < sts.size)
    {a : ℕ} (ha : a < 2 ^ sts[c].m) {k : Fin (FlockSoundness.Discharge.Composed.maskWordsOf sts[c]).length}
    (hk : maskIdx sts[c] a = some k) :
    (maskW sts w μ).hid.getD (a * sts.size + c) false = (μ c k.val).bit (a % 128) := by
  have hmod : (a * sts.size + c) % sts.size = c := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hc]
  have hdiv : (a * sts.size + c) / sts.size = a := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hc, Nat.add_zero]
  rw [maskW_getD sts w μ (maskW_lt sts hc ha)]
  unfold maskBit
  simp only [hmod, hdiv, Array.getElem?_eq_getElem hc, hk]

theorem maskW_table_off (sts : Array Flock.Stmt) (w : Claims.Flock.Wit) (μ : ℕ → ℕ → F128) {c : ℕ}
    (hc : c < sts.size) {a : ℕ} (hk : maskIdx sts[c] a = none) :
    (maskW sts w μ).hid.getD (a * sts.size + c) false = w.hid.getD (a * sts.size + c) false := by
  have hmod : (a * sts.size + c) % sts.size = c := by
    rw [Nat.mul_comm, Nat.mul_add_mod, Nat.mod_eq_of_lt hc]
  have hdiv : (a * sts.size + c) / sts.size = a := by
    rw [Nat.mul_comm, Nat.mul_add_div (by omega), Nat.div_eq_of_lt hc, Nat.add_zero]
  refine maskW_getD_off sts w μ ?_
  unfold maskBit
  simp only [hmod, hdiv, Array.getElem?_eq_getElem hc, hk]

end FlockComplete.ZkTables
