import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork

/-!
# Flock's KS from one wait's bound

`Flock.KS` at `E^rep` and a decoder that meets `ListComplete` follows from one wait's bound on `k` accepted runs
(`Flock.OneWait`), with the numbers' side conditions `K > 0`, `k ≥ 8` and `ε_table ≥ 0` (`Flock.KS_of_oneWait`):

* **Extraction** (`Flock.Extracts`, `Flock.assembles_of_extracts`): when the accepted runs' opened columns, each
  position once, number at least `nMin M` and agree with a trace `M` on a fraction `agree` of them, and `M` reads as a
  witness, the decoder lists `M` and `assemble` returns a witness `check` accepts.
* **One wait** (`Flock.OneWait`, the hypothesis): at every statement and prover there are escapes, fitting the weight
  `C(x)`, such that at every tape with acceptance `ε > 0`, `ε · Pr[k accepted runs don't extract] ≤
  ε_table + escapes(ω) + (N₀/(e·k))·ε`. This is the old soundness's per-tape bound (`joint_le_accB`) at the deployed
  verifier, on the opened columns, with each finder's escape tied to the verifier's calls.
* **Assembly**: `ERep_trunc` at `rate = N₀/(e·k) ≤ 1/4` (`Flock.rate_le`, from `k = ⌈4N₀/e⌉`) gives
  `4·(ε_table + 𝔼[escapes]) + 2^{−m′} + 2k/K + e^{−k/8}`, and `Escapes.le` puts `4·𝔼[escapes]` under `√(γ·Pr[B])`.

`assemble` decodes the waits' opened columns as they come, `vs.flatMap cols`, which repeats a position whenever two
reruns open it; `ListComplete` speaks only of lists at distinct positions. So the theorem is at decoders that read
each position once (`Flock.ReadsOnce`); every decoder `d` that meets `ListComplete` gives one, `d ∘ dedupPos`, that
meets it too (`Flock.KS_of_oneWait_dedup`).
-/

namespace Claims

namespace Flock

/-! ## Each position once -/

/-- The opened columns, each position once (its last occurrence). -/
def dedupPos {Col : Type} (os : List (ℕ × Col)) : List (ℕ × Col) := os.pwFilter fun a b => a.1 ≠ b.1

theorem dedupPos_nodup {Col : Type} (os : List (ℕ × Col)) : ((dedupPos os).map Prod.fst).Nodup := by
  unfold dedupPos List.Nodup
  rw [List.pairwise_map]
  exact List.pairwise_pwFilter _

theorem dedupPos_of_nodup {Col : Type} (os : List (ℕ × Col)) (h : (os.map Prod.fst).Nodup) : dedupPos os = os := by
  unfold dedupPos
  rw [List.pwFilter_eq_self]
  unfold List.Nodup at h
  rwa [List.pairwise_map] at h

/-- A decoder that reads each opened position once. -/
def ReadsOnce {Col T : Type} (decode : List (ℕ × Col) → List T) : Prop := ∀ os, decode (dedupPos os) = decode os

theorem readsOnce_dedup {Col T : Type} (decode : List (ℕ × Col) → List T) : ReadsOnce (decode ∘ dedupPos) :=
  fun os => by simp only [Function.comp_apply, dedupPos_of_nodup _ (dedupPos_nodup os)]

theorem listComplete_dedup {Col T : Type} [DecidableEq Col] {enc : T → ℕ → Col} {agree : ℝ} {nMin : T → ℕ}
    {decode : List (ℕ × Col) → List T} (h : ListComplete enc agree nMin decode) :
    ListComplete enc agree nMin (decode ∘ dedupPos) := by
  intro os M hnd hn hag
  simp only [Function.comp_apply, dedupPos_of_nodup os hnd]
  exact h os M hnd hn hag

/-! ## Extraction -/

theorem check_sound (H : Msg → Msg) (parse : Msg → Option Stmt) : ∀ x w, check H parse x w = true → R H parse x w := by
  intro x w h
  unfold check at h
  split at h
  · rename_i s hs
    exact ⟨s, hs, of_decide_eq_true h⟩
  · simp at h

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col]

/-- **Flock's extraction event** on accepted reruns `vs`: their opened columns, each position once, are at least
`nMin M` and agree with a trace `M`'s encoding on a fraction `agree` of them, and `M` reads as a witness of the parsed
statement. -/
def Extracts (H : Msg → Msg) (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) : Prop :=
  ∃ (M : T) (s : Stmt), D.parse x = some s ∧ RelS H s (D.read M) ∧
    D.nMin M ≤ (dedupPos (vs.flatMap D.cols)).length ∧
    D.agree * (dedupPos (vs.flatMap D.cols)).length ≤
      ((dedupPos (vs.flatMap D.cols)).filter fun p => decide (D.enc M p.1 = p.2)).length

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Extraction succeeds** on the event: the decoder lists the trace, and `assemble` returns a witness that `check`
accepts. -/
theorem assembles_of_extracts (H : Msg → Msg) (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hlc : ListComplete D.enc D.agree D.nMin decode) (hro : ReadsOnce decode) (x : Msg) (vs : List Seen)
    (h : Extracts H D x vs) : Assembles D.cols (assemble H D.parse decode D.read x) (check H D.parse x) vs := by
  obtain ⟨M, s, hs, hrel, hn, hag⟩ := h
  have hM : M ∈ decode (vs.flatMap D.cols) := by
    rw [← hro]
    exact hlc _ M (dedupPos_nodup _) hn hag
  have hsome : (((decode (vs.flatMap D.cols)).map D.read).find? fun w => decide (RelS H s w)).isSome :=
    List.find?_isSome.2 ⟨D.read M, List.mem_map_of_mem hM, by simpa using hrel⟩
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hsome
  refine ⟨w, ?_, ?_⟩
  · unfold assemble
    rw [hs]
    exact hw
  · unfold check
    rw [hs]
    have hc := List.find?_some hw
    exact hc

/-! ## One wait's bound, and KS -/

/-- **Flock's one-wait bound**, what the soundness proof gives at the deployed verifier: at every statement and prover,
escapes fitting the weight `C(x)` such that at every tape with acceptance `ε > 0`,
`ε · Q/ε^k ≤ ε_table + escapes(ω) + (N₀/(e·k))·ε`, with `Q` the probability that `k` independent runs all accept and
their opened columns don't extract. -/
def OneWait (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.Fits (D.weight x) ∧
    ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts H D x) / accPr H D.V P x ω ^ D.k x) ≤
        D.εtable x + e.term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] in
/-- `E^rep`'s relative rate `N₀/(e·k)` is at most `1/4`, as `k = ⌈4·N₀/e⌉`. -/
theorem rate_le (D : Deploy Coin Col T) (x : Msg) (hk : 0 < D.k x) : D.N₀ x / (Real.exp 1 * D.k x) ≤ 1 / 4 := by
  have hk' : (0 : ℝ) < D.k x := by exact_mod_cast hk
  have he := Real.exp_pos 1
  rw [div_le_iff₀ (mul_pos he hk')]
  have hc : 4 * D.N₀ x / Real.exp 1 ≤ D.k x := Nat.le_ceil _
  rw [div_le_iff₀ he] at hc
  linarith

/-- **Flock: KS from one wait's bound**, at every decoder that meets `ListComplete` and reads each position once,
when `K > 0`, `k ≥ 8` and `ε_table ≥ 0`. -/
theorem KS_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x) (hk : ∀ x, 8 ≤ D.k x)
    (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWait H D) (decode : List (ℕ × Col) → List T)
    (hlc : ListComplete D.enc D.agree D.nMin decode) (hro : ReadsOnce decode) :
    Claims.KS H D.V (R H D.parse) (E H D decode) (κ D) (γ D) := by
  intro x Ω _ _ P
  obtain ⟨e, hfit, hw⟩ := h x Ω P
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x)
        (Assembles D.cols (assemble H D.parse decode D.read x) (check H D.parse x)) / accPr H D.V P x ω ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts H D decode hlc hro x _ hex)⟩
  have hb := ERep_trunc H D.V (R H D.parse) D.waits D.k D.cap D.cols (assemble H D.parse decode D.read)
    (check H D.parse) (check_sound H D.parse) x P (hK x) (hk x) (fun ω => D.εtable x + e.term ω)
    (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by have := hk x; omega)) hwait
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  show PrBad H D.V (R H D.parse)
      (ERep D.V D.waits D.k D.cap D.cols (assemble H D.parse decode D.read) (check H D.parse)) x P ≤
    κ D x + Real.sqrt (γ D x * PrB H D.V P x)
  unfold κ
  linarith

/-- **Flock: KS from one wait's bound, at every decoder that meets `ListComplete`**, reading each opened position once:
`E^rep` at `decode ∘ dedupPos`. -/
theorem KS_of_oneWait_dedup (H : Msg → Msg) (D : Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, 8 ≤ D.k x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWait H D) (decode : List (ℕ × Col) → List T)
    (hlc : ListComplete D.enc D.agree D.nMin decode) :
    Claims.KS H D.V (R H D.parse) (E H D (decode ∘ dedupPos)) (κ D) (γ D) :=
  KS_of_oneWait H D hK hk hεt h _ (listComplete_dedup hlc) (readsOnce_dedup decode)

end Flock

end Claims
