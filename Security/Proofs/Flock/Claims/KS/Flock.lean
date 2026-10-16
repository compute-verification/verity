import Proofs.Flock.Claims.KS.Wait
import Proofs.Flock.Claims.KS.Fork
import Proofs.Flock.Claims.Bits
import Proofs.Flock.Claims.DeployCode

/-!
# Flock's KS from one wait's bound

`Flock.KS` at `E^rep` and every decoder that meets `ListWithin` follows from one wait's bound on `k` accepted runs
(`Flock.OneWait`), with the numbers' side conditions `K > 0`, `ε_table ≥ 0`, and `k ≥ 8` and the relation splitting by
table (`Flock.Splits`) wherever the verifier accepts some run (`Flock.NoAcc`) (`Flock.KS_of_oneWait`):

* **Extraction** (`Flock.Extracts`, `Flock.assembles_of_extracts`): when at every table `j` of the statement the
  accepted runs' table-`j` columns, filled into the word of a trace `M`'s codeword length (`fill`), agree with `M`'s
  encoding on at least `(1 − θ M)·len M` positions, and `M` read alone is a witness of the table's own statement
  (`Flock.ExtractsAt`), the decoder lists each table's `M`, `assemble` returns a witness, and where the relation splits
  by table `check` accepts it.
* **One wait** (`Flock.OneWait`, the hypothesis): at every statement and prover there are escapes, fitting the weight
  `C(x)`, such that at every tape with acceptance `ε > 0`, `ε · Pr[k accepted runs don't extract] ≤
  ε_table + escapes(ω) + (N₀/(e·k))·ε`. This is the old soundness's per-tape bound (`joint_le_accB`) at the deployed
  verifier, on the opened columns, with each finder's escape tied to the verifier's calls.
* **Assembly**: `ERep_trunc` at `rate = N₀/(e·k) ≤ 1/4` (`Flock.rate_le`, from `k = ⌈4N₀/e⌉`) gives
  `4·(ε_table + 𝔼[escapes]) + 2^{−m′} + 2k/K + e^{−k/8}`, and `Escapes.le` puts `4·𝔼[escapes]` under `√(γ·Pr[B])`.
-/

namespace Claims

namespace Flock

/-! ## Extraction -/

theorem check_sound (parse : Msg → Option Stmt) : ∀ x w, check parse x w = true → R parse x w := by
  intro x w h
  unfold check at h
  split at h
  · rename_i s hs
    exact ⟨s, hs, of_decide_eq_true h⟩
  · simp at h

variable {Coin Col T : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col]

/-- **Flock's extraction event at table `j`** on accepted reruns `vs`: a trace `M` whose encoding agrees with the
zero-filled word of their table-`j` columns, at `M`'s codeword length, on at least `(1 − θ M)·len M` positions, and
that, read alone, is a witness of the table's own statement. -/
def ExtractsAt (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) (j : ℕ) : Prop :=
  ∃ (M : T) (t : Stmt), D.table x j = some t ∧ RelS t (D.read [M]) ∧
    (1 - D.θ M) * D.len M ≤ ((Finset.univ : Finset (Fin (D.len M))).filter fun i =>
      D.enc M i.val = fill (D.len M) (vs.flatMap fun v => D.cols v j) i).card

/-- **Flock's extraction event** on accepted reruns `vs`: the statement parses and every one of its tables
extracts. -/
def Extracts (D : Deploy Coin Col T) (x : Msg) (vs : List Seen) : Prop :=
  ∃ s, D.parse x = some s ∧ ∀ j < s.tables, ExtractsAt D x vs j

/-- **The relation splits by table at `x`**: traces, one per table of the parsed statement, each of which read alone is
a witness of its table's own statement, read together as a witness of the statement. -/
def Splits (D : Deploy Coin Col T) (x : Msg) : Prop :=
  ∀ s, D.parse x = some s → ∀ Ms : List T, Ms.length = s.tables →
    (∀ j (hj : j < Ms.length), ∃ t, D.table x j = some t ∧ RelS t (D.read [Ms[j]])) → RelS s (D.read Ms)

theorem flatMap_singleton_flatMap {α β : Type} (f : α → ℕ → List β) (j : ℕ) :
    ∀ vs : List α, ((vs.flatMap fun v => [f v]).flatMap fun o => o j) = vs.flatMap fun v => f v j
  | [] => rfl
  | v :: vs => by
    simp only [List.flatMap_cons, List.cons_append, List.nil_append]
    rw [flatMap_singleton_flatMap f j vs]

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Extraction succeeds** on the event, where the relation splits by table: the decoder lists each table's trace,
and `assemble` returns a witness that `check` accepts. -/
theorem assembles_of_extracts (D : Deploy Coin Col T) (decode : List (ℕ × Col) → List T)
    (hlw : ListWithin D.len D.enc D.θ decode) (x : Msg) (vs : List Seen) (hsp : Splits D x) (h : Extracts D x vs) :
    Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x) vs := by
  obtain ⟨s, hs, hall⟩ := h
  have hsome := assemble_isSome (table := D.table) (decode := decode) (read := D.read)
    (opens := vs.flatMap fun v => [D.cols v]) hs fun j hj => by
      obtain ⟨M, t, ht, hR, hag⟩ := hall j hj
      refine ⟨t, ht, M, ?_, hR⟩
      rw [flatMap_singleton_flatMap]
      exact hlw _ M hag
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hsome
  refine ⟨w, hw, ?_⟩
  obtain ⟨Ms, rfl, hlen, hMs⟩ := assemble_eq_some hs hw
  unfold check
  rw [hs]
  exact decide_eq_true (hsp s hs Ms hlen fun j hj => by
    obtain ⟨t, ht, -, hR⟩ := hMs j hj
    exact ⟨t, ht, hR⟩)

/-! ## One wait's bound, and KS -/

/-- **Flock's one-wait bound**, what the soundness proof gives at the deployed verifier: at every statement and prover,
escapes fitting the weight `C(x)` such that at every tape with acceptance `ε > 0`,
`ε · Q/ε^k ≤ ε_table + escapes(ω) + (N₀/(e·k))·ε`, with `Q` the probability that `k` independent runs all accept and
their opened columns don't extract. -/
def OneWait (H : Msg → Msg) (D : Deploy Coin Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.Fits (D.weight x) ∧
    ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (Extracts D x) / accPr H D.V P x ω ^ D.k x) ≤
        D.εtable x + e.term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] [DecidableEq Col] [Zero Col] in
/-- `E^rep`'s relative rate `N₀/(e·k)` is at most `1/4`, as `k = ⌈4·N₀/e⌉`. -/
theorem rate_le (D : Deploy Coin Col T) (x : Msg) (hk : 0 < D.k x) : D.N₀ x / (Real.exp 1 * D.k x) ≤ 1 / 4 := by
  have hk' : (0 : ℝ) < D.k x := by exact_mod_cast hk
  have he := Real.exp_pos 1
  rw [div_le_iff₀ (mul_pos he hk')]
  have hc : 4 * D.N₀ x / Real.exp 1 ≤ D.k x := Nat.le_ceil _
  rw [div_le_iff₀ he] at hc
  linarith

/-- The verifier accepts no run at `x`. -/
def NoAcc (H : Msg → Msg) (V : Verifier Coin) (x : Msg) : Prop :=
  ∀ (Ω : Type) (P : Prover Ω) (ω : Ω) (r : Fin (V.k x) → Coin), (run H P V x ω r).out ≠ .acc

/-- **Flock: KS from one wait's bound**, when `K > 0`, `ε_table ≥ 0`, and `k ≥ 8` and the relation splits by table
wherever the verifier accepts some run. -/
theorem KS_of_oneWait (H : Msg → Msg) (D : Deploy Coin Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, (8 ≤ D.k x ∧ Splits D x) ∨ NoAcc H D.V x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWait H D) :
    KS H D := by
  intro decode hlw x Ω _ _ P
  rcases hk x with ⟨hk8, hsp⟩ | hno
  swap
  · have h0 : PrBad H D.V (R D.parse) (E D decode) x P = 0 := by
      unfold PrBad
      rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    rw [h0]
    exact add_nonneg (κ_pos D x (hεt x)).le (Real.sqrt_nonneg _)
  obtain ⟨e, hfit, hw⟩ := h x Ω P
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x)
        (Assembles (fun v => [D.cols v]) (assemble D.parse D.table decode D.read x) (check D.parse x)) /
          accPr H D.V P x ω ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extracts D decode hlw x _ hsp hex)⟩
  have hb := ERep_trunc H D.V (R D.parse) D.waits D.k D.cap (fun v => [D.cols v])
    (assemble D.parse D.table decode D.read) (check D.parse) (check_sound D.parse) x P (hK x) hk8
    (fun ω => D.εtable x + e.term ω) (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by omega)) hwait
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  show PrBad H D.V (R D.parse)
      (ERep D.V D.waits D.k D.cap (fun v => [D.cols v]) (assemble D.parse D.table decode D.read) (check D.parse)) x
        P ≤
    κ D x + Real.sqrt (γ D x * PrB H D.V P x)
  unfold κ
  linarith

end Flock

end Claims
