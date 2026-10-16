import Proofs.Flock.Claims.KS.Cover

/-!
# The radius form: KS from one wait's bound at a filled word

The decoder spec in the old soundness theorems' form (Daniel's ruling of 16 Oct, as proofs decided it): `E^rep`
assembles one word `y` of the codeword's length from its accepted reruns' opened columns, the first column opened at
each position and the zero column where none is (`fillWord`, as `extractTable` fills), and the decoder must list every
trace `M` with `dist(y, enc M) ≤ θ·N`, distance per whole column (`hdist`), `θ = 1 − √ρ − 1/200` at the trace's code
(`ListCompleteθ`). `nMin` and the in-range condition are gone. Until `Definitions/Claims.lean` states it, the form is
here under the names the frozen text can adopt:

* `Radius`: what `Deploy` would carry for it: the word's length `N x` and zero column at the statement, and each
  trace's codeword length and radius;
* `assembleθ`, `Eθ`: `assemble` and `E` decoding the filled word;
* `ExtractsFill`, `OneWaitθ`: `Extracts` and `OneWait` at the filled word;
* `KSθ`: `Flock.KS` at `Eθ` and every decoder meeting `ListCompleteθ`, with the same `κ` and `γ`.

**KS from one wait's bound** (`KSθ_of_oneWaitθ`): `ExtractsFill` puts the trace in the decoder's list
(`assembles_of_extractsFill`), and the rest is `KS_of_oneWait`'s assembly. A decoder meets the spec at every radius
(`bruteθ`, `listCompleteθ_bruteθ`), so `KSθ` asks something of the verifier.

**At the deployed records** (`plainRadius`, `zkRadius`): the word has table 0's level-0 length `2^d` and its lanes'
zero column; a trace's code has length `2^d` and radius `1 − √(k/2^d) − 1/200` at its dimension `k` (`2^cols`, or
`2^cols + t` under `--zk`), the old theorems' `Level.radius` and `padRadius`. KS at both instances asks only
`OneWaitθ` (`plain_of_oneWaitθ_sha512`, `zk_of_oneWaitθ_sha512`).
-/

namespace Claims.Flock

variable {Col T : Type} [DecidableEq Col]

/-! ## The filled word and its distance to a codeword -/

/-- **The word `E^rep` assembles**: `N` positions, at each the first column opened there, `z` where none is. -/
def fillWord (z : Col) (N : ℕ) (os : List (ℕ × Col)) : List Col := (List.range N).map (fillAt z os)

omit [DecidableEq Col] in
theorem length_fillWord (z : Col) (N : ℕ) (os : List (ℕ × Col)) : (fillWord z N os).length = N := by
  simp [fillWord]

/-- **Distance per whole column** from a word to a codeword, over the word's positions. -/
def hdist (y : List Col) (c : ℕ → Col) : ℕ := ((Finset.range y.length).filter fun q => y[q]? ≠ some (c q)).card

theorem hdist_fillWord (z : Col) (N : ℕ) (os : List (ℕ × Col)) (c : ℕ → Col) :
    hdist (fillWord z N os) c = ((Finset.range N).filter fun q => fillAt z os q ≠ c q).card := by
  unfold hdist
  rw [length_fillWord]
  congr 1
  refine Finset.filter_congr fun q hq => ?_
  rw [Finset.mem_range] at hq
  simp [fillWord, hq]

/-! ## The spec, the extractor and the event -/

/-- **List decoding at radius `θ`**: from a word of a trace's codeword length, `decode` lists every trace whose
encoding is within `θ·N` of the word, per whole column. -/
def ListCompleteθ (enc : T → ℕ → Col) (len : T → ℕ) (θ : T → ℝ) (decode : List Col → List T) : Prop :=
  ∀ (y : List Col) (M : T), y.length = len M → (hdist y (enc M) : ℝ) ≤ θ M * len M → M ∈ decode y

/-- **The radius form's numbers**, what `Deploy` would carry for it: the word's length and zero column at each
statement (table 0's level 0), and each trace's codeword length and radius. -/
structure Radius (Col T : Type) where
  N : Msg → ℕ
  zero : Msg → Col
  len : T → ℕ
  θ : T → ℝ

/-- **Flock's assemble step at the filled word**: the opened columns filled into one word, decoded to candidate
traces, each read for its witness; the first candidate in the relation. -/
def assembleθ (H : Msg → Msg) (parse : Msg → Option Stmt) (Rd : Radius Col T) (decode : List Col → List T)
    (read : T → Wit) (x : Msg) (opens : List (ℕ × Col)) : Option Wit :=
  match parse x with
  | none => none
  | some s => ((decode (fillWord (Rd.zero x) (Rd.N x) opens)).map read).find? fun w => decide (RelS H s w)

/-- A decoder meets the spec at every radius: the traces of the word's length within the radius, listed. -/
noncomputable def bruteθ [Fintype T] (enc : T → ℕ → Col) (len : T → ℕ) (θ : T → ℝ) (y : List Col) : List T :=
  open Classical in (Finset.univ.filter fun M => y.length = len M ∧ (hdist y (enc M) : ℝ) ≤ θ M * len M).toList

theorem listCompleteθ_bruteθ [Fintype T] (enc : T → ℕ → Col) (len : T → ℕ) (θ : T → ℝ) :
    ListCompleteθ enc len θ (bruteθ enc len θ) := by
  intro y M hl hd
  unfold bruteθ
  rw [Finset.mem_toList, Finset.mem_filter]
  exact ⟨Finset.mem_univ _, hl, hd⟩

variable {Coin : Type} [Fintype Coin] [Nonempty Coin] [Inhabited Coin]

/-- **Flock's extractor at the filled word**: `E^rep` with `assembleθ`. -/
noncomputable def Eθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (decode : List Col → List T) :
    Extractor Coin Wit D.V :=
  ERep D.V D.waits D.k D.cap D.cols (assembleθ H D.parse Rd decode D.read) (check H D.parse)

/-- **Flock's extraction event at the filled word**, on accepted reruns `vs`: the word of their opened columns is
within `θ·N` of the encoding of a trace `M` of the word's length, and `M` reads as a witness of the parsed
statement. -/
def ExtractsFill (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (x : Msg) (vs : List Seen) : Prop :=
  ∃ (M : T) (s : Stmt), D.parse x = some s ∧ RelS H s (D.read M) ∧ Rd.N x = Rd.len M ∧
    (hdist (fillWord (Rd.zero x) (Rd.N x) (vs.flatMap D.cols)) (D.enc M) : ℝ) ≤ Rd.θ M * Rd.len M

omit [Fintype Coin] [Nonempty Coin] [Inhabited Coin] in
/-- **Extraction succeeds** on the event: the decoder lists the trace, and `assembleθ` returns a witness that `check`
accepts. -/
theorem assembles_of_extractsFill (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T)
    (decode : List Col → List T) (hlc : ListCompleteθ D.enc Rd.len Rd.θ decode) (x : Msg) (vs : List Seen)
    (h : ExtractsFill H D Rd x vs) :
    Assembles D.cols (assembleθ H D.parse Rd decode D.read x) (check H D.parse x) vs := by
  obtain ⟨M, s, hs, hrel, hN, hd⟩ := h
  have hM : M ∈ decode (fillWord (Rd.zero x) (Rd.N x) (vs.flatMap D.cols)) :=
    hlc _ M (by rw [length_fillWord, hN]) hd
  have hsome : (((decode (fillWord (Rd.zero x) (Rd.N x) (vs.flatMap D.cols))).map D.read).find? fun w =>
      decide (RelS H s w)).isSome :=
    List.find?_isSome.2 ⟨D.read M, List.mem_map_of_mem hM, by simpa using hrel⟩
  obtain ⟨w, hw⟩ := Option.isSome_iff_exists.1 hsome
  refine ⟨w, ?_, ?_⟩
  · unfold assembleθ
    rw [hs]
    exact hw
  · unfold check
    rw [hs]
    have hc := List.find?_some hw
    exact hc

/-- **Flock's one-wait bound at the filled word**: `OneWait` with `ExtractsFill` for `Extracts`. -/
def OneWaitθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) : Prop :=
  ∀ (x : Msg) (Ω : Type) [Fintype Ω] [Nonempty Ω] (P : Prover Ω), ∃ e : Escapes H D.V P x, e.Fits (D.weight x) ∧
    ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x) (ExtractsFill H D Rd x) / accPr H D.V P x ω ^ D.k x) ≤
        D.εtable x + e.term ω + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω

/-- **Flock: KS at the filled word**, at `Eθ` and every decoder that meets `ListCompleteθ`, with `Flock.KS`'s `κ`
and `γ`. -/
def KSθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) : Prop :=
  ∀ decode, ListCompleteθ D.enc Rd.len Rd.θ decode →
    Claims.KS H D.V (R H D.parse) (Eθ H D Rd decode) (κ D) (γ D)

/-- **Flock: KS at the filled word from one wait's bound**, when `K > 0`, `ε_table ≥ 0`, and `k ≥ 8` wherever the
verifier accepts some run. -/
theorem KSθ_of_oneWaitθ (H : Msg → Msg) (D : Deploy Coin Col T) (Rd : Radius Col T) (hK : ∀ x, 0 < D.cap x)
    (hk : ∀ x, 8 ≤ D.k x ∨ NoAcc H D.V x) (hεt : ∀ x, 0 ≤ D.εtable x) (h : OneWaitθ H D Rd) : KSθ H D Rd := by
  intro decode hlc x Ω _ _ P
  rcases hk x with hk8 | hno
  swap
  · have h0 : PrBad H D.V (R H D.parse) (Eθ H D Rd decode) x P = 0 := by
      unfold PrBad
      rw [Pr_congr (q := fun _ => False) fun s => ⟨fun hs => hno Ω P s.1 s.2.1 hs.1, False.elim⟩, Pr_false]
    rw [h0]
    exact add_nonneg (κ_pos D x (hεt x)).le (Real.sqrt_nonneg _)
  obtain ⟨e, hfit, hw⟩ := h x Ω P
  have hwait : ∀ ω, 0 < accPr H D.V P x ω →
      accPr H D.V P x ω * (accFail H D.V P x ω (D.k x)
        (Assembles D.cols (assembleθ H D.parse Rd decode D.read x) (check H D.parse x)) /
          accPr H D.V P x ω ^ D.k x) ≤
        (D.εtable x + e.term ω) + D.N₀ x / (Real.exp 1 * D.k x) * accPr H D.V P x ω := by
    intro ω hε
    refine le_trans ?_ ((hw ω hε).trans (le_of_eq (by ring)))
    refine mul_le_mul_of_nonneg_left (div_le_div_of_nonneg_right ?_ (pow_nonneg hε.le _)) hε.le
    unfold accFail
    exact Pr_mono fun σ ⟨hacc, hno⟩ => ⟨hacc, fun hex => hno (assembles_of_extractsFill H D Rd decode hlc x _ hex)⟩
  have hb := ERep_trunc H D.V (R H D.parse) D.waits D.k D.cap D.cols (assembleθ H D.parse Rd decode D.read)
    (check H D.parse) (check_sound H D.parse) x P (hK x) hk8 (fun ω => D.εtable x + e.term ω)
    (fun ω => add_nonneg (hεt x) (e.term_nonneg ω)) (rate_le D x (by omega)) hwait
  have hEx : 4 * Ex (fun ω => D.εtable x + e.term ω) ≤
      4 * D.εtable x + Real.sqrt (γ D x * PrB H D.V P x) := by
    rw [Ex_add, Ex_const]
    have := e.le (D.weight x) hfit
    unfold γ
    linarith
  show PrBad H D.V (R H D.parse)
      (ERep D.V D.waits D.k D.cap D.cols (assembleθ H D.parse Rd decode D.read) (check H D.parse)) x P ≤
    κ D x + Real.sqrt (γ D x * PrB H D.V P x)
  unfold κ
  linarith

end Claims.Flock

namespace FlockDeploy.KS

open Flock

local notation "𝔽" => FlockLevel3.GF128

/-! ## The deployed radius form -/

/-- **The radius at dimension `k` and length `n`**: `1 − √(k/n) − 1/200`, the old theorems' `Level.radius` and
`padRadius` at the code's rate. -/
noncomputable def radiusOf (k n : ℕ) : ℝ := 1 - Real.sqrt ((k : ℝ) / n) - 1 / 200

/-- **The word's length**: table 0's level-0 codeword length `2^d`; `0` where the statement does not set up. -/
def wordLen (zk : Bool) (x : Claims.Msg) : ℕ :=
  match (tables zk x).head? with
  | some (_, st) => 2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).d
  | none => 0

/-- **The zero column**: table 0's level-0 witness lanes, each `0`. -/
noncomputable def zeroCol (zk : Bool) (x : Claims.Msg) : List 𝔽 :=
  match (tables zk x).head? with
  | some (_, st) =>
    List.replicate (2 ^ (if zk then Claims.Flock.zkLevel0 st.m else Claims.Flock.plainLevel0 st.m).lanes) 0
  | none => []

/-- **The plain radius form**: a trace's code has length `2^d` and radius `1 − √(2^cols/2^d) − 1/200`. -/
noncomputable def plainRadius : Claims.Flock.Radius (List 𝔽) Claims.Flock.PlainTrace where
  N := wordLen false
  zero := zeroCol false
  len M := 2 ^ Claims.Flock.plainLen M.1
  θ M := radiusOf (2 ^ Claims.Flock.plainCols M.1) (2 ^ Claims.Flock.plainLen M.1)

/-- **The masked radius form**: a trace's padded code has length `2^d` and radius `1 − √((2^cols + t)/2^d) − 1/200`. -/
noncomputable def zkRadius : Claims.Flock.Radius (List 𝔽) Claims.Flock.ZkTrace where
  N := wordLen true
  zero := zeroCol true
  len M := 2 ^ Claims.Flock.zkLen M.1
  θ M := radiusOf (2 ^ Claims.Flock.zkCols M.1 + Claims.Flock.zkPad M.1) (2 ^ Claims.Flock.zkLen M.1)

/-- **The deployed plain verifier: KS at the filled word from one wait's bound**, at SHA-512. -/
theorem plain_of_oneWaitθ_sha512 (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.OneWaitθ sha512 (deployPlain rel) plainRadius) :
    Claims.Flock.KSθ sha512 (deployPlain rel) plainRadius :=
  Claims.Flock.KSθ_of_oneWaitθ sha512 (deployPlain rel) plainRadius (fun _ => cap_pos) (k_range_plain rel)
    (fun x => εtable_nonneg false x) h

/-- **The deployed masked (`--zk`) verifier: KS at the filled word from one wait's bound**, at SHA-512. -/
theorem zk_of_oneWaitθ_sha512 (rel : Readable.Parsed → Readable.Relation)
    (h : Claims.Flock.OneWaitθ sha512 (deployZk rel) zkRadius) :
    Claims.Flock.KSθ sha512 (deployZk rel) zkRadius :=
  Claims.Flock.KSθ_of_oneWaitθ sha512 (deployZk rel) zkRadius (fun _ => cap_pos) (k_range_zk rel)
    (fun x => εtable_nonneg true x) h

/-- A plain decoder meets the radius-form spec. -/
theorem listCompleteθ_plain (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListCompleteθ (deployPlain rel).enc plainRadius.len plainRadius.θ
      (Claims.Flock.bruteθ (deployPlain rel).enc plainRadius.len plainRadius.θ) :=
  Claims.Flock.listCompleteθ_bruteθ _ _ _

/-- A masked decoder meets the radius-form spec. -/
theorem listCompleteθ_zk (rel : Readable.Parsed → Readable.Relation) :
    Claims.Flock.ListCompleteθ (deployZk rel).enc zkRadius.len zkRadius.θ
      (Claims.Flock.bruteθ (deployZk rel).enc zkRadius.len zkRadius.θ) :=
  Claims.Flock.listCompleteθ_bruteθ _ _ _

end FlockDeploy.KS
