import Proofs.Flock.Soundness.Discharge.Exec.LawDraw
import Proofs.Flock.Soundness.Discharge.Exec.LiveOn

/-!
# The link points at the live law's fallback draw: statements

`liveOf dg mPts N wr` fixes the link points `mPts`, while setupH's `m_pts` follows the draw's proved units
(`setupH_mPts_of_proved`). The live stratified law draws `∑ s, k s` units on every run that succeeds
(`execOS_draw_card`) and every unit on a run that reads past the source's bytes (`execOS_draw_card_or`). This area
bounds the second kind of run and pays for it in the headline, so that the decoder is needed only at the draws of
`∑ s, k s` units, where setupH sets up at the one value `mPtsOf c.g (∑ s, k s)`.

* `WordUniform`: a word at a fixed offset of a uniform tape is uniform below `2^(8w)`.
* `WordReject`: `Flock.Draw.uniform m` rejects such a word with probability `(2^(8w) mod m) / 2^(8w)`, `w = width m`,
  at most `2^-64`.
* `WordsRun`: when each stratum's stream holds the `need` bytes `onWith` reads first and the first `k_s` words of
  each stream are accepted (`WordOK`, at the fixed offsets `wordOff`), the live run succeeds.
* `FallbackRare`, `DrawCardRare`: under A3, a run fails, and so the draw has other than `∑ s, k s` units, with
  probability at most `(∑ s, k s) · 2^-64` (a union bound over the words; no independence used).
* `DecodesWhere`, `simAuditWhere`, `ExecRefinesWhere`: R11 with the decoder right only at the law's draws that `good`
  admits, for the extra term `Pr[¬ good]` that `ExecRefines` carries as `prob bad`.
* `DecodesWhereOfOn`: `DecodesWhere` from a `DecodesOn` that holds at every law whose draws are all `good`.
* `MPtsAtCard`: setupH's `m_pts` at a draw file that proves exactly its draw's units (`hdj`) is `mPtsOf c.g` of the
  draw's size.
* `HeadlineCard`: `Prog.flock_headline_live_on` with the decoder needed only at the draws of `∑ s, k s` units, its
  bound gaining exactly `(∑ s, k s) · 2^-64`.

`hdj`, that the server's draw file proves exactly its draw's units, is a property of the file the challenger writes,
not of flock-verify; it stays a named hypothesis.
-/

namespace FlockSoundness.Discharge.DrawWords

open FlockSoundness.Game FlockSoundness.Audit FlockSoundness.Audit.Law FlockSoundness.Discharge.Exec Finset
open Model Accounting Refine Audit.Partition
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-! ## The first words of a stratum's stream -/

/-- Where `subset k size`'s `i`-th word starts when every earlier word was accepted: after the `width (size − j)` bytes
of each word `j < i`. -/
def wordOff (size i : ℕ) : ℕ := ∑ j ∈ range i, Flock.Draw.width (size - j)

/-- **`subset`'s `i`-th word is accepted at the first try**: the `width (size − i)`-byte big-endian word at
`wordOff size i` is below the largest multiple of `size − i` under `2^(8w)`, `Flock.Draw.uniform`'s test. -/
def WordOK (size : ℕ) (b : ByteArray) (i : ℕ) : Prop :=
  Flock.Draw.beNat b (wordOff size i) (Flock.Draw.width (size - i)) <
    2 ^ (8 * Flock.Draw.width (size - i)) - 2 ^ (8 * Flock.Draw.width (size - i)) % (size - i)

/-- **A word at a fixed offset of a uniform tape is uniform below `2^(8w)`.** -/
def WordUniform : Prop :=
  ∀ {ℓ : ℕ} (pos w : ℕ), pos + w ≤ ℓ → ∀ B : Finset ℕ,
    prCoin (fun g : Fin ℓ → Fin 256 => Flock.Draw.beNat (streamOf g) pos w ∈ B) =
      ((B.filter (· < 2 ^ (8 * w))).card : ℝ≥0∞) / ((2 ^ (8 * w) : ℕ) : ℝ≥0∞)

/-- **A word is rejected with probability `(2^(8w) mod m) / 2^(8w)`, at most `2^-64`**, `w = width m`: the word at a
fixed offset of a uniform tape fails `Flock.Draw.uniform m`'s test. -/
def WordReject : Prop :=
  ∀ {ℓ : ℕ} (pos m : ℕ), 0 < m → pos + Flock.Draw.width m ≤ ℓ →
    prCoin (fun g : Fin ℓ → Fin 256 => ¬ Flock.Draw.beNat (streamOf g) pos (Flock.Draw.width m) <
        2 ^ (8 * Flock.Draw.width m) - 2 ^ (8 * Flock.Draw.width m) % m) =
      ((2 ^ (8 * Flock.Draw.width m) % m : ℕ) : ℝ≥0∞) / ((2 ^ (8 * Flock.Draw.width m) : ℕ) : ℝ≥0∞) ∧
    ((2 ^ (8 * Flock.Draw.width m) % m : ℕ) : ℝ≥0∞) / ((2 ^ (8 * Flock.Draw.width m) : ℕ) : ℝ≥0∞) ≤
      (2 ^ 64 : ℝ≥0∞)⁻¹

/-- **The deterministic lemma.** When each stratum's stream holds at least the `need` bytes that `onWith` reads first,
and the first `k_s` words of each stream are accepted at the first try, the live stratified run on those bytes
succeeds. -/
def WordsRun : Prop :=
  ∀ {m : ℕ} (S : Fin m → Flock.Draw.Stratum) (K pop : ℕ) {L : Fin m → ℕ} (f : (i : Fin m) → Fin (L i) → Fin 256),
    (∀ s, (S s).need ≤ L s) → (∀ s, ∀ i < (S s).k, WordOK (S s).size (streamOf (f s)) i) →
    ∃ d, runOS f (.stratified K (Array.ofFn S)) pop = some d

/-- **The live run reads past the source's bytes with probability at most `(∑ s, k s) · 2^-64`**, on a uniform byte
source (A3) whose streams hold at least `need` bytes each. -/
def FallbackRare : Prop :=
  ∀ {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ), (∀ s, k s ≤ (stratum σ s).card) →
    ∀ (S : Fin m → Flock.Draw.Stratum), ExecStrata σ k S → ∀ (K pop : ℕ) {Ω : Type} [Fintype Ω]
    (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256), Assumptions.UniformRandomBytes L t →
    (∀ s, (S s).need ≤ L s) →
    prCoin (fun ω => runOS (t ω) (.stratified K (Array.ofFn S)) pop = none) ≤
      ((∑ s, k s : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹

/-- **The live law's draw has other than `∑ s, k s` units with probability at most `(∑ s, k s) · 2^-64`**, under the
same conditions. -/
def DrawCardRare : Prop :=
  ∀ {n m : ℕ} (σ : Fin n → Fin m) (k : Fin m → ℕ), (∀ s, k s ≤ (stratum σ s).card) →
    ∀ (S : Fin m → Flock.Draw.Stratum), ExecStrata σ k S → ∀ (K pop : ℕ) {Ω : Type} [Fintype Ω] [Nonempty Ω]
    (L : Fin m → ℕ) (t : Ω → (i : Fin m) → Fin (L i) → Fin 256), Assumptions.UniformRandomBytes L t →
    (∀ s, (S s).need ≤ L s) →
    prCoin (fun ω => ((execOS (n := n) t (.stratified K (Array.ofFn S)) pop).draw ω).card ≠ ∑ s, k s) ≤
      ((∑ s, k s : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹

/-! ## R11 with the decoder right at the `good` draws only -/

section Where

variable {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}

variable (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
  (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
  (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
  (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
  (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
  (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ)

/-- **The decoder is right on the sessions flock-verify accepts, at the law's draws that `good` admits** (R11b):
`DecodesOn` at the draws `L.draw ω` with `good (L.draw ω)` only. -/
def DecodesWhere (L : Law n) (good : Finset (Fin n) → Prop) : Prop :=
  ∀ ω, good (L.draw ω) → ∀ R, Decodes (fun (caps : Fin 1 → Cap D) pend os => tableOutL H E Lf (caps 0) pend os)
    (fun o => Accepts I (wr (L.draw ω) R o)) (·.accepted = true) (opn (L.draw ω) R) (dec (L.draw ω) R)
    (one (L.draw ω) R) (fin (L.draw ω) R) (capOf (L.draw ω) R) (repsOf (D := D) execArith (T (L.draw ω) R))

/-- **The simulated audit strategy, from the decoder at the `good` draws**: `simAuditOn`'s, with the live verdict read
as `good (L.draw ω) ∧ Accepts`, so that at a draw `good` refuses the simulation is `simSessL`'s at a verdict that never
holds, where `Decodes` is vacuous. -/
noncomputable def simAuditWhere {L : Law n} (good : Finset (Fin n) → Prop)
    (hdec : DecodesWhere H E Lf T I wr opn dec one fin capOf L good)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) :
    Strategy (audit L Reg (batchedSessionL H E Lf fun S R => [T S R])) :=
  ⟨τ.1, fun ω => Strategy.map (fun ok => (ok, L.draw ω)) _ (Strategy.map _ _
    (simSessL execArith H E Lf (T (L.draw ω) τ.1) (fun o => good (L.draw ω) ∧ Accepts I (wr (L.draw ω) τ.1 o))
      (opn (L.draw ω) τ.1) dg (dec (L.draw ω) τ.1) (one (L.draw ω) τ.1) (fin (L.draw ω) τ.1) (capOf (L.draw ω) τ.1) N
      (fun o ho => hdec ω ho.1 τ.1 o ho.2)
      (Strategy.ofMap (wr (L.draw ω) τ.1) _
        (Strategy.ofMap (fun t => (τ.1, L.draw ω, t)) _ (τ.2 ω)))))⟩

end Where

/-- **R11 for one-table sessions, with the decoder right at the `good` draws only.** Against any live prover `τ`,
flock-verify's acceptance is no likelier than the salted audit's acceptance against `simAuditWhere … τ`, plus the
live event that the draw is not `good`. -/
def ExecRefinesWhere : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (N : ℕ) {L : Law n} (good : Finset (Fin n) → Prop)
    (hdec : DecodesWhere H E Lf T I wr opn dec one fin capOf L good)
    (τ : Strategy (auditLive L Reg (liveOf dg mPts N wr))),
    ExecRefines (auditLive L Reg (liveOf dg mPts N wr)) τ (LiveAccepts I) (fun o => ¬ good o.2.1)
      (audit L Reg (batchedSessionL H E Lf fun S R => [T S R]))
      (simAuditWhere H E Lf T I wr opn dg dec one fin capOf N good hdec τ)

/-- **`DecodesWhere` from `DecodesOn`**: a decoder right at every law whose draws are all `good` is right at the `good`
draws of any law (at the law on the `good` outcomes). -/
def DecodesWhereOfOn : Prop :=
  ∀ {D Sl : Type} [Nonempty D] [Inhabited D] [Inhabited Sl] {mPts n : ℕ} {Reg : Type}
    (H : List UInt8 → D) (E : Merkle.Enc (List 𝔽) D) (Lf : Merkle.Leaf (List 𝔽) Sl D H)
    (T : Finset (Fin n) → Reg → TabSpec mPts) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 D × Salts Sl)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap D)) (L : Law n) (good : Finset (Fin n) → Prop),
    (∀ L' : Law n, (∀ ω, good (L'.draw ω)) → DecodesOn H E Lf T I wr opn dec one fin capOf L') →
    DecodesWhere H E Lf T I wr opn dec one fin capOf L good

/-- **`m_pts` at a draw file that proves exactly its draw's units** (`hdj`, a property of the file the challenger
writes): every statement setupH accepts at the draw file `dj S` has `m_pts = mPtsOf c.g S.card`, `c` the circuit
file's parse. At the live law's draws of `∑ s, k s` units that is the one value `mPtsOf c.g (∑ s, k s)`. -/
def MPtsAtCard : Prop :=
  ∀ {c : Flock.Circuit} {n : ℕ} {I : Inputs}, I.tags.typed = false →
    Flock.HmRow.parse I.tags I.circuitFile I.tables = .ok c → ∀ (dj : Finset (Fin n) → Lean.Json),
    (∀ B D, Flock.Draw.UnitDraw.ofJson (dj B) = .ok D → D.proved.size = B.card) →
    ∀ (S : Finset (Fin n)) (st : Flock.Stmt),
    Flock.Stmt.setupH I.tags I.circuitFile I.publicFile I.coins I.tables (some (dj S)) I.partition I.program = .ok st →
    st.mPts = mPtsOf c.g S.card

/-- **The headline at the executable, for one-table sessions, with the decoder right only at the draws of
`∑ s, k s` units.** `Prog.flock_headline_live_on`'s hypotheses, with `hdec` asked only at the law's draws of
`∑ s, k s` units (`DecodesWhere`), and each stratum's stream holding at least the `need` bytes `onWith` reads first
(`hLb`). Its bound is `Prog.flock_headline_live_on`'s plus exactly `(∑ s, k s) · 2^-64`: `Prog.flock_headline_execL`'s
`Pr[bad]` at `bad o := o.2.1.card ≠ ∑ s, k s`, bounded by `DrawCardRare` under A3. -/
def HeadlineCard : Prop :=
  ∀ {N' n : ℕ} (p : FlockSoundness.Prog N' n) (outs : Finset (Fin N')) {Reg : Type} {mPts : ℕ}
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (_hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (_hA3 : Assumptions.UniformRandomBytes Lb src) (_hLb : ∀ s, (St s).need ≤ Lb s)
    (T : Finset (Fin n) → Reg → TabSpec mPts)
    (tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin [T S R].length)
    (pp : ProgPlaces p outs (fun S R => [T S R]) tab)
    (ones : Finset (Fin N')) (_hConst : (p.partition outs).ConstCols (fun S R => [T S R]) tab pp.derived.place ones)
    (zeros : Finset (Fin N')) (_hZero : (p.partition outs).ZeroCols (fun S R => [T S R]) tab pp.derived.place zeros)
    (lay : Binding.Layout (p.partition outs) (fun S R => [T S R]) tab
      ((p.partition outs).placeDecoder (fun S R => [T S R]) tab pp.derived.place))
    (rs : Binding.HmRows lay) (hHm : rs.Computes)
    {k : ℕ} (_hk : 1 ≤ k) {Rw M : ℕ} (_hRw : 1 ≤ Rw) (_hM : 1 ≤ M) {ρ : ℝ} (_hρ : ρ < 1)
    (_hr : ∀ S R j, rateB [T S R] j k ≤ ρ) {t' : ℝ} (_ht : 0 ≤ t')
    (K : ℕ) (I : Inputs) (wr : Finset (Fin n) → Reg → LiveOut mPts → Transcript)
    (opn : Finset (Fin n) → Reg → Array ByteArray → Opens 𝔽 Dig × Salts (List UInt8)) (dg : ByteArray → ByteArray)
    (dec : Finset (Fin n) → Reg → ℕ → ByteArray → List Msg) (one : Finset (Fin n) → Reg → ℕ → Bool)
    (fin : Finset (Fin n) → Reg → Fin 2 → Array ByteArray → List Msg)
    (capOf : Finset (Fin n) → Reg → ByteArray → (Fin 1 → Cap Dig)) (N : ℕ)
    (hdec : DecodesWhere H512 enc512 leaf512 T I wr opn dec one fin capOf
      (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (fun S => S.card = ∑ s, ks s))
    (τ : Strategy (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)))
    (_hCR : LinkCRL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
      (reg (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ))
      (cont (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ)) k Rw M t')
    {qF qS : ℝ} (_hKS : ∀ ω j, TableCRL execArith H512 enc512 leaf512
      [T ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
        (reg (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ))] j
      (batchStratL H512 enc512 leaf512 (fun S R => [T S R])
        (cont (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ) ω)) qF qS),
    prob (fun o => LiveAccepts I o ∧
        K ≤ ((p.partition outs).wrong
          (Xpub ones zeros
            (XplurL (p.partition outs) H512 enc512 leaf512 (fun S R => [T S R]) tab (rs.binding hHm)
              (reg (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ))
              (cont (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ)) k Rw))).card)
        (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr)) τ ≤
      (Law.stratified σs ks hks).miss K +
        StrictCR.ksAvgStrict (fun S R => [T S R]) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
          (reg (simAuditWhere H512 enc512 leaf512 T I wr opn dg dec one fin capOf N _ hdec τ)) +
        (p.partition outs).linkBoundE (L := Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
          (fun S R => [T S R]) tab (rs.binding hHm) k Rw M ρ t' +
        ((∑ s, ks s : ℕ) : ℝ≥0∞) * (2 ^ 64 : ℝ≥0∞)⁻¹

end FlockSoundness.Discharge.DrawWords
