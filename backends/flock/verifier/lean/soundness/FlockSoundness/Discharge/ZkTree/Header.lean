import FlockSoundness.Discharge.ZkTree.SoundAt

/-!
# The record's header, read by flock-verify

The coin server's record of a `--zk` session carries the session's header twice: the prover's `Hello`, with its nonce
in `coins`, and `coin_tree = {key, nonce, root, spec}`. `Record.decode` reads them at S2 and nowhere else: S2's
coin-tree branch (`HeaderOk`) checks `Hello` against the verifier's own (`CoinTree.checkHello`) and the record's
`coin_tree` at its nonce (`CoinTree.checkRecord`), and discards the key; every later step reads the parsed record at
other keys, and `Zk.verify` reads the record only through `Record.decode`, `setupOf` only through its `unit_draw`.

* `decode_header`: two records that parse, agree at every key but `hello` and `coin_tree` (`AgreeOffHeader`) and
  both pass S2 decode to the same session.
* `headerOk_of_decode`: a record `Record.decode` accepts on the coin tree passes S2.
* `acceptsZK_header`: so flock-verify `--zk` accepts a record whose header was replaced by one that passes S2 whenever
  it accepts the original.
* `HeaderOnly`, `headerSound_of_only`: a writer `wrH` that writes, at every header, the reference writer `wr`'s record
  with only `hello` and `coin_tree` changed, `wr`'s own header passing S2, is accepted only where `wr` is
  (`HeaderSound`).
* `prRealH_le`: an event that holds at `wrH` only if it holds at `wr` is no likelier under `wrH`.
* `zk_session_sound_live`: `zk_session_sound_real` with `HeaderOnly`, a statement of what the writer writes, in place
  of `HeaderBlind`, a statement of what flock-verify accepts.

`HeaderBlind`'s converse half is false: a prover chooses its nonce, and a `Hello` whose nonce is not 64 lowercase hex
digits fails S2 (`checkHello`), so the record `wrH` writes at that header is refused where `wr`'s may be accepted. The
bound needs only the half `HeaderSound` states.
-/

namespace FlockSoundness.Discharge.ZkTree

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed Discharge.ZkCoins
open Lean (Json)
open scoped ENNReal

local notation "𝔽" => FlockLevel3.GF128

/-- **S2 on the coin tree** (`Record.decode`'s S2 when the spec names a coin tree), at a parsed record `j`: the
record's `hello` is the verifier's own `Hello` with a nonce put in, and its `coin_tree` passes `checkRecord` at that
nonce. -/
def HeaderOk (spec : SessionSpec) (j : Json) : Prop :=
  ∃ ct own hs nonce ctj key, spec.coinTree = some ct ∧ Json.parse spec.hello = .ok own ∧
    j.getObjVal? "hello" = .ok (.str hs) ∧ CoinTree.checkHello own hs = .ok nonce ∧
    j.getObjVal? "coin_tree" = .ok ctj ∧ CoinTree.checkRecord ct nonce ctj = .ok key

/-- Two parsed records with the same value, or the same absence, at every key but `hello` and `coin_tree`. -/
def AgreeOffHeader (j j' : Json) : Prop :=
  ∀ k, k ≠ "hello" → k ≠ "coin_tree" → j.getObjVal? k = j'.getObjVal? k

theorem str_of_get {j : Json} {k s : String} (h : j.getObjVal? k = .ok (.str s)) : Record.str j k = pure s := by
  simp only [Record.str, Record.fld, h, Json.getStr?]
  rfl

theorem fld_of_get {j v : Json} {k : String} (h : j.getObjVal? k = .ok v) : Record.fld j k = pure v := by
  simp only [Record.fld, h]
  rfl

theorem get_of_fld {j v : Json} {k : String} (h : Record.fld j k = .ok v) : j.getObjVal? k = .ok v := by
  unfold Record.fld at h
  split at h
  · rename_i hv
    cases h
    exact hv
  · cases h

theorem get_of_str {j : Json} {k s : String} (h : Record.str j k = .ok s) : j.getObjVal? k = .ok (.str s) := by
  unfold Record.str at h
  obtain ⟨v, hv, h⟩ := ebind_ok h
  rw [get_of_fld hv]
  split at h
  · rename_i hs
    cases h
    cases v <;> simp only [Json.getStr?, reduceCtorEq] at hs
    cases hs
    rfl
  · cases h

theorem ok_bind {ε α β : Type} (a : α) (f : α → Except ε β) : (Except.ok a >>= f) = f a := rfl

/-- **The header is read at S2 only**: records that agree off the header and both pass S2 decode alike. -/
theorem decode_header {spec : SessionSpec} {r r' : String} {j j' : Json} (hj : Json.parse r = .ok j)
    (hj' : Json.parse r' = .ok j') (hag : AgreeOffHeader j j') (h : HeaderOk spec j) (h' : HeaderOk spec j') :
    Record.decode spec r = Record.decode spec r' := by
  obtain ⟨ct, own, hs, nonce, ctj, key, hct, hown, h1, h2, h3, h4⟩ := h
  obtain ⟨ct', own', hs', nonce', ctj', key', hct', hown', h1', h2', h3', h4'⟩ := h'
  obtain rfl : ct = ct' := Option.some.inj (hct.symm.trans hct')
  obtain rfl : own = own' := by
    have := hown.symm.trans hown'
    injection this
  unfold Record.decode
  rw [hj, hj']
  dsimp only
  simp only [hct, hown, Except.toOption, str_of_get h1, str_of_get h1', fld_of_get h3, fld_of_get h3', h2, h2',
    pure_bind]
  simp only [ok_bind, h4, h4']
  have hstr : ∀ k, k ≠ "hello" → k ≠ "coin_tree" → Record.str j k = Record.str j' k := fun k a b => by
    simp only [Record.str, Record.fld, hag k a b]
  have hfld : ∀ k, k ≠ "hello" → k ≠ "coin_tree" → Record.fld j k = Record.fld j' k := fun k a b => by
    simp only [Record.fld, hag k a b]
  have harr : ∀ k, k ≠ "hello" → k ≠ "coin_tree" → Record.arr j k = Record.arr j' k := fun k a b => by
    simp only [Record.arr, Record.fld, hag k a b]
  have hnull : ∀ k, k ≠ "hello" → k ≠ "coin_tree" → Record.isNull j k = Record.isNull j' k := fun k a b => by
    simp only [Record.isNull, hag k a b]
  have hdk : spec.dk = "sha512" ∨ spec.dk = "sha256" := by
    unfold SessionSpec.dk
    split <;> simp
  have hget : ∀ k, k ≠ "hello" → k ≠ "coin_tree" → j.getObjVal? k = j'.getObjVal? k := hag
  rcases hdk with hdk | hdk <;> rw [hdk] <;> simp (disch := decide) only [hstr, hfld, harr, hnull, hget]

/-- **A record decoded on the coin tree passes S2.** -/
theorem headerOk_of_decode {spec : SessionSpec} {r : String} {j : Json} {sess : Session} {ct : CoinTree.Spec}
    (hct : spec.coinTree = some ct) (hj : Json.parse r = .ok j) (h : Record.decode spec r = .ok sess) :
    HeaderOk spec j := by
  unfold Record.decode at h
  rw [hj] at h
  dsimp only at h
  simp only [hct, pure_bind] at h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  obtain ⟨_, _, h⟩ := ebind_ok h
  split at h
  · rename_i own hown
    obtain ⟨hs, h1, h⟩ := ebind_ok h
    obtain ⟨nonce, h2, h⟩ := ebind_ok h
    obtain ⟨ctj, h3, h⟩ := ebind_ok h
    obtain ⟨key, h4, -⟩ := ebind_ok h
    have hown' : Json.parse spec.hello = .ok own := by
      cases hp : Json.parse spec.hello <;> simp only [hp, Except.toOption, reduceCtorEq, Option.some.injEq] at hown
      rw [hown]
    exact ⟨ct, own, hs, nonce, ctj, key, hct, hown', get_of_str h1, h2, get_of_fld h3, h4⟩
  · cases h

/-- **flock-verify `--zk` is blind to a header that passes S2**: it accepts the record with its header replaced
whenever it accepts the original. -/
theorem acceptsZK_header {I : Inputs} {t t' : Exec.Transcript} {j j' : Json} (hp : t'.proofs = t.proofs)
    (hj : Json.parse t.record = .ok j) (hj' : Json.parse t'.record = .ok j') (hag : AgreeOffHeader j j')
    (h' : ∀ spec sts, setupOf (zkInputs I) t'.record = .ok (spec, sts) → HeaderOk spec j') :
    AcceptsZK I t → AcceptsZK I t' := by
  rintro ⟨spec, sts, hset, hshape, hver⟩
  have hdraw : recordDraw t'.record = recordDraw t.record := by
    unfold recordDraw
    rw [hj, hj']
    dsimp only
    rw [hag "unit_draw" (by decide) (by decide)]
  have hset' : setupOf (zkInputs I) t'.record = .ok (spec, sts) := by
    unfold setupOf at hset ⊢
    rw [hdraw]
    exact hset
  refine ⟨spec, sts, hset', hshape, ?_⟩
  obtain ⟨hsome, sess, -, hdec, -⟩ := zk_verify_ok _ _ _ _ hver
  obtain ⟨ct, hct⟩ := Option.isSome_iff_exists.1 hsome
  have hd := decode_header hj hj' hag (headerOk_of_decode hct hj hdec) (h' spec sts hset')
  rw [hp]
  unfold Flock.Zk.verify at hver ⊢
  rw [← hd]
  exact hver

variable {mPts n : ℕ} {Reg : Type}

/-- **The writer changes only the header**: at every header `h`, `wrH` writes `wr`'s proofs and a record that parses,
as `wr`'s does, and agrees with it off `hello` and `coin_tree`; and `wr`'s header passes S2 at every spec the
record's draw sets up. -/
def HeaderOnly {Hd : Type} (I : Inputs) (wrH : Hd → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ h S R o, (wrH h S R o).proofs = (wr S R o).proofs ∧ ∃ j j', Json.parse (wrH h S R o).record = .ok j ∧
    Json.parse (wr S R o).record = .ok j' ∧ AgreeOffHeader j j' ∧
    ∀ spec sts, setupOf (zkInputs I) (wr S R o).record = .ok (spec, sts) → HeaderOk spec j'

/-- **flock-verify accepts what `wrH` writes only where it accepts what `wr` writes.** -/
def HeaderSound {Hd : Type} (I : Inputs) (wrH : Hd → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) : Prop :=
  ∀ h S R o, AcceptsZK I (wrH h S R o) → AcceptsZK I (wr S R o)

theorem headerSound_of_only {Hd : Type} {I : Inputs}
    {wrH : Hd → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript}
    {wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript} (h : HeaderOnly I wrH wr) :
    HeaderSound I wrH wr := by
  intro hd S R o hacc
  obtain ⟨hp, j, j', hj, hj', hag, hok⟩ := h hd S R o
  exact acceptsZK_header hp.symm hj hj' hag hok hacc

open Classical in
/-- `prR` is monotone in its event. -/
theorem prR_mono {I Blk Salt Op α : Type} {p p' : α → Prop} (hp : ∀ a, p a → p' a) (op : OpenView I Blk Salt Op) :
    ∀ {g : TGame I Blk α} {G : Game α} (h : ErD g G) (s : OStrat Op G) (z : I → Blk × Salt),
      prR p op h s z ≤ prR p' op h s z := by
  intro g G h
  induction h with
  | ret a =>
      intro s z
      simp only [prR]
      split_ifs with h1 h2
      · exact le_rfl
      · exact absurd (hp a h1) h2
      · exact (zero_le : (0 : ℝ≥0∞) ≤ _)
      · exact le_rfl
  | send M f ih => intro s z; exact ih s.1 s.2 z
  | coin i d f ih => intro s z; exact ih _ _ _ z
  | fresh C f ih => intro s z; exact le_rfl

section Run

variable {E₂ I Blk Cs D Nonce Root Op β γ : Type} [Field E₂] [Fintype E₂] [Fintype I] [DecidableEq I] [Fintype Blk]

/-- **An event that holds at `outH` only where it holds at `out`** is no likelier with the header written. -/
theorem prRealH_le (L : Audit.Law n) {g : TGame I Blk β} {G : Game β} (hgG : ErD g G)
    (outH : Nonce × Key E₂ × Root → Finset (Fin n) → Reg → β → γ) (out : Finset (Fin n) → Reg → β → γ)
    (dig : Nonce → I → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    (rt : Nonce → Key E₂ → (I → D) → Root) (op : Nonce → Key E₂ → (I → D) → OpenView I Blk (Salt E₂) Op)
    (ρ : RealProver L Reg Nonce E₂ Root Op G) (E : Reg × Finset (Fin n) × γ → Prop)
    (hE : ∀ h S R o, E (R, S, outH h S R o) → E (R, S, out S R o)) :
    prRealH L hgG outH dig cs H rt op ρ E ≤ prReal L hgG out dig cs H rt op ρ E :=
  avg_mono fun _ => avg_mono fun _ => avg_mono fun z => prR_mono (fun _ => hE _ _ _ _) _ hgG _ z

end Run

/-- **One `--zk` sampled session of the live coin tree is sound**, the writer changing only the record's header
(`HeaderOnly`). -/
theorem zk_session_sound_live
    (I : Inputs) (ht : I.tags.typed = false)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript)
    (dg : ByteArray → ByteArray) (N : ℕ)
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub) (hpt : pub.tables = none)
    (hscope : Layout.scopeOk c = true)
    -- the law and A3
    {m : ℕ} (σs : Fin n → Fin m) (ks : Fin m → ℕ) (hks : ∀ s, ks s ≤ (Law.stratum σs s).card)
    (St : Fin m → Flock.Draw.Stratum) (hS : Law.ExecStrata σs ks St) (Kd pop : ℕ)
    {Ω : Type} [Fintype Ω] [Nonempty Ω] (Lb : Fin m → ℕ) (src : Ω → (i : Fin m) → Fin (Lb i) → Fin 256)
    (hA3 : Assumptions.UniformRandomBytes Lb src)
    -- the coin server's draw at each of the law's draws
    (hDraw : ∀ ω, DrawOkZK I dj mPts ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω))
    -- the verify command's tags and the coin server's record of the live session
    (hTags : TagsOk I) (hRec : RecordsLiveZK I dj wr)
    -- the count curve's parameters
    {k Rw M : ℕ} (hk : 1 ≤ k) (hRw : 1 ≤ Rw) (hM : 1 ≤ M) {ρ : ℝ} (hρ : ρ < 1)
    (hr : ∀ S R j, rateZ (planZAt (Reg := Reg) dj y₀ S R) j k ≤ ρ) {t' : ℝ} (ht' : 0 ≤ t') (K₀ : ℕ)
    -- the coin tree: `nS` streams of `2^rl` rounds, the session stream `iS` and the reps' streams `i0`, `i1`
    {E₂ Blk Cs D Nonce Root Op : Type} [Field E₂] [Fintype E₂] [DecidableEq E₂] (hE : Fintype.card E₂ = 2)
    [Fintype Blk] [Nonempty Blk] [DecidableEq Blk] [DecidableEq Cs] [Inhabited Op]
    {nS : ℕ} (rl : ℕ) (iS i0 i1 : Fin nS) (hS0 : iS ≠ i0) (hS1 : iS ≠ i1) (h01 : i0 ≠ i1)
    (dl : Option (Dec Blk (Fin 2 → Fin mPts → 𝔽))) (dr : (n : ℕ) → Option (Dec Blk (Fin n → 𝔽)))
    -- the leaves: the round digests, the salt hash (at most `2^512` values) and the keyed leaf hash, at each nonce
    (dig : Nonce → Fin nS × Fin (2 ^ rl) → Blk → (Fin 512 → E₂)) (cs : Nonce → Salt E₂ → Cs)
    (hcs : ∀ ν, (Finset.univ.image (cs ν)).card ≤ 2 ^ 512) (H : Nonce → Key E₂ → (Fin 512 → E₂) × Cs → D)
    -- the root and the openings the prover is shown, and the coin server's record with the session's header
    (rt : Nonce → Key E₂ → (Fin nS × Fin (2 ^ rl) → D) → Root)
    (op : Nonce → Key E₂ → (Fin nS × Fin (2 ^ rl) → D) → OpenView (Fin nS × Fin (2 ^ rl)) Blk (Salt E₂) Op)
    (wrH : Nonce × Key E₂ × Root → Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (hOnly : HeaderOnly I wrH wr)
    -- the prover of the live session
    (ρ' : RealProver (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg Nonce E₂ Root Op
      (liveTable dg mPts N))
    {qF qS : ℝ} :
    ∃ Mx : Mix (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
        PUnit, ∃ vw : Mx.Ρ → TreeView (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) E₂
          (Fin nS × Fin (2 ^ rl)) Blk D,
      (∀ x, Mx.σ x = (ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop)
        (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) ∧
      ((∀ v ω j, TableCRZ execArith hs512 (planZAt dj y₀
          ((Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop).draw ω)
          (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)))) j
        (zcStrat execArith hs512 (planZAt dj y₀)
          (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr v)) ω)) qF qS) →
        prRealH (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wrH
            dig cs H rt op ρ' (LiveAcceptsZK I) ≤
          ∑ x, Mx.w x * prob (fun o => LiveAcceptsZK I o ∧
              (((keyProg c hU n).partition (Layout.outsOf c hU n)).wrong
                (XplurZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                  ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                  (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                  (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) k Rw)).card < K₀)
              (auditLive (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) Reg (liveOf dg mPts N wr))
              ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)) +
          ((Law.stratified σs ks hks).miss K₀ +
            ∑ x, Mx.w x * (ksAvgStrictZ (planZAt dj y₀) (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) k qF qS
                (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))) +
              linkBoundZC execArith hs512 ((keyProg c hU n).partition (Layout.outsOf c hU n)) (planZAt dj y₀) (tabZAt dj y₀)
                ((rsZAt ht dj y₀ hc hU hpub hpt hscope).bindingZ (rsZAt_computes ht dj y₀ hc hU hpub hpt hscope))
                k Rw M ρ t' (reg (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x))))
                (cont (simZ execArith hs512 _ I wr _ dg _ _ _ _ N (decodesOnZK_of_runsOnZC execArith hs512 _ I wr _ _ _ _ (runsZC_live I ht dj y₀ wr _ tablesAtZK_live hTags hRec hDraw)) ((ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op).play (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) wr (vw x)))))) +
          ((nS * 2 ^ rl : ℕ) : ℝ≥0∞) * ENNReal.ofReal (1 / 2 ^ 256)) := by
  obtain ⟨Mx, vw, hσ, -, hb⟩ := zk_session_sound_tree_at I ht dj y₀ wr dg N hc hU hpub hpt hscope σs ks hks St hS
    Kd pop Lb src hA3 hDraw hTags hRec hk hRw hM hρ hr ht' K₀ hE rl iS i0 i1 hS0 hS1 h01 dl dr dig cs hcs H
    (ρ'.toTree (Law.execOS (n := n) src (.stratified Kd (Array.ofFn St)) pop) (erD_liveTableC dg rl i0 i1 dr iS dl N) rt op) (qF := qF) (qS := qS)
  refine ⟨Mx, vw, hσ, fun hCR => ?_⟩
  exact (prRealH_le _ _ wrH wr _ _ _ _ _ _ _ fun h S R o => headerSound_of_only hOnly h S R o).trans
    ((realProver_le_tree _ _ wr dig cs H rt op ρ' _).trans (hb hCR))

end FlockSoundness.Discharge.ZkTree
