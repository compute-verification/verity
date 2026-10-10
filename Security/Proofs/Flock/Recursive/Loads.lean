import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.One

/-!
# Recursion: where V*'s statement enters a session (gap 8)

A session of V*'s statement `Q` is a `flock-verify verify --zk --registered` session of the verifier's own inputs `I`,
as the prover supplies them (`ZkOuterR`), and no check the verifier of record makes reads `Q`. What ties a session to
`Q` is one definition and one open lemma, the only place the recursive audit's proof takes the equation:

* `VStmt.Loads Q I`: the inputs load `Q`: their circuit is `Q`'s (`circZ I = Q.c`) and their registered population has
  `Q`'s `nv` units (`(pubZ I).n = Q.nv`). If V*'s statements fix only their unit, by name, and what the VBridge reads,
  this definition is what changes.
* `loads_of_verifies` (open): a session the verifier of record accepts, at inputs that pass the statement check
  `--registered` runs (`ScopeZR`), loads V*'s statement. The verifier's check that a session's circuit and population
  are V*'s is to prove it.
* `VStarParses` (an obligation, open until gap 4 builds `Vs j` from the partition): V*'s statement `j`'s file under its
  bindings parses, at V*'s constant tags and no lookup tables, to `Vs j`'s circuit, and `Vs j` has its instances.
* `loads_of_check`, `loads_of_verifiesV`: at V*'s tags and no lookup tables (`zkTags_vstar`), a session whose file is
  statement `j`'s under `bind j` loads `Vs j`; a session `AcceptsZKV` accepts as statement `j` under `bind j` has that
  file (`vstarFile_of_accepts`). This is `loads_of_verifies` at the check, once (i)'s event reads `AcceptsZKV` and the
  verifier refuses other tags and lookup tables.
* `typable_of_accepts`: such a session types (`ZkOuterR.Typable`): `Loads` gives its circuit and population, and the
  record's draw sets it up (`setupZ_of_verifiesZK`, at the record's custody).
* `prob_sessionsR_untypable`, `prob_recT_le`: where a session of some index never types, the raw sessions' event is
  empty; elsewhere the raw recursive audit's event is no likelier than the typed sessions' (`zsOf`) at the transported
  strategy (`recT`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed
open scoped ENNReal

/-- **The inputs `I` load V*'s statement `Q`**: their circuit is `Q`'s and their registered population has `Q`'s
units. -/
def VStmt.Loads (Q : VStmt) (I : Inputs) : Prop :=
  circZ I = Q.c ∧ (pubZ I).n = Q.nv

/-- Obligation: open until gap 4 builds `Vs j` from the partition, as this parse of V*'s file.
**V*'s statement `j`, written under its bindings, parses to `Vs j`'s circuit, and `Vs j` has one unit per instance.**
At V*'s tags `t₀` (the `--statement` the verifier requires of a V* session), the file `circuitFile stmts[j] (bind j)`
that `vstar stage` writes parses, under `--zk`'s tags for one table (`Zk.tags` on the coin tree of
`Zk.coinSpec #[t₀.table]`, at the file's `mask_log` and count 1) and with no lookup tables, to `(Vs j).c`; and
`(Vs j).nv` is the statement's instance count. A fact about our code, not a premise: the tags are a constant of the file,
so no session input enters it. -/
def VStarParses (t₀ : Flock.Tags) (stmts : Array VStar.Stmt) (bind : Fin stmts.size → VStar.Bindings)
    (Vs : Fin stmts.size → VStmt) : Prop :=
  ∀ j cf, RecVStar.circuitFile stmts[j] (bind j) = .ok cf →
    HmRow.parse (Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table]))) (Flock.Zk.maskLogOf cf) 1) cf
      [] = .ok (Vs j).c ∧ (Vs j).nv = stmts[j].instances

/-- **`--zk`'s tags at V*'s statement are a constant of the circuit file**: at `--statement` `t₀` and at most one table
(`TagsOk`), the coin tree's tables are `#[t₀.table]` (`tableNames`) and the identity's hooks are one table's
(`Zk.tags` reads only `count > 1`). -/
theorem zkTags_vstar {I : Inputs} {t₀ : Flock.Tags} (hT : I.tags = t₀) (hc : I.count ≤ 1) :
    zkTags I = Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table]))) (Flock.Zk.maskLogOf I.circuitFile)
      1 := by
  subst hT
  have h1 : ¬ (1 < I.count) := by omega
  simp only [zkTags, zkCoins, Flock.Zk.tags, Flock.Tags.tableNames, if_pos hc, gt_iff_lt, h1, Nat.lt_irrefl]

/-- **A session of V*'s statement `j` loads `Vs j`**, at V*'s tags and no lookup tables, when its circuit file is
`circuitFile stmts[j] (bind j)` and its registered population has the statement's instances: the session's circuit is
the file's parse at `zkTags I` (`ScopeZ.parse`), which is the obligation's constant (`zkTags_vstar`). -/
theorem loads_of_check {t₀ : Flock.Tags} {stmts : Array VStar.Stmt} {bind : Fin stmts.size → VStar.Bindings}
    {Vs : Fin stmts.size → VStmt} (hob : VStarParses t₀ stmts bind Vs) (j : Fin stmts.size) {Reg' : Type}
    (x : (Vs j).OuterR Reg') (hck : ScopeZR x.I) (hT : x.I.tags = t₀) (htab : x.I.tables = [])
    (hf : RecVStar.circuitFile stmts[j] (bind j) = .ok x.I.circuitFile) (hn : (pubZ x.I).n = stmts[j].instances) :
    (Vs j).Loads x.I := by
  obtain ⟨hp, hnv⟩ := hob j _ hf
  have hZ := (hck.scopeZ x.hTags.1).parse x.hTags.1
  rw [zkTags_vstar hT x.hTags.2.2.2, htab, hp] at hZ
  exact ⟨(Except.ok.inj hZ).symm, hn.trans hnv.symm⟩

/-- **A session the verifier of record accepts as V*'s statement `j`, under the family's bindings for it, loads
`Vs j`**, where its inputs pass the statement check, at V*'s tags and no lookup tables: `--rec-part j` makes the
session's statement `stmts[j]` and `--rec-bindings` makes its file `circuitFile stmts[j] (bind j)`
(`vstarFile_of_accepts`), and `loads_of_check` at the obligation. The tags and the tables are taken until the verifier
refuses a session at other tags or with lookup tables. -/
theorem loads_of_verifiesV {t₀ : Flock.Tags} {i : RecVStar.Inner} {stmts : Array VStar.Stmt}
    {bind : Fin stmts.size → VStar.Bindings} {Vs : Fin stmts.size → VStmt} (hob : VStarParses t₀ stmts bind Vs)
    (hss : RecVStar.statements i = .ok stmts) (j : Fin stmts.size) {Reg' : Type} (x : (Vs j).OuterR Reg')
    (t : Exec.Transcript) (hacc : AcceptsZKV i j (bind j) x.I x.own t) (hck : ScopeZR x.I) (hT : x.I.tags = t₀)
    (htab : x.I.tables = []) : (Vs j).Loads x.I := by
  obtain ⟨ss, s, hss', hj, -, hn, hf, -⟩ := vstarFile_of_accepts x t hacc hck
  rw [hss] at hss'
  cases hss'
  rw [Array.getElem?_eq_getElem j.2] at hj
  cases hj
  exact loads_of_check hob j x hck hT htab hf hn

/-- **Open (gap 8): a session the verifier of record accepts, at inputs that pass the statement check, loads V*'s
statement.** The one place `VStmt.Loads` enters the recursive audit's proof. No check of the verifier reads V*'s
statement yet, so this is a named gap and not a theorem of the executable; the verifier's check that a session's circuit
and population are V*'s statement's is to prove it. -/
theorem loads_of_verifies {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (t : Exec.Transcript)
    (_hacc : AcceptsZKR x.I x.own t) (_hck : ScopeZR x.I) : Q.Loads x.I := by
  sorry

/-- **A session the verifier of record accepts types**, where its inputs pass the statement check: they load V*'s
statement (`loads_of_verifies`), and the record carries the draw (custody), at which the session sets up
(`setupZ_of_verifiesZK`). -/
theorem typable_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (hRec : RecordsLiveZK x.I x.dj x.wr)
    (ω : Q.law.live.Ω) (R : Reg') (o : LiveOut Q.mPts)
    (hacc : AcceptsZKR x.I x.own (x.wr (Q.law.live.draw ω) R o)) (hck : ScopeZR x.I) : x.Typable := by
  have hL := loads_of_verifies x _ hacc hck
  have hacc' := acceptsZK_of_R hacc
  have hck' := hck.scopeZ x.hTags.1
  exact ⟨hck', hL.1, hL.2, _, setupZ_of_verifiesZK x.I x.hTags.1 x.hTags.2.2.2 x.dj ⟨hacc', hck'⟩
    (hRec _ R o hacc').1 (x.hDraw ω)⟩

/-- **A session that never types accepts nowhere** where its inputs pass the statement check: an acceptance would type
it (`typable_of_accepts`). -/
theorem prob_game_untypable {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (hRec : RecordsLiveZK x.I x.dj x.wr)
    (hx : ¬ x.Typable) (hck : ScopeZR x.I) (dg : ByteArray → ByteArray) (N : ℕ) (τ : Strategy (x.game dg N)) :
    prob (LiveAcceptsZKR x.I x.own) (x.game dg N) τ = 0 :=
  le_antisymm ((prob_auditLive_liveOf_mono dg N x.wr τ (p := LiveAcceptsZKR x.I x.own) (q := fun _ => False)
    fun ω o' h => hx (typable_of_accepts x hRec ω τ.1 o' h hck)).trans (prob_false _ _).le) (zero_le ..)

/-- **Where a session of index `j₀` never types, the raw sessions' event is empty**, for an event at which every
session accepts and passes the statement check: the event is `j₀`'s acceptance, which `prob_game_untypable` empties at
each of the prover's names for `j₀` (`prob_seqC_exists_le`). -/
theorem prob_sessionsR_untypable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : (j : Fin S) → (Vs j).OuterR Reg') (hRec : ∀ j, RecordsLiveZK (zr j).I (zr j).dj (zr j).wr) (j₀ : Fin S)
    (hx : ¬ (zr j₀).Typable) (p : VRec Vs Reg' dg N → Prop)
    (hp : ∀ b, p b → AcceptsR zr b ∧ ∀ j, ScopeZR (zr j).I) (s : Strategy (sessionsR zr dg N)) :
    prob p (sessionsR zr dg N) s = 0 := by
  by_cases hck : ScopeZR (zr j₀).I
  · refine le_antisymm (((prob_mono (q := fun b : VRec Vs Reg' dg N =>
        ∃ j, j = j₀ ∧ LiveAcceptsZKR (zr j).I (zr j).own (b j).2) (fun b hb => ⟨j₀, rfl, (hp b hb).1 j₀⟩) _ _).trans
      (prob_seqC_exists_le S _ _ (fun j r => j = j₀ ∧ LiveAcceptsZKR (zr j).I (zr j).own r.2)
        (fun _ _ => (0 : ℝ≥0∞)) ?_ s)).trans_eq ?_) (zero_le ..)
    · intro j τ
      by_cases hj : j = j₀
      · subst hj
        exact le_trans (prob_mono (fun a ha => ha.2) _ _) (prob_game_untypable _ (hRec _) hx hck dg N _).le
      · exact le_trans (prob_mono (q := fun _ => False) (fun a ha => hj ha.1) _ _) (prob_false _ _).le
    · simp only [Finset.sum_const_zero]
      exact exG_const 0 _ _
  · exact le_antisymm ((prob_mono (q := fun _ => False) (fun b hb => hck ((hp b hb).2 j₀)) _ _).trans
      (prob_false _ _).le) (zero_le ..)

/-- **The raw recursive audit's event is no likelier than the typed one's**, at the transported strategy (`recT`), when
the event has every session accept and pass the statement check (`hp`) and gives `q` where every session types (`hq`):
where every session at a registration types, the typed sessions are the raw ones (`sessions_zsOf`, `prob_sessT`); where
one doesn't, the raw event is empty there (`prob_sessionsR_untypable`). -/
theorem prob_recT_le {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type}
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZK (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (zr R ω t R₂ j).wr) (h : Typed zr)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N))
    (p q : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Prop)
    (hp : ∀ o, p o → AcceptsVR (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧ ∀ j, ScopeZR (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).I)
    (hq : ∀ o, (∀ j, (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).Typable) → p o → q o) :
    prob p (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
      prob q (recGame L Reg I Cm fun R ω t => outerV (zsOf zr hRec h R ω t) dg N) (recT zr hRec h σ) := by
  refine prob_recMap_le (fun R ω t => outerT (zsOf zr hRec h R ω t) (zr R ω t)) p q σ fun ω t s => ?_
  rw [prob_outerVR, prob_outerT]
  by_cases hall : ∀ j, (zr σ.1 ω t s.1 j).Typable
  · rw [prob_sessT _ _ (sessions_zsOf zr hRec h σ.1 ω t s.1 hall)]
    exact prob_mono (fun b hb => hq (σ.1, ω, t, (s.1, b)) hall hb) _ _
  · push Not at hall
    obtain ⟨j₀, hj₀⟩ := hall
    rw [prob_sessionsR_untypable (zr σ.1 ω t s.1) (fun j => hRec σ.1 ω t s.1 j) j₀ hj₀ _
      (fun b hb => hp (σ.1, ω, t, (s.1, b)) hb) _]
    exact zero_le ..

/-- **Where every session at a history and registration types, the typed sessions accept and pass the statement check
where the raw ones do**: each is its build (`zsOf_pos`), with the raw session's inputs and own ports. -/
theorem verifies_zsOf {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZK (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (o : D × VRec Vs Reg' dg N) (hall : ∀ j, (zr R ω t o.1 j).Typable)
    (hacc : AcceptsVR (zr R ω t) o) (hck : ∀ j, ScopeZR (zr R ω t o.1 j).I) :
    AcceptsV (zsOf zr hRec h R ω t) o ∧ ∀ j, ScopeZR (zsOf zr hRec h R ω t o.1 j).I := by
  have hz : zsOf zr hRec h R ω t o.1 = fun j => (zr R ω t o.1 j).build (hall j) (hRec R ω t o.1 j) :=
    funext fun j => zsOf_pos zr hRec h R ω t o.1 j (hall j)
  unfold AcceptsV Accepts
  rw [hz]
  exact ⟨hacc, hck⟩

end FlockSoundness.Discharge.Recursion
