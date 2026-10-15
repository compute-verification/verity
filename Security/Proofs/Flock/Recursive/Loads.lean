import Proofs.Flock.Recursive.Raw
import Proofs.Flock.Recursive.VStarCheck
import Proofs.Flock.Soundness.Discharge.FailClosed.One

/-!
# Recursion: where V*'s statement enters a session (gap 8)

A session of V*'s statement `Q` is a `flock-verify verify --zk --registered` session of the verifier's own inputs `I`,
as the prover supplies them (`ZkOuterR`). What ties a session to `Q` is the verifier's check that the session is V*'s
statement of its index (`AcceptsZKV`, `RecVStar.checkStmt`), the one place the recursive audit's proof takes the
equation:

* `VStmt.Loads Q I`: the inputs load `Q`: their circuit is `Q`'s (`circZ I = Q.c`) and their registered population has
  `Q`'s `nv` units (`(pubZ I).n = Q.nv`). If V*'s statements fix only their unit, by name, and what the VBridge reads,
  this definition is what changes.
* `loads_of_check`, `loads_of_verifiesV`: at V*'s tags and no lookup tables (`parse_zkTags_vstar`), a session whose file
  parses to `Q`'s circuit and whose population is `Q`'s loads `Q`; a session `AcceptsZKV` accepts as statement `j` under
  `bind j` has statement `j`'s file and no lookup tables (`vstarFile_of_accepts`), whose parse is in scope
  (`scope_of_accepts`) and so is `Vs j`'s circuit (`VStarParsesScoped`). The verifier compares the tags by name only; the inputs' tags are the
  statement their name names (`TagsFound`, an obligation on our code), so they are V*'s (`find?_circuit`).
* `typable_of_accepts`: a session that accepts and loads `Q` types (`ZkOuterR.Typable`): `Loads` gives its circuit and
  population, and the record's draw sets table 0 up (`setupZHJ_of_acceptsZK`, at the record's custody).
* `prob_sessionsR_untypable`, `prob_recT_le`: where a session of some index never types, the raw sessions' event is
  empty when every session accepts as V*'s statement of its index; elsewhere the raw recursive audit's event is no
  likelier than the typed sessions' (`zsOf`) at the transported strategy (`recT`).
-/

namespace FlockSoundness.Discharge.Recursion

open Game Audit Exec Refine Flock FlockLevel3 Model Accounting ZkSession ZkLink ZkExec ZkBind Integrate Placed StrictCR
  DecodeZL Composed ZkReg FailClosed
open FlockSoundness.Discharge.Tables (LiveOutJ)
open scoped ENNReal

/-- **The inputs `I` load V*'s statement `Q`**: their circuit is `Q`'s and their registered population has `Q`'s
units. -/
def VStmt.Loads (Q : VStmt) (I : Inputs) : Prop :=
  circZ I = Q.c ∧ (pubZ I).n = Q.nv

/-- **`--zk`'s parse at V*'s statement is a constant of the circuit file**: at `--statement` `t₀`, `--zk`'s tags are
`t₀`'s but for the identity (`withCoins` on a coin tree leaves the tags as they are, `withCoins_tree`), which
`HmRow.parse` does not read (`parse_identity`), whatever the table count. -/
theorem parse_zkTags_vstar {I : Inputs} {t₀ : Flock.Tags} (hT : I.tags = t₀) (b : ByteArray)
    (tb : List (String × ByteArray)) :
    HmRow.parse (zkTags I) b tb = HmRow.parse (Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table])))
      (Flock.Zk.maskLogOf I.circuitFile) 1) b tb := by
  subst hT
  simp only [zkTags, zkCoins, Flock.Zk.tags, withCoins_tree, parse_identity]

/-- **A session of V*'s statement `Q` loads it**, at V*'s tags and no lookup tables, when its circuit file parses at
the file's constant tags to `Q`'s circuit and its registered population has `Q`'s units: the session's circuit is the
file's parse at `zkTags I` (`ScopeZR.parseH`, untyped: `Hidden.parse_untyped`), which is that constant
(`parse_zkTags_vstar`). -/
theorem loads_of_check {t₀ : Flock.Tags} {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg') (hck : ScopeZR x.I)
    (hT : x.I.tags = t₀) (htab : x.I.tables = [])
    (hp : HmRow.parse (Flock.Zk.tags (t₀.withCoins (.tree (Flock.Zk.coinSpec #[t₀.table])))
      (Flock.Zk.maskLogOf x.I.circuitFile) 1) x.I.circuitFile [] = .ok Q.c) (hn : (pubZ x.I).n = Q.nv) :
    Q.Loads x.I := by
  have hZ : HmRow.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok (circZ x.I) :=
    (Hidden.parse_untyped (zkTyped hck.typed) (hck.parseH x.hTags.1)).1
  rw [parse_zkTags_vstar hT, htab, hp] at hZ
  exact ⟨(Except.ok.inj hZ).symm, hn⟩

/-- **A session the verifier of record accepts as V*'s statement `j`, under the family's bindings for it, loads
`Vs j`**, where its inputs pass the statement check, at V*'s tags: `--rec-part j` makes the session's statement the
`j`th of the inner shape's, `--rec-bindings` makes its file that statement's under `bind j`, and the verifier refuses
lookup tables (`vstarFile_of_accepts`); then `loads_of_check` at the obligation. Refusal (5) compares the tags' names
only (`tags.name != Tags.circuit.name`); the inputs' tags are the statement their name names (`TagsFound`, an
obligation on our code), and `Tags.circuit`'s name names `Tags.circuit` (`find?_circuit`), so they are V*'s. -/
theorem loads_of_verifiesV {S : ℕ} {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings} {Vs : Fin S → VStmt}
    (hob : VStarParsesScoped Flock.Tags.circuit i bind Vs) (j : Fin S) {Reg' : Type} (x : (Vs j).OuterR Reg')
    (t : Exec.Transcript) {rows : Option (Array Flock.Registered.Port)} (hacc : AcceptsZKV i j (bind j) rows x.I x.own t)
    (hck : ScopeZR x.I) (hF : TagsFound x.I) : (Vs j).Loads x.I := by
  obtain ⟨htag, htab, ss, s, hss, hj, -, hn, hf, -⟩ := vstarFile_of_accepts x t hacc hck
  have hT : x.I.tags = Flock.Tags.circuit := by
    have h : Flock.Tags.find? x.I.tags.name = some x.I.tags := hF
    rw [htag, find?_circuit] at h
    exact (Option.some.inj h).symm
  have hZ : HmRow.parse (zkTags x.I) x.I.circuitFile x.I.tables = .ok (circZ x.I) :=
    (Hidden.parse_untyped (zkTyped hck.typed) (hck.parseH x.hTags.1)).1
  rw [parse_zkTags_vstar hT, htab] at hZ
  obtain ⟨hc, hnv⟩ := hob j ss s _ _ hss hj hf hZ (scope_of_accepts x t hacc hck)
  exact ⟨hc, hn.trans hnv.symm⟩

/-- **A session the verifier of record accepts types**, where its inputs pass the statement check and load V*'s
statement (`hL`): the record carries the draw (custody), at which table 0 sets up (`setupZHJ_of_acceptsZK`). -/
theorem typable_of_accepts {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg')
    (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => Q.J) x.wr) (ω : Q.law.live.Ω) (R : Reg') (o : LiveOutJ Q.J Q.mPts)
    (hacc : AcceptsZKR x.I x.own (x.wr (Q.law.live.draw ω) R o)) (hck : ScopeZR x.I) (hL : Q.Loads x.I) :
    x.Typable := by
  have hacc' := acceptsZK_of_R hacc
  exact ⟨hck, hL.1, hL.2, _, _, ZkHidden.setupZHJ_of_acceptsZK x.I x.hTags.1 hck.typed (hck.parseH x.hTags.1)
    (Checks.outNet hck) x.dj hacc' (hRec _ R o hacc').1 (x.hDraw ω) ⟨0, nTab_pos _⟩⟩

/-- **A session that never types accepts nowhere** where its inputs pass the statement check, for an event at which it
accepts and loads V*'s statement: an acceptance would type it (`typable_of_accepts`). -/
theorem prob_game_untypable {Q : VStmt} {Reg' : Type} (x : Q.OuterR Reg')
    (hRec : RecordsLiveZKJ x.I x.dj (fun _ _ => Q.J) x.wr)
    (hx : ¬ x.Typable) (hck : ScopeZR x.I) {P : Reg' × Finset (Fin Q.nv) × Exec.Transcript → Prop}
    (hP : ∀ o, P o → LiveAcceptsZKR x.I x.own o ∧ Q.Loads x.I) (dg : ByteArray → ByteArray) (N : ℕ)
    (τ : Strategy (x.game dg N)) : prob P (x.game dg N) τ = 0 :=
  le_antisymm ((prob_auditLive_liveJ_mono dg N x.wr τ (p := P) (q := fun _ => False)
    fun ω o' h => hx (typable_of_accepts x hRec ω τ.1 o' (hP _ h).1 hck (hP _ h).2)).trans (prob_false _ _).le)
    (zero_le ..)

/-- **Where a session of index `j₀` never types, the raw sessions' event is empty**, for an event at which every
session accepts as V*'s statement of its index and passes the statement check, where every session's inputs carry the
tags their name names (`TagsFound`): the event is `j₀`'s acceptance, at which `j₀`'s session loads `Vs j₀`
(`loads_of_verifiesV`), which `prob_game_untypable` empties at each of the prover's names for `j₀`
(`prob_seqC_exists_le`). -/
theorem prob_sessionsR_untypable {S : ℕ} {Vs : Fin S → VStmt} {Reg' : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)} (hob : VStarParsesScoped Flock.Tags.circuit i bind Vs)
    (zr : (j : Fin S) → (Vs j).OuterR Reg') (hRec : ∀ j, RecordsLiveZKJ (zr j).I (zr j).dj (fun _ _ => (Vs j).J) (zr j).wr)
    (hF : ∀ j, TagsFound (zr j).I) (j₀ : Fin S) (hx : ¬ (zr j₀).Typable) (p : VRec Vs Reg' dg N → Prop)
    (hp : ∀ b, p b → AcceptsRV i bind rows zr b ∧ ∀ j, ScopeZR (zr j).I) (s : Strategy (sessionsR zr dg N)) :
    prob p (sessionsR zr dg N) s = 0 := by
  by_cases hck : ScopeZR (zr j₀).I
  · have hP : ∀ o : Reg' × Finset (Fin (Vs j₀).nv) × Exec.Transcript,
        LiveAcceptsZKV i j₀ (bind j₀) rows (zr j₀).I (zr j₀).own o →
        LiveAcceptsZKR (zr j₀).I (zr j₀).own o ∧ (Vs j₀).Loads (zr j₀).I :=
      fun o h => ⟨h.1, loads_of_verifiesV hob j₀ (zr j₀) o.2.2 h hck (hF j₀)⟩
    refine le_antisymm (((prob_mono (q := fun b : VRec Vs Reg' dg N =>
        ∃ j, j = j₀ ∧ LiveAcceptsZKV i j (bind j) rows (zr j).I (zr j).own (b j).2)
        (fun b hb => ⟨j₀, rfl, (hp b hb).1 j₀⟩) _ _).trans
      (prob_seqC_exists_le S _ _ (fun j r => j = j₀ ∧ LiveAcceptsZKV i j (bind j) rows (zr j).I (zr j).own r.2)
        (fun _ _ => (0 : ℝ≥0∞)) ?_ s)).trans_eq ?_) (zero_le ..)
    · intro j τ
      by_cases hj : j = j₀
      · subst hj
        exact le_trans (prob_mono (fun a ha => ha.2) _ _) (prob_game_untypable _ (hRec _) hx hck hP dg N _).le
      · exact le_trans (prob_mono (q := fun _ => False) (fun a ha => hj ha.1) _ _) (prob_false _ _).le
    · simp only [Finset.sum_const_zero]
      exact exG_const 0 _ _
  · exact le_antisymm ((prob_mono (q := fun _ => False) (fun b hb => hck ((hp b hb).2 j₀)) _ _).trans
      (prob_false _ _).le) (zero_le ..)

/-- **The raw recursive audit's event is no likelier than the typed one's**, at the transported strategy (`recT`), when
the event has every session accept as V*'s statement of its index and pass the statement check (`hp`), where every
session's inputs carry the tags their name names (`hF`), and gives `q` at a whole history where every session types
(`hq`): where every session at a registration types, the typed sessions are the raw ones (`sessions_zsOf`,
`prob_sessT`); where one doesn't, the raw event is empty there (`prob_sessionsR_untypable`). -/
theorem prob_recT_le {S : ℕ} {Vs : Fin S → VStmt} {Reg' Reg₂ : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    {n : ℕ} {L : Law n} {Reg : Type} {I : Inner} {Cm : Type} {i : RecVStar.Inner} {bind : Fin S → VStar.Bindings}
    {rows : Option (Array Flock.Registered.Port)} (hob : VStarParsesScoped Flock.Tags.circuit i bind Vs)
    (zr : Reg → L.Ω → List (Cm × I.Coin) → Reg₂ → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (hF : ∀ R ω t R₂ j, TagsFound (zr R ω t R₂ j).I)
    (σ : Strategy (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N))
    (p q : Reg × L.Ω × List (Cm × I.Coin) × (Reg₂ × VRec Vs Reg' dg N) → Prop)
    (hp : ∀ o, p o → AcceptsVRV i bind rows (zr o.1 o.2.1 o.2.2.1) o.2.2.2 ∧
      ∀ j, ScopeZR (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).I)
    (hq : ∀ o, o.2.2.1.length = I.r → (∀ j, (zr o.1 o.2.1 o.2.2.1 o.2.2.2.1 j).Typable) → p o → q o) :
    prob p (recGame L Reg I Cm fun R ω t => outerVR (zr R ω t) dg N) σ ≤
      prob q (recGame L Reg I Cm fun R ω t => outerV (zsOf zr hRec h R ω t) dg N) (recT zr hRec h σ) := by
  refine prob_recMap_le (fun R ω t => outerT (zsOf zr hRec h R ω t) (zr R ω t)) p q σ fun ω t s ht => ?_
  rw [prob_outerVR, prob_outerT]
  by_cases hall : ∀ j, (zr σ.1 ω t s.1 j).Typable
  · rw [prob_sessT _ _ (sessions_zsOf zr hRec h σ.1 ω t s.1 hall)]
    exact prob_mono (fun b hb => hq (σ.1, ω, t, (s.1, b)) ht hall hb) _ _
  · push Not at hall
    obtain ⟨j₀, hj₀⟩ := hall
    rw [prob_sessionsR_untypable hob (zr σ.1 ω t s.1) (fun j => hRec σ.1 ω t s.1 j) (hF σ.1 ω t s.1) j₀ hj₀ _
      (fun b hb => hp (σ.1, ω, t, (s.1, b)) hb) _]
    exact zero_le ..

/-- **Where every session at a history and registration types, the typed sessions accept and pass the statement check
where the raw ones do**: each is its build (`zsOf_pos`), with the raw session's inputs and own ports. -/
theorem verifies_zsOf {S : ℕ} {Vs : Fin S → VStmt} {Reg' A B C D : Type} {dg : ByteArray → ByteArray} {N : ℕ}
    (zr : A → B → C → D → (j : Fin S) → (Vs j).OuterR Reg')
    (hRec : ∀ R ω t R₂ j, RecordsLiveZKJ (zr R ω t R₂ j).I (zr R ω t R₂ j).dj (fun _ _ => (Vs j).J)
      (zr R ω t R₂ j).wr) (h : Typed zr)
    (R : A) (ω : B) (t : C) (o : D × VRec Vs Reg' dg N) (hall : ∀ j, (zr R ω t o.1 j).Typable)
    (hacc : AcceptsVR (zr R ω t) o) (hck : ∀ j, ScopeZR (zr R ω t o.1 j).I) :
    AcceptsV (zsOf zr hRec h R ω t) o ∧ ∀ j, ScopeZR (zsOf zr hRec h R ω t o.1 j).I := by
  have hz : zsOf zr hRec h R ω t o.1 = fun j => (zr R ω t o.1 j).build (hall j) (hRec R ω t o.1 j) :=
    funext fun j => zsOf_pos zr hRec h R ω t o.1 j (hall j)
  unfold AcceptsV Accepts
  rw [hz]
  exact ⟨hacc, hck⟩

end FlockSoundness.Discharge.Recursion
