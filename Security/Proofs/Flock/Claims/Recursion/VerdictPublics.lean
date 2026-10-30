import Proofs.Flock.Claims.Recursion.VerdictFramed

/-!
# The decoded session's public words are empty, at every decision

`TargetPublics`: at a decision of `nextR`, the session the verdict decodes has no public words. The reader's record
writes its link's `publics` as the empty object (`linkObj_publics`), the substitutions that make the verdict's record
(`jAt`) keep the link's `publics` (`fld_link_jAt`), and the decoder reads a table missing from `publics` as empty bytes
(S7, `decodeJWithK_publics`).
-/

namespace FlockDeploy

open Flock
open Lean (Json)
open FlockComplete.JsonRT (recordObj recordList linkObj linkList getObjVal?_mkObj recordList_distinct
  linkList_distinct)
open FlockSoundness.Refine (ebind_ok)

/-- **S7 reads a table missing from `publics` as empty bytes.** -/
theorem decodeJWithK_publics (keep : Bool → Bool) (hs : Hashes Flock.V) {spec : SessionSpec} {j : Json}
    {sess : Session} (h : Record.decodeJWithK keep hs spec j = .ok sess) :
    ∃ l pubs, Record.fld j "link" = .ok l ∧ Record.fld l "publics" = .ok pubs ∧
      List.Forall₂ (fun t p => Record.isNull pubs t = true → p = ByteArray.empty) spec.tables.toList
        sess.publics.toList := by
  unfold Record.decodeJWithK at h
  dsimp only at h
  repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
  all_goals
    cases h
    refine ⟨_, _, by assumption, by assumption, ?_⟩
    refine List.Forall₂.imp (fun t p h hnull => ?_) (Claims.Bridge.Custody.mapM_forall₂ (by assumption))
    beta_reduce at h
    try dsimp only at h
    first | rw [if_pos hnull] at h | simp only [hnull, if_true] at h
    obtain ⟨p', hp', h⟩ := ebind_ok h
    simp only [pure, Except.pure, Except.ok.injEq] at hp'
    subst hp'
    repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
    all_goals
      simp only [pure, Except.pure, Except.ok.injEq] at h
      first | exact h.symm | exact h

theorem isNull_mkObj_nil (t : String) : Record.isNull (Json.mkObj []) t = true := rfl

/-- **The verdict's record keeps the reader's link `publics`.** -/
theorem fld_link_jAt {H : Claims.Msg → Claims.Msg} {spec : SessionSpec} {sts : Array Stmt} {c₁ : Coin}
    {mids : List (Claims.Msg × Coin)} {f : FinalP} {l : Json}
    (h : Record.fld (jAt H spec sts c₁ mids f) "link" = .ok l) :
    ∃ l₀, Record.fld f.j "link" = .ok l₀ ∧ l.getObjVal? "publics" = l₀.getObjVal? "publics" := by
  unfold jAt coinSubJ fillCoins eraseCoins at h
  simp only [mapRounds, setPoints] at h
  rw [fld_mapKey_ne _ _ (by decide : "link" ≠ "streams")] at h
  obtain ⟨l₁, h1, rfl⟩ := fld_mapKey_self h
  rw [fld_mapKey_ne _ _ (by decide : "link" ≠ "streams")] at h1
  obtain ⟨l₂, h2, rfl⟩ := fld_mapKey_self h1
  rw [shaSubJ, fld_mapKey_ne _ _ (proofKey_ne_link spec).symm] at h2
  exact ⟨l₂, h2, by rw [getObjVal?_mapKey_ne _ _ (by decide), getObjVal?_mapKey_ne _ _ (by decide)]⟩

theorem fld_recordObj_link (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) (streams : Array (Array Round)) (proofs : Array ByteArray) :
    Record.fld (recordObj spec h rootB points streams proofs) "link" = .ok (linkObj spec h rootB points) := by
  have hv : (recordObj spec h rootB points streams proofs).getObjVal? "link" = .ok (linkObj spec h rootB points) :=
    getObjVal?_mkObj (recordList_distinct ..) (by simp [recordList])
  unfold Record.fld
  rw [hv]

theorem linkObj_publics (spec : SessionSpec) (h : ByteArray → ByteArray) (rootB : Array ByteArray)
    (points : Array (Array F128)) : (linkObj spec h rootB points).getObjVal? "publics" = .ok (Json.mkObj []) :=
  getObjVal?_mkObj (linkList_distinct ..) (by simp [linkList])

end FlockDeploy

namespace RecDeploy.Rows.DecAt

open Flock
open FlockDeploy (Coin FinalP)

variable {rd : Claims.Msg → Option FinalP} {x : Claims.Msg} {ms : List Claims.Msg} {cs : List Coin}

/-- **`TargetPublics`: the decoded session's public words are empty**, at every decision. -/
theorem publics_empty (D : DecAt rd x ms cs) : (D.sess.publics.all fun p => p.size == 0) = true := by
  have hdec := D.decode
  rw [Record.decodeJWith_eq_decodeJWithK] at hdec
  obtain ⟨l, pubs, hl, hp, hF⟩ := FlockDeploy.decodeJWithK_publics id _ hdec
  obtain ⟨l₀, hl₀, hpub⟩ := FlockDeploy.fld_link_jAt hl
  obtain ⟨r₀, rss, ms', -, -, hj, -⟩ := FlockDeploy.decR_framesR D.dec
  rw [hj, FlockDeploy.recJ, FlockDeploy.fld_recordObj_link] at hl₀
  injection hl₀ with hl₀
  subst hl₀
  rw [FlockDeploy.linkObj_publics] at hpub
  unfold Record.fld at hp
  rw [hpub] at hp
  dsimp only at hp
  injection hp with hp
  subst hp
  have hall : ∀ p ∈ D.sess.publics.toList, p = ByteArray.empty := by
    intro p hp
    obtain ⟨i, hi, rfl⟩ := List.mem_iff_getElem.1 hp
    have hlen := (FlockDeploy.forall₂_getElem.1 hF).1
    exact (FlockDeploy.forall₂_getElem.1 hF).2 i (by omega) hi (FlockDeploy.isNull_mkObj_nil _)
  rw [Array.all_eq_true_iff_forall_mem]
  intro p hp
  rw [hall p (Array.mem_toList_iff.2 hp)]
  rfl

end RecDeploy.Rows.DecAt
