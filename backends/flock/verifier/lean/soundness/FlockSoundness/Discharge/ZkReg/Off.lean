import FlockSoundness.Registered.Binding
import FlockSoundness.Discharge.ZkReg.Checked

/-!
# The registered leaves at each outcome's drawn units

`Registered.Reads.prob_off_le` bounds the event that some registered read's leaf is not the registrant's, for one read
set `rd` checked at every accepted outcome. The verifier's registered check at a `--zk` outcome reads only the rows of
that outcome's drawn instances (`checked_of_accepts` holds for reads that are the verifier's own at the outcome's draw,
`OwnReads`), so a read set over more units than one draw holds is checked at no outcome: with reads at units 0 and 1
under a law that draws one unit, `prob_off_le`'s `hchk` fails at every accepted outcome. Here the read set depends on
the outcome, and the check is asked only at the outcomes the game can reach (`Game.Out`):

* `drawnReads lay S`, `Reads.restrict`: the commit strings the units of `S` read, and `rd`'s reads there;
* `Reads.prob_off_le_run`: `prob_off_le` at reads and a registration that depend on the outcome;
* `Reads.prob_offDrawn_le`: at the reads of each outcome's drawn units, the event that a drawn unit reads a registered
  read whose leaf is not the registrant's;
* `zk_session_offR`: that in the live audit of `flock-verify verify --zk --registered F` sessions, under any law and live
  session, its check from `checked_of_accepts`, given that the registrant's reads at the draw are the verifier's own at
  each accepted live outcome `(R, L.draw ω, wr (L.draw ω) R lo)` (`hown`, the bridge's `FlockBridge.ZkReg.OwnAtDraws`,
  the shape `ownAtDraws_of_check` proves). At an arbitrary triple the draw need not be the transcript's recorded one,
  and `OwnReads` needs each read's unit in the recorded draw, so `hown` is asked at the live audit's outcomes only
  (`auditLive_outcome`).

Its event is the complement, at a registered read, of `zk_session_regValsR`'s leaf condition: the two together cover
every drawn unit's registered read.
-/

namespace FlockSoundness.Registered

open Game Refine Binding Audit
open scoped ENNReal

/-- `prob_mono` at the outcomes `G` can reach: an event that implies another at each of them is no likelier. -/
theorem prob_mono_reach {β : Type} {G : Game β} {p q : β → Prop} (h : ∀ o, Game.Out G o → p o → q o)
    (s : Strategy G) : prob p G s ≤ prob q G s := by
  classical
  calc prob p G s
      ≤ prob (fun o => q o ∨ (p o ∧ ¬ Game.Out G o)) G s :=
        prob_mono (fun o hp => by
          by_cases ho : Game.Out G o
          · exact Or.inl (h o ho hp)
          · exact Or.inr ⟨hp, ho⟩) G s
    _ ≤ prob q G s + prob (fun o => p o ∧ ¬ Game.Out G o) G s := prob_or_le _ _ G s
    _ = prob q G s := by
        rw [prob_eq_zero_of_out (p := fun o => p o ∧ ¬ Game.Out G o) (fun o ho hn => hn.2 ho) s, add_zero]

variable {C : Circuit} {n : ℕ} {P : Audit.Partition C n} {Reg : Type} {mPts : ℕ}
  {plan : Finset (Fin n) → Reg → List (TabSpec mPts)}
  {tab : (S : Finset (Fin n)) → (R : Reg) → Fin n → Fin (plan S R).length}
  {decode : Audit.Partition.DecoderB (C := C) (Reg := Reg) plan tab}

variable (lay : Binding.Layout P plan tab decode) in
/-- The commit strings the units of `S` read: those holding one of their wires. -/
def drawnReads (S : Finset (Fin n)) (p : lay.Pos) : Prop :=
  ∃ u ∈ S, ∃ g ∈ P.io u, lay.pos g = p

namespace Reads

variable {lay : Binding.Layout P plan tab decode} {rs : Binding.HmRows lay}

/-- `rd`'s reads where `S` holds, with the registrant's openings unchanged. -/
def restrict (rd : Reads lay rs) (S : lay.Pos → Prop) : Reads lay rs where
  isRead p := rd.isRead p ∧ S p
  read_row p h := rd.read_row p h.1
  port := rd.port
  readAt := rd.readAt
  wv := rd.wv
  ws := rd.ws
  wpath := rd.wpath
  committed p h := rd.committed p h.1

/-- **The registered reads' term at reads and a registration that depend on the outcome**, under `SHA512CRStrict`
(`cr/sha-512`): in a game `G` played by `σ` whose accepted outcomes `o` (those `G` can reach) pass the registered check
at the registration `R o` for the reads `rdo o`, through `path o`, the probability of accepting while some read of
`rdo o` has a leaf that is not the registrant's is at most `qR²/2^513`, for the finder that plays `G` once and outputs
the extractor's pair. -/
theorem prob_off_le_run {β : Type} (G : Game β) (σ : Strategy G) (accept : β → Prop) (R : β → Reg)
    (rdo : β → Reads lay rs) (path : β → lay.Pos → ByteArray)
    (hchk : ∀ o, Game.Out G o → accept o → (rdo o).Checked (R o) (path o)) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o => (rdo o).pick (R o) (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ (rdo o).Off (R o)) G σ ≤ ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

/-- **No drawn unit reads a registered read whose leaf is not the registrant's**, except with probability
`qR²/2^513` under `SHA512CRStrict` (`cr/sha-512`): in a game `G` played by `σ`, with drawn units `drawn o` and
registration `R o` at each outcome, if every accepted outcome `G` can reach passes the registered check on `rd`'s reads
at the commit strings its drawn units read (`hchk`), for the registered-path finder at those reads (`hR`). -/
theorem prob_offDrawn_le {β : Type} (G : Game β) (σ : Strategy G) (accept : β → Prop) (drawn : β → Finset (Fin n))
    (R : β → Reg) (path : β → lay.Pos → ByteArray) (rd : Reads lay rs)
    (hchk : ∀ o, Game.Out G o → accept o → (rd.restrict (drawnReads lay (drawn o))).Checked (R o) (path o)) {qR : ℝ}
    (hR : (StrictCR.oneRun G σ fun o => (rd.restrict (drawnReads lay (drawn o))).pick (R o) (path o)).CR H512 qR) :
    prob (fun o => accept o ∧ ∃ u ∈ drawn o, ∃ g ∈ P.io u, rd.isRead (lay.pos g) ∧
        lay.regLeaf (R o) (lay.pos g) ≠ rd.leaf (lay.pos g)) G σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end Reads

end FlockSoundness.Registered

namespace FlockSoundness.Discharge.ZkReg

open Game Audit Exec Refine Flock FlockLevel3 Model ZkLink ZkExec DecodeZL Placed
open scoped ENNReal

variable {mPts n : ℕ} {Reg : Type}

/-- **The live audit's outcomes**: the registration `R`, the law's draw at some `ω`, and the live session's transcript
at that draw and registration for some live outcome `lo`. -/
theorem auditLive_outcome {L : Audit.Law n} {dg : ByteArray → ByteArray} {N : ℕ}
    {wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript} {o : Reg × Finset (Fin n) × Exec.Transcript}
    (h : Game.Out (auditLive L Reg (liveOf dg mPts N wr)) o) :
    ∃ (R : Reg) (ω : L.Ω) (lo : LiveOut mPts), o = (R, L.draw ω, wr (L.draw ω) R lo) := by
  unfold auditLive at h
  cases h with
  | send R h =>
    cases h with
    | coin ω h =>
      obtain ⟨t, ht, rfl⟩ := Game.Out.of_map h
      unfold liveOf at ht
      obtain ⟨lo, -, rfl⟩ := Game.Out.of_map ht
      exact ⟨R, ω, lo, rfl⟩

/-- **No drawn unit of a `--zk --registered` session reads a registered read whose leaf is not the registrant's**,
except with probability `qR²/2^513` under `cr/sha-512` for the registered-path finder at each outcome's drawn reads
(`hR`): in the live audit under any law `L` and live session `wr`, played by `σ`, if the registrant's reads at the draw
are the verifier's own at each accepted live outcome `(R, L.draw ω, wr (L.draw ω) R lo)` (`hown`, the shape
`ownAtDraws_of_check` proves from `RecordsLiveZK` and `DrawOkZK`). With `L` the session's law, it covers the registered
reads `zk_session_regValsR` leaves out. -/
theorem zk_session_offR
    (I : Inputs) (ht : I.tags.typed = false) (own : Array Flock.Registered.Port)
    (dj : Finset (Fin n) → Lean.Json) {S₀ : Finset (Fin n)} (y₀ : DrawSetupZK I mPts S₀ (dj S₀))
    -- the session's circuit and public file
    {c : Flock.Circuit} (hc : HmRow.parse (zkTags I) I.circuitFile I.tables = .ok c) (hU : Built c.unit)
    {pub : Public} (hpub : HmRow.loadPublic (zkTags I) c I.publicFile = .ok pub)
    (hscope : Layout.scopeOk c = true)
    -- the verify command's tags
    (hTags : TagsOk I)
    -- the live audit: any law and live session
    (L : Audit.Law n) (wr : Finset (Fin n) → Reg → LiveOut mPts → Exec.Transcript) (dg : ByteArray → ByteArray) (N : ℕ)
    -- the registered reads, the verifier's own at the draw of each accepted live outcome
    (rd : FlockSoundness.Registered.Reads (laySh (Reg := Reg) ht dj y₀ hc hU (pub := pub))
      (rowsSh ht dj y₀ hc hU hpub hscope))
    (hown : ∀ (R : Reg) (ω : L.Ω) (lo : LiveOut mPts), LiveAcceptsZKR I own (R, L.draw ω, wr (L.draw ω) R lo) →
      OwnReads own c pub (wr (L.draw ω) R lo)
        (rd.restrict (FlockSoundness.Registered.drawnReads (laySh ht dj y₀ hc hU (pub := pub)) (L.draw ω))).isRead
        rd.port rd.readAt)
    -- the live prover, and cr/sha-512 for the registered-path finder
    (σ : Strategy (auditLive L Reg (liveOf dg mPts N wr))) {qR : ℝ}
    (hR : (StrictCR.oneRun (auditLive L Reg (liveOf dg mPts N wr)) σ fun o =>
      (rd.restrict (FlockSoundness.Registered.drawnReads (laySh ht dj y₀ hc hU (pub := pub)) o.2.1)).pick o.1
        (regPaths c pub o.2.2 rd.port)).CR H512 qR) :
    prob (fun o => LiveAcceptsZKR I own o ∧
        ∃ u ∈ o.2.1, ∃ g ∈ ((keyProg c hU n).partition (Layout.outsOf c hU n)).io u,
          rd.isRead ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ∧
          (laySh ht dj y₀ hc hU (pub := pub)).regLeaf o.1 ((laySh ht dj y₀ hc hU (pub := pub)).pos g) ≠
            rd.leaf ((laySh ht dj y₀ hc hU (pub := pub)).pos g))
        (auditLive L Reg (liveOf dg mPts N wr)) σ ≤
      ENNReal.ofReal (qR ^ 2 / 2 ^ (513 : ℝ)) := by
  sorry

end FlockSoundness.Discharge.ZkReg
