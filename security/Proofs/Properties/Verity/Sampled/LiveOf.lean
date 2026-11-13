module

public import Proofs.Properties.Verity.Sampled.SessionAt
public import Proofs.Flock.Bridge.DrawnLaw
public import Proofs.Flock.Bridge.Refused
public import Proofs.Flock.Soundness.Discharge.FailClosed.Scope
public import Proofs.Flock.Soundness.Discharge.ZkBind.Rate

/-!
# What v2's acceptance gives of the session at the registered files (`liveOf`)

The deployment's session at the parameters `θ` is `liveOf`'s on the registered files (`Instance.sessionOf`): one v2
sets up on them but for the coin commitment's families (`LiveSpec`), where one exists, and none elsewhere, where
SPOT's published δ is `⊤`. Each field of such a session is a value v2's check derives on an accepted play, and this file
derives them:

* **The parameters** (`program_of_paramsAt`): the verifier's program is the one a descriptor parses to, as v2's
  parameters decode it (`Spot.OneStage.paramsParsed`).
* **The sampling design** (`lawAt_designAt`): where v2 samples, the design `lawAt` reads off its parameters is
  `designAt`'s at the verifier's descriptor, query, law object and work table.
* **The lookup tables** (`pinnedTables_eq`): on blobs that hold the registered circuit's lookup tables and no other
  pinned table (`HoldsRegisteredTables`), each at its registered file (`TablesSent`), the tables v2 reads are the ones
  the circuit file registers.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

/-! ## The parameters -/

/-- A do block's `if c then throw e` before the rest `jp` accepts only where the rest does. -/
theorem jp_ok {ε γ : Type} {c : Prop} [Decidable c] {e : ε} {jp : Unit → Except ε γ} {p : γ}
    (h : (if c then (throw e : Except ε Unit) >>= jp else jp ()) = .ok p) : jp () = .ok p := by
  split at h
  · cases h
  · exact h

/-- **v2's parameters' verifier program is one some bytes parse to**: `paramsParsed` parses it from the canonical text
of the parameters' `program`. -/
set_option maxHeartbeats 2000000 in -- `paramsParsed` is some hundred binds, each peeled against the rest of the block
theorem paramsParsed_program {j : Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsParsed j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsParsed at h
  repeat' (first
    | (with_reducible obtain ⟨_, _, h⟩ := ebind_ok h)
    | (with_reducible replace h := jp_ok h; try beta_reduce at h)
    | (split at h))
  all_goals first
    | (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact ⟨_, by assumption⟩)

theorem oneStage_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStage.Params}
    (h : _root_.Spot.OneStage.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.verifier.program := by
  unfold _root_.Spot.OneStage.paramsOfJson at h
  obtain ⟨p', hp', h⟩ := ebind_ok h
  replace h := jp_ok h
  simp only [pure, Except.pure, Except.ok.injEq] at h
  subst h
  exact paramsParsed_program hp'

theorem oneStageZk_paramsOfJson_program {j : Json} {p : _root_.Spot.OneStageZk.Params}
    (h : _root_.Spot.OneStageZk.paramsOfJson j = .ok p) :
    ∃ b, _root_.Spot.Program.parse b = .ok p.base.verifier.program := by
  unfold _root_.Spot.OneStageZk.paramsOfJson at h
  repeat' (first
    | (obtain ⟨_, _, h⟩ := ebind_ok h)
    | (split at h)
    | (cases h; done))
  all_goals first
    | (simp [throw, throwThe, MonadExceptOf.throw] at h; done)
    | (simp only [pure, Except.pure, Except.ok.injEq] at h
       subst h
       exact oneStage_paramsOfJson_program ‹_›)

/-- **v2's parameters in `θ` name a verifier program some descriptor parses to** (`paramsParsed_program`). -/
theorem program_of_paramsAt {θ : ByteArray} {p : _root_.Spot.OneStageZk.Params} (hp : ParamsAt θ p) :
    ∃ desc : Claims.Msg, _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program := by
  obtain ⟨_, -, x, -, hj⟩ := hp
  obtain ⟨_, -, hj⟩ := ebind_ok hj
  obtain ⟨_, hr⟩ := Binding.dispatch_params hj
  obtain ⟨b, hb⟩ := oneStageZk_paramsOfJson_program hr
  exact ⟨b.data.toList, hb⟩

/-! ## The sampling design -/

/-- **Where v2 samples, `lawAt` reads `designAt`'s design**: at a verifier whose program the descriptor `desc` parses
to, under the sampling design object `Draw.derive` writes, a sample `drawnFrom` makes is under `drawnLaw`'s design
(`drawsUnder_of_drawnFrom`), which is `designAt`'s there and the one `lawAt` reads: the design `deriveLaw` expands,
else the one `flockLaw` parses from the object. -/
theorem lawAt_designAt {desc : Claims.Msg} {v : _root_.Spot.Registration.Verifier} {lo : Json}
    (hp : _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok v.program)
    (hlo : _root_.Spot.Draw.derive v.program v.query v.law v.workTable = .ok lo) {n : ℕ} {src : ByteArray}
    {r : Array ℕ × ℕ} (h : _root_.Spot.OneStage.drawnFrom v lo n src = .ok r) :
    lawAt v n = .ok (Flock.Bridge.designAt desc v.query v.law v.workTable n) := by
  obtain ⟨law, ⟨hl, -⟩, -⟩ := Flock.Bridge.drawsUnder_of_drawnFrom hp hlo h
  obtain ⟨p', derived, parsed, hp', hd, hpar, rfl⟩ := Flock.Bridge.drawnLaw_ok hl
  obtain rfl : p' = v.program := Except.ok.inj (hp'.symm.trans hp)
  have hdes : Flock.Bridge.designAt desc v.query v.law v.workTable n = derived.getD parsed := by
    unfold Flock.Bridge.designAt
    rw [hl]
  rw [hdes]
  have hlo' := hlo
  unfold _root_.Spot.Draw.derive at hlo'
  obtain ⟨derived', hd', hlo'⟩ := Properties.Spot.SecurityProofs.bind_eq_ok.1 hlo'
  obtain rfl : derived' = derived := Except.ok.inj (hd'.symm.trans hd)
  obtain rfl : Properties.Spot.SecurityProofs.lawObj v.law derived' = lo := Except.ok.inj hlo'
  simp only [lawAt, hlo, hd, hpar, ok_bind]
  rfl

/-! ## The lookup tables -/

/-- **On blobs holding the registered circuit's lookup tables, each at its registered file, v2's tables are the ones the
circuit file registers**: a pinned table the blobs hold is one the circuit looks up (`HoldsRegisteredTables`) at its
registered file (`TablesSent`), and one they don't hold it doesn't look up. -/
theorem pinnedTables_eq {s : Verity.Sent} {circuitB : ByteArray}
    (hl : _root_.Spot.OneStageZk.lookupsOf circuitB =
      .ok ((_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs)).map (·.1)))
    (ht : TablesSent s) :
    pinnedTables circuitB = _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray s.blobs) := by
  classical
  simp only [pinnedTables, hl]
  rw [tablesOf_ofArray]
  refine List.filterMap_congr fun t htm => ?_
  cases hf : tableFile t.sha512 s.blobs with
  | none =>
    rw [Option.map_none, if_neg]
    intro hc
    have hm : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) := by simpa using hc
    obtain ⟨y, hy, hyn⟩ := List.mem_map.1 hm
    obtain ⟨t', ht', hft'⟩ := List.mem_filterMap.1 hy
    obtain ⟨a, ha, rfl⟩ := Option.map_eq_some_iff.1 hft'
    have htt : t' = t := List.inj_on_of_nodup_map FlockSoundness.Refine.mufu_nodup ht' htm hyn
    subst htt
    rw [hf] at ha
    cases ha
  | some a =>
    have hmem : t.name ∈ ((Flock.MUFU_TABLES.filterMap fun t =>
        (tableFile t.sha512 s.blobs).map fun b => (t.name, b)).map (·.1)) :=
      List.mem_map.2 ⟨(t.name, a), List.mem_filterMap.2 ⟨t, htm, by rw [hf]; rfl⟩, rfl⟩
    rw [if_pos (by simpa using hmem)]
    have hx : ∃ b : ByteArray, Flock.hexOf (Flock.Sha512.hash b) = t.sha512 := ⟨a, tableFile_hash hf⟩
    have hpf : pinnedFile t.sha512 = some hx.choose := by
      unfold pinnedFile
      rw [dif_pos hx]
    rw [hpf, ht t htm a hx.choose hf hpf]

/-! ## v2's statement check -/

/-- **v2's statement check at the session's inputs `I`, public file `pubB` and registered reads `rd`**: `verify
--zk`'s `ProvedScope.check` on the circuit v2's checks read, then its template rule, the `scope` v2's run hands each
session's verdict (`Spot.OneStageZk.run`). -/
def scopeOf (I : Flock.Witness.Inputs) (pubB : ByteArray) (rd : _root_.Spot.OneStageZk.Reads) : Flock.V Unit := do
  Flock.ProvedScope.checkOf
    (Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables) (Flock.Witness.zkTags I) true pubB
    I.count false rd.registered.isSome
  _root_.Spot.OneStageZk.templateOk
    (← Flock.ProvedScope.circuitOf (Flock.Witness.zkTags I) I.circuitFile I.tables)

/-- A session's verdict accepts only where its statement check (`scope`) passes. -/
theorem verdictOf_scope {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.Spot.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.Spot.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : scope = .ok () := by
  unfold _root_.Spot.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨⟩, hs, -⟩ := ebind_ok h
  exact hs

/-- **One session's `step` after a run that accepted so far, where every session accepted, passes its statement
check** (`verdictOf_scope`). -/
theorem step_scope {I : Flock.Witness.Inputs} {rd : _root_.Spot.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.Spot.OneStageZk.RunFiles} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) : scope = .ok () := by
  unfold _root_.Spot.OneStageZk.step at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · obtain ⟨_, -, h⟩ := ebind_ok h
    exact verdictOf_scope (push_ok h hall)
  · cases h

/-- **v2's run on one session hands its verdict the statement check `scopeOf`** at the session's inputs, public file
and registered reads, as `run_event` with the check named. -/
theorem run_scope {p : _root_.Spot.OneStageZk.Own} {s : _root_.Spot.OneStageZk.RunFiles}
    {pu : Option (Array ℕ)} {bs : _root_.Core.Blobs} {r : _root_.Spot.OneStageZk.Run}
    (h : _root_.Spot.OneStageZk.run p #[s] pu bs = .ok r) :
    ∃ tags0 circuitB pubB rd, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.circuit "circuit" = .ok circuitB ∧ bs.get p.pub "public file" = .ok pubB ∧
      ∃ ksFiles population programSha want,
        _root_.Spot.OneStageZk.step (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) rd bs
          (scopeOf (_root_.Spot.OneStageZk.inputsOf tags0 p circuitB pubB bs) pubB rd) ksFiles population
          programSha want ⟨#[], #[], #[], true⟩ s = .ok r := by
  unfold _root_.Spot.OneStageZk.run at h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  split at h
  · rename_i tags0 htags
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨circuitB, hc, h⟩ := ebind_ok h
    obtain ⟨pubB, hp, h⟩ := ebind_ok h
    obtain ⟨_, -, h⟩ := ebind_ok h
    obtain ⟨rd, -, h⟩ := ebind_ok h
    obtain ⟨r', hr', h⟩ := ebind_ok h
    have e : r' = r := by
      cases pu with
      | none => simpa only [pure, Except.pure, Except.ok.injEq] using h
      | some want =>
        repeat' (first | (obtain ⟨_, _, h⟩ := ebind_ok h) | (split at h))
        all_goals simpa only [pure, Except.pure, Except.ok.injEq] using h
    subst e
    exact ⟨tags0, circuitB, pubB, rd, htags, hc, hp, _, _, _, _, foldlM_single hr'⟩
  · cases h

/-- **Where v2's check accepts at the coins `ξ`, its session passes `verify --zk`'s statement check**
(`ProvedScope.check`) at the statement's tags under `--zk` (`zkTags`), on the circuit and public files the parameters
name, with no public input words of the verifier's own. -/
theorem checkBlobsAt_scope {p : _root_.Spot.OneStageZk.Params} {c : _root_.Spot.OneStage.Commitment}
    {ξ : ByteArray} {e : _root_.Spot.OneStageZk.Evidence} {bs : _root_.Core.Blobs}
    {pub : _root_.Spot.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.Spot.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ tags0 circuitB pubB reg, Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      bs.get p.own.circuit "circuit" = .ok circuitB ∧ bs.get p.own.pub "public file" = .ok pubB ∧
      Flock.ProvedScope.check
        (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p.own circuitB pubB bs)) true circuitB pubB
        (_root_.Spot.OneStageZk.tablesOf bs) 1 false reg = .ok () := by
  obtain ⟨r, -, hsz, -, -, hrun, hall⟩ := checkBlobsAt_run h
  rw [eq_single hsz] at hrun
  obtain ⟨tags0, circuitB, pubB, rd, ht, hc, hp, _, _, _, _, hstep⟩ := run_scope hrun
  have hs := step_scope hstep hall
  unfold scopeOf at hs
  obtain ⟨⟨⟩, hck, -⟩ := ebind_ok hs
  exact ⟨tags0, circuitB, pubB, rd.registered.isSome, ht, hc, hp, hck⟩

/-! ## The session's descriptor -/

/-- **Where `verify --zk`'s ports check passes at a flat untyped statement, the circuit file reads as a descriptor whose
unit is `Built`** and the public file loads on its circuit: the check parses the circuit (`HmOut.parse`, the one
`ProvedScope.circuitOf` reads there, `circuitOf_parse`) and loads the public file before it checks anything else
(`PortRoots.checkWith`), and `Desc.parseStmt` of an untemplated statement is `Desc.parse` (`built_of_parseStmt`). -/
theorem desc_of_ports {I : FlockSoundness.Discharge.Exec.Inputs}
    (hflat : (FlockSoundness.Discharge.ZkExec.zkTags I).hiddenOutputs = true ∧
      (FlockSoundness.Discharge.ZkExec.zkTags I).isTmpl = false)
    (ht : (FlockSoundness.Discharge.ZkExec.zkTags I).typed = false)
    (h : Flock.PortRoots.checkOf
      (Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables)
      (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok ()) :
    ∃ D pubO, Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D ∧
      FlockSoundness.Built D.unroll.unit ∧
      Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO ∧
      Flock.ProvedScope.circuitOf (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D.unroll := by
  rw [circuitOf_parse hflat.1 hflat.2] at h ⊢
  unfold Flock.PortRoots.checkOf Flock.PortRoots.checkWith at h
  split at h
  · cases h
  rename_i c hp
  split at h
  · cases h
  split at h
  · cases h
  split at h
  · cases h
  rename_i pubO hl
  have hD : Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok (.single c) := by
    unfold Flock.Desc.parse
    rw [hp]
  refine ⟨.single c, pubO, hD, ?_, hl, hp⟩
  refine FlockSoundness.Discharge.Hidden.built_of_parseStmt (cf := I.circuitFile) (tables := I.tables) (D := .single c) ?_ ht
  unfold Flock.Desc.parseStmt
  rw [if_neg (by simp [hflat.2])]
  exact hD

/-- **Where `verify --zk`'s form check passes at an untemplated statement, the circuit it parses is in
`Layout.scopeOkH`**: `KsScope.unitWhy` refuses any other (`FailClosed.scopeOkH_eq`). -/
theorem scopeH_of_form {t : Flock.Tags} {cf pf : ByteArray} {tables : List (String × ByteArray)} {count : ℕ}
    {draw : Option Json} {c : Flock.Circuit} (hc : Flock.ProvedScope.circuitOf t cf tables = .ok c)
    (hi : t.isTmpl = false) (h : Flock.KsScope.checkOf (Flock.ProvedScope.circuitOf t cf tables) t pf count draw = .ok ()) :
    FlockSoundness.Discharge.Layout.scopeOkH c = true := by
  rw [hc] at h
  unfold Flock.KsScope.checkOf at h
  cases htw : Flock.KsScope.tagsWhy t with
  | some _ => simp [htw] at h
  | none =>
    cases hl : Flock.HmOut.loadPublic t c pf with
    | error _ => simp [htw, hl] at h
    | ok pub =>
      cases hfw : Flock.KsScope.filesWhy t count c pub pf with
      | some _ => simp [htw, hl, hfw] at h
      | none =>
        unfold Flock.KsScope.filesWhy at hfw
        cases hu : Flock.KsScope.unitWhy t count c with
        | some _ => simp [hu] at hfw
        | none =>
          unfold Flock.KsScope.unitWhy at hu
          rw [← FlockSoundness.Discharge.FailClosed.scopeOkH_eq]
          cases hs : Flock.ProvedScope.scopeOkH c <;> simp [hi, hs] at hu ⊢

/-! ## An accepted play's session -/

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

open FlockSoundness.Discharge.Recursion.Witness (toExec) in
/-- **What v2's acceptance of a play gives of its session**: where the sampled verdict accepts a play on the committed
files `F` at the parameters `θ`, v2's parameters `p` in `θ` set its session up at the statement's tags on `F`'s
circuit and public files (`checkBlobsAt_event`, `get_named`), whose lookup tables are the ones the circuit file
registers where the blobs hold each at its registered file (`pinnedTables_eq`), and a descriptor of the verifier's
program reads the session's sampling design off `designAt` (`lawAt_designAt`). -/
theorem accepted_parts {V : Vocab d α} {θ : ByteArray} (hD : DigestsAt V θ) {k : Verity.Registration.Composite}
    {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange} (hP : ∃ p, ParamsAt θ p)
    (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o) (ht : TablesSent o.sent) :
    ∃ p tags0 pubB record proofs n desc, ParamsAt θ p ∧
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some tags0 ∧
      pubB = V.publicFile θ F ∧
      Accepted (toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own (V.circuitFile θ F) pubB
        (_root_.Core.Blobs.ofArray o.sent.blobs))) { record := record, proofs := proofs } p.base.verifier n ∧
      _root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.sent.blobs) = pinnedTables (V.circuitFile θ F) ∧
      _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program ∧
      lawAt p.base.verifier n = .ok (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law
        p.base.verifier.workTable n) ∧
      ∃ reg, Flock.ProvedScope.check (Flock.Witness.zkTags (_root_.Spot.OneStageZk.inputsOf tags0 p.own
          (V.circuitFile θ F) pubB (_root_.Core.Blobs.ofArray o.sent.blobs))) true (V.circuitFile θ F) pubB
        (_root_.Spot.OneStageZk.tablesOf (_root_.Core.Blobs.ofArray o.sent.blobs)) 1 false reg = .ok () := by
  obtain ⟨hθ, hFo, -⟩ := Binding.play_reaches hreach
  have hreg : V.Registered θ F := by
    have h1 := hv.1
    rw [hθ, hFo] at h1
    exact h1
  obtain ⟨p₀, hp₀⟩ := hP
  have hp₀' : ParamsAt o.sent.params p₀ := by
    rw [show o.sent.params = θ from hθ]
    exact hp₀
  obtain ⟨a, b, e, p, cm, ev, ξ, dr, hcase, hacc⟩ := case_drew hv.2.1 hp₀' (verdict_case hv)
  have hpp : p₀ = p := paramsAt_case hv.2.1 hcase hacc hp₀'
  subst hpp
  obtain ⟨-, -, -, -, -, pub, prof, hck⟩ := hacc
  obtain ⟨tags0, circuitB, pubB, record, proofs, n, ht0, hc, hpb, -, -, hn, hAc⟩ := checkBlobsAt_event hck
  have hs : o.sent.blobs = F ++ o.blobs := by
    show o.files ++ o.blobs = _
    rw [hFo]
  have hcf : circuitB = V.circuitFile θ F := by
    rw [hs] at hc
    exact get_named (hD.circuit p₀ hp₀) hreg.1 hc
  have hpf : pubB = V.publicFile θ F := by
    rw [hs] at hpb
    exact get_named (hD.publicFile p₀ hp₀) hreg.2 hpb
  obtain ⟨circuitB', hg, hl⟩ := tables_registered hp₀' hv
  have e0 : circuitB' = circuitB := Except.ok.inj (hg.symm.trans hc)
  rw [e0] at hl
  have htab := (pinnedTables_eq hl ht).symm
  obtain ⟨desc, hdesc⟩ := program_of_paramsAt hp₀
  obtain ⟨out, pub', prof', hat⟩ := checkBlobsAt_checkAt hck
  obtain ⟨lo, n', read, hlo, hn', hdf⟩ := checkAt_drew hat
  have hnn : n' = n := Option.some.inj (hn'.symm.trans hn)
  rw [hnn] at hdf
  obtain ⟨tags1, circuitB1, pubB1, reg, ht1, hc1, hp1, hsc⟩ := checkBlobsAt_scope hck
  have e1 : tags1 = tags0 := Option.some.inj (ht1.symm.trans ht0)
  have e2 : circuitB1 = circuitB := Except.ok.inj (hc1.symm.trans hc)
  have e3 : pubB1 = pubB := Except.ok.inj (hp1.symm.trans hpb)
  rw [e1, e2, e3] at hsc
  subst hcf
  exact ⟨p₀, tags0, pubB, record, proofs, n, desc, hp₀, ht0, hpf, hAc, htab, hdesc,
    lawAt_designAt hdesc hlo hdf, reg, hsc⟩

open FlockSoundness.Discharge.Recursion.Witness (toExec) in
/-- **The session's inputs, descriptor, public file and population on an accepted play** (`accepted_parts`): the inputs
v2 hands `flock-verify` on the committed files, whose tags are untyped and hidden-output with retained rounds and
`hm96-sha512/v1` leaves (`FailClosed.check_ok` at v2's statement check under `--zk`), the descriptor and public file
its ports check reads (`desc_of_ports`), in `Layout.scopeOkH` (`scopeH_of_form`), with the record's population, which the program's proof blocks under the
verifier's query are in order (`popMatches_of_popOk`). -/
theorem session_parts {V : Vocab d α} {θ : ByteArray} (hD : DigestsAt V θ) {k : Verity.Registration.Composite}
    {F : Array ByteArray} {c : ByteArray} {o : Verity.Exchange} (hP : ∃ p, ParamsAt θ p)
    (hreach : Verity.Game.Reaches (V.game θ F c) o) (hv : V.Verdict k .sampled o) (ht : TablesSent o.sent) :
    ∃ (p : _root_.Spot.OneStageZk.Params) (n : ℕ) (I : FlockSoundness.Discharge.Exec.Inputs) (D : Flock.Desc)
      (pubO : Flock.Public) (desc : Claims.Msg), ParamsAt θ p ∧
      Flock.Tags.find? _root_.Spot.OneStageZk.STATEMENT = some I.tags ∧ I.count = 1 ∧ I.work = none ∧
      I.stratified = none ∧ I.closure = none ∧ I.circuitFile = V.circuitFile θ F ∧
      I.publicFile = V.publicFile θ F ∧ I.tables = pinnedTables (V.circuitFile θ F) ∧
      I.partition = some p.own.partition ∧ I.program = some p.own.program ∧
      I.tags.typed = false ∧ FlockSoundness.Discharge.ZkHidden.TagsOkZHJ I ∧
      Flock.Desc.parse (FlockSoundness.Discharge.ZkExec.zkTags I) I.circuitFile I.tables = .ok D ∧
      FlockSoundness.Built D.unroll.unit ∧
      Flock.HmOut.loadPublic (FlockSoundness.Discharge.ZkExec.zkTags I) D.unroll I.publicFile = .ok pubO ∧
      FlockSoundness.Discharge.Layout.scopeOkH D.unroll = true ∧
      Flock.PortRoots.checkOf (.ok D.unroll) (FlockSoundness.Discharge.ZkExec.zkTags I) I.publicFile = .ok () ∧
      pubO.n = n ∧ _root_.Spot.Program.parse ⟨desc.toArray⟩ = .ok p.base.verifier.program ∧
      Flock.Bridge.PopMatches desc p.base.verifier.query n ∧
      lawAt p.base.verifier n = .ok (Flock.Bridge.designAt desc p.base.verifier.query p.base.verifier.law
        p.base.verifier.workTable n) := by
  obtain ⟨p, tags0, pubB, record, proofs, n, desc, hp, ht0, hpf, hAc, htab, hdesc, hlaw, reg, hsc⟩ :=
    accepted_parts hD hP hreach hv ht
  subst hpf
  obtain ⟨⟨-, htyp, hret, hleaf, -⟩, -⟩ := FlockSoundness.Discharge.FailClosed.check_ok hsc
  obtain ⟨D, pubO, hDp, hU, hpubO, hcD⟩ := desc_of_ports hAc.flat htyp hAc.ports
  have hports := hAc.ports
  rw [hcD] at hports
  exact ⟨p, n, toExec (_root_.Spot.OneStageZk.inputsOf tags0 p.own (V.circuitFile θ F) (V.publicFile θ F)
      (_root_.Core.Blobs.ofArray o.sent.blobs)), D, pubO, desc, hp, ht0, rfl, rfl, rfl, rfl, rfl, rfl, htab, rfl, rfl,
    htyp, ⟨hAc.flat.1, hret, hleaf.resolve_right fun h => absurd h.1 (by decide)⟩, hDp, hU, hpubO,
    scopeH_of_form hcD hAc.flat.2 hAc.form, hports, hAc.loads hpubO,
    hdesc, Flock.Bridge.popMatches_of_popOk hdesc hAc.pop, hlaw⟩

/-! ## The bound -/

open FlockSoundness.Discharge.ZkLink FlockSoundness.Discharge.ZkBind FlockSoundness.Discharge.ZkHidden in
/-- **The session's planned rate is at most `ρ` wherever `2^23 ≤ ρ·(e·k)`**: every table of the plan, a set-up one or
the reference setup's refused one, has a schedule `Zk.schedule` returned, whose level 0's `logLen` is at most `23`
(`zkLogLenCap_23`), as `hr_planZHJ_23` for the plan without the verifier's own words. -/
theorem hr_planZHJP {n mPts : ℕ} {I : FlockSoundness.Discharge.Exec.Inputs} {own : Option ByteArray}
    (dj : Finset (Fin n) → Json) {S₀ : Finset (Fin n)} {d₀ : Json}
    (y₀ : DrawSetupZHJP (inputsJ I 0) own mPts S₀ d₀) {Reg : Type} {k : ℕ} {ρ : ℝ}
    (hkρ : (2 : ℝ) ^ 23 ≤ ρ * (Real.exp 1 * k)) (S : Finset (Fin n)) (R : Reg)
    (j : Fin (planZHJP (Reg := Reg) dj y₀ S R).length) :
    rateZ (planZHJP (Reg := Reg) dj y₀ S R) j k ≤ ρ := by
  unfold rateZ
  refine (div_le_div_of_nonneg_right (pow_le_pow_right₀ (by norm_num) ?_) (by positivity)).trans (cap_div_le hkρ)
  simp only [planZHJP, List.get_ofFn]
  unfold tableZHJP
  split_ifs with h
  · exact zkLogLenCap_23 _ _ _ (Classical.choice h).hz
  · exact zkLogLenCap_23 _ _ _ y₀.hz

end Verity.Sampled
