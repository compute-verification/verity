module

public import Proofs.Properties.Verity.Sampled.RecordDraw

/-!
# What v2 holds of the record's draw

The record's draw (`recordDraw_firstStop`, `MovesOf`'s clause (i)) reads v2's log at its coins (`DrawReadAll`) and
v2's hold on the record's `unit_draw` (`DrawHeld`). This module gives what v2's code gives of them.

* **v2's log at its coins** (`drawReadAll_coinsRead`): `coinsRead` is `logRead` (`coinsRead_eq_logRead`), whose read
  shows the draw wherever v2 accepts (`drawReadAll_logRead`).
* **v2 accepts only a subset draw** (`recordDraw_subset`): where v2's case accepts, the record v2 reads holds a draw,
  since `verify --zk` refuses a session without one (`verdictOf_drawn`); its law is one a soundness claim draws by at
  `--zk` (`ProvedScope.drawLaw`, `lawWhy_zk`), the subset or the stratified law; and the session v2 sets up has no
  `--stratified` input, so `Stmt.setupTables` refuses a stratified draw (`setupTables_stratified`). So `DrawHeld` reads
  the record's draw at the subset law of the units v2 drew, whatever law v2 drew them by.
-/

@[expose] public section

namespace Verity.Sampled

open Lean (Json)
open FlockSoundness.Refine (ebind_ok)

variable {ι : Type} {d : Accounting.Device ι} {α : Type} [Fintype α]

/-! ## v2's log at its coins -/

/-- **v2's log at its coins shows the draw** wherever v2 accepts: `coinsRead` is `logRead`. -/
theorem drawReadAll_coinsRead (k : Verity.Registration.Composite) : DrawReadAll k coinsRead :=
  coinsRead_eq_logRead ▸ drawReadAll_logRead k

/-! ## v2 accepts only a subset draw -/

/-- **`verify --zk` refuses a session without a draw**: where a session's verdict passes, its record's draw is some. -/
theorem verdictOf_drawn {I : Flock.Witness.Inputs} {scope ksFiles population : Flock.V Unit}
    {coinKeys : Array String} {record : String} {draw : Option Json} {st : _root_.SampledProofs.OneStageZk.SetUp}
    {files : Flock.V (Array ByteArray)} {coinKey : Option String}
    (h : _root_.SampledProofs.OneStageZk.verdictOf I scope ksFiles population coinKeys record draw st files coinKey
      = .ok ()) : draw.isSome = true := by
  unfold _root_.SampledProofs.OneStageZk.verdictOf at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨⟨_, _, _⟩, -, h⟩ := ebind_ok h
  dsimp only at h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, -, h⟩ := ebind_ok h
  obtain ⟨_, hd, -⟩ := ebind_ok h
  cases draw with
  | none => simp [Flock.ProvedScope.drawLaw] at hd
  | some _ => rfl

/-- **One session's `step` from the empty run, where every session accepted, reads a record that holds a draw.** -/
theorem step_drawn {I : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads}
    {bs : _root_.Core.Blobs} {scope ksFiles population : Flock.V Unit} {programSha : Option String} {want : Json}
    {s : _root_.SampledProofs.OneStageZk.RunFiles} {r : _root_.SampledProofs.OneStageZk.Run}
    (h : _root_.SampledProofs.OneStageZk.step I rd bs scope ksFiles population programSha want ⟨#[], #[], #[], true⟩ s
      = .ok r)
    (hall : r.all = true) :
    ∃ recB record, bs.get s.record s!"session {s.name}'s record" = .ok recB ∧
      String.fromUTF8? recB = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  unfold _root_.SampledProofs.OneStageZk.step at h
  obtain ⟨recB, hb, h⟩ := ebind_ok h
  dsimp only at h
  split at h
  · rename_i record hrec
    exact ⟨recB, record, hb, hrec, verdictOf_drawn (push_ok h hall)⟩
  · cases h

/-- **Where v2's check accepts, the session's record holds a draw.** -/
theorem checkBlobsAt_drawn {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ξ : ByteArray} {e : _root_.SampledProofs.OneStageZk.Evidence}
    {bs : _root_.Core.Blobs} {pub : _root_.SampledProofs.OneStage.Public} {prof : _root_.Core.Integrity}
    (h : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ e bs = .accept pub prof) :
    ∃ record, sessionRecord e bs = some record ∧ (Flock.Witness.recordDraw record).isSome = true := by
  obtain ⟨r, hsz, hrun, hall⟩ := checkBlobsAt_run h
  have hs := eq_single hsz
  rw [hs] at hrun
  obtain ⟨_, _, _, _, -, -, -, -, _, _, _, _, _, hstep⟩ := run_verified hrun
  obtain ⟨recB, record, hb, hrec, hd⟩ := step_drawn hstep hall
  exact ⟨record, sessionRecord_eq hs hb hrec, hd⟩

/-- **`setupAt` at a draw holds the draw's law** (`ProvedScope.drawLaw` at `--zk`). -/
theorem setupAt_drawLaw {W : Flock.Witness.Inputs} {rd : _root_.SampledProofs.OneStageZk.Reads} {dj : Json}
    {x : Flock.SessionSpec × Array Flock.Setup × Json × Json}
    (h : _root_.SampledProofs.OneStageZk.setupAt W rd (some dj) = .ok x) :
    Flock.ProvedScope.drawLaw true (Flock.Witness.zkTags W) false (some dj) = .ok () := by
  unfold _root_.SampledProofs.OneStageZk.setupAt at h
  obtain ⟨⟨_, _⟩, -, h⟩ := ebind_ok h
  try dsimp only at h
  obtain ⟨⟨⟩, hu, -⟩ := ebind_ok h
  simpa only [Option.isSome_some, ↓reduceIte] using hu

/-- **A draw `drawLaw` passes decodes**, to a draw whose law `lawWhy` doesn't refuse. -/
theorem drawLaw_ok {zk : Bool} {tags : Flock.Tags} {pt : Bool} {dj : Json}
    (h : Flock.ProvedScope.drawLaw zk tags pt (some dj) = .ok ()) :
    ∃ D, Flock.Draw.UnitDraw.ofJson dj = .ok D ∧ Flock.ProvedScope.lawWhy zk tags pt D = none := by
  simp only [Flock.ProvedScope.drawLaw] at h
  cases hD : Flock.Draw.UnitDraw.ofJson dj with
  | error e => simp [hD] at h
  | ok D =>
    simp only [hD] at h
    cases hw : Flock.ProvedScope.lawWhy zk tags pt D with
    | some why => simp [hw] at h
    | none => exact ⟨D, rfl, hw⟩

/-- **The laws a soundness claim draws by at `--zk`**, at public tables: the subset and the stratified law. -/
theorem lawWhy_zk {tags : Flock.Tags} {D : Flock.Draw.UnitDraw}
    (h : Flock.ProvedScope.lawWhy true tags false D = none) :
    (∃ kd, D.law = .subset kd) ∨ ∃ K St, D.law = .stratified K St := by
  unfold Flock.ProvedScope.lawWhy at h
  cases hl : D.law with
  | subset kd => exact .inl ⟨kd, rfl⟩
  | stratified K St => exact .inr ⟨K, St, rfl⟩
  | bernoulli a b => simp [hl] at h
  | work K ws => simp [hl] at h

/-- **A session without `--stratified` refuses a stratified draw**: `Stmt.setupTables` holds a stratified draw's law to
the verifier's own `K`, and refuses it without one. -/
theorem setupTables_stratified {tags : Flock.Tags} {cf pf : ByteArray} {coins : Flock.Coins} {count : ℕ}
    {tables : List (String × ByteArray)} {dj : Json} {partition : Option Json} {program : Option ByteArray}
    {r : Flock.SessionSpec × Array Flock.Stmt} {D : Flock.Draw.UnitDraw} {K : ℕ} {St : Array Flock.Draw.Stratum}
    (h : Flock.Stmt.setupTables tags cf pf coins count tables (some dj) partition program = .ok r)
    (hD : Flock.Draw.UnitDraw.ofJson dj = .ok D) (hl : D.law = .stratified K St) : False := by
  unfold Flock.Stmt.setupTables at h
  simp only [hD, hl, Except.map] at h
  cases partition <;> cases program <;> simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h

/-- **v2 accepts only a subset draw**: where composite `k` accepted what the prover sent and v2's case in it accepts,
the record v2 reads holds a draw (`unit_draw`), which decodes to a draw by the subset law. -/
theorem recordDraw_subset {k : Verity.Registration.Composite} {s : Verity.Sent} {a b e : ByteArray}
    {p : _root_.SampledProofs.OneStageZk.Params} {c : _root_.SampledProofs.OneStage.Commitment}
    {ev : _root_.SampledProofs.OneStageZk.Evidence} {ξ : ByteArray} {dr : _root_.Properties.Challenger.Draws}
    (hS : Verity.SameAccepts k s) (hcase : V2Case k s a b e)
    (hacc : V2Accepts (_root_.Core.Blobs.ofArray s.blobs) a b e p c ev ξ dr) {record : String}
    (hr : sentRecord s = some record) :
    ∃ dj D kd, Flock.Witness.recordDraw record = some dj ∧ Flock.Draw.UnitDraw.ofJson dj = .ok D ∧
      D.law = .subset kd := by
  obtain ⟨record', _, _, _, _, -, hr', -, hv⟩ := sent_verified hS hcase hacc
  obtain ⟨-, -, -, -, -, _, _, hck⟩ := hacc
  obtain ⟨record'', hrec'', hdrawn⟩ := checkBlobsAt_drawn hck
  obtain rfl := Option.some.inj (hr'.symm.trans hr)
  obtain ⟨_, _, _, _, _, -, -, -, -, -, hrec, hst, -, -⟩ := hv
  obtain rfl := Option.some.inj (hrec''.symm.trans hrec)
  obtain ⟨dj, hdj⟩ := Option.isSome_iff_exists.1 hdrawn
  rw [hdj] at hst
  obtain ⟨D, hD, hw⟩ := drawLaw_ok (setupAt_drawLaw hst)
  obtain ⟨_, hss, -⟩ := setupAt_ok hst
  obtain ⟨hset, -⟩ := sessionAt_ok hss
  rcases lawWhy_zk hw with ⟨kd, hk⟩ | ⟨K, St, hk⟩
  · exact ⟨dj, D, kd, hdj, hD, hk⟩
  · exact (setupTables_stratified hset hD hk).elim

end Verity.Sampled
