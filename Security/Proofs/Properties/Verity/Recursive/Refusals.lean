import Proofs.Properties.Verity.Recursive.Names
import Proofs.Properties.Verity.Recursive.BaseSetup
import Proofs.Properties.Verity.Recursive.Compiled
import Proofs.Properties.Verity.Recursive.PartStatement
import Proofs.Flock.Claims.Recursion.DeployChecked
import Proofs.Flock.Claims.Recursion.RestVerdictBasesHold
import Proofs.Flock.Soundness.Refine.Walk

/-!
# The compiled component's refusals, as `acc_of_accepts` reads them (`Refusals`)

`Verity.Recursive.Refusals` is what `acc_of_accepts` reads of the compiled component beyond its setups: Flock's own
checks on the inner statement (`InnerOkAt`) and the files the parts' statements encode short enough for their length
words. Each is the compiled component's own refusal:

* **`innerOkAt_of_compiled`.** The compiled `innerHeld` runs `innerOk`: Flock's plain setup of the inner statement on its
  circuit, public file and draw, at the verifier's partition and program and `--stratified` at the population they give
  (`HmRow.stratifiedLaw` at `K = 1`), then each check `FlockDeploy.Ins.setup` runs and the scope check. Where `x` names
  the component's inputs (`Names`), the inner statement `x` carries reads as those inputs (`ins_inner`), so the model's
  setup is the compiled one step for step (`setup_inner`), its scope check is the compiled one, it carries the draw, and
  at one table the session's draw is the draw (`RecDeploy.drawTiedS`). That is the model's `innerOk`.
* **`plain_of_compiled`**, beside `Refusals`. The parts' statement `verity/flock-circuit` has no plain leaves: each
  part's verdict refuses them, so an accepted session with a part shows it.
* **The sizes.** The compiled component refuses a staged public file, inner record, coin server's transcript or staged
  rows file of `2 ^ 56` bytes or more (`fits`), so each file the blobs show is under `2 ^ 64` bytes (`record_of_compiled`,
  `transcript_of_compiled`, `rows_of_compiled`). `Names` pins every field `x` encodes to a file the component read or the
  canonical bytes of a value it read, so `x` is no longer than the compiled `statementSize` (`x_len_le`), which the
  component refuses at `2 ^ 56` bytes or more (`statementFits`): `pub_of_compiled` gives `x` with each part's public file
  under `2 ^ 64` bytes at every part `nS x` counts, with no hypothesis on `x`.
-/

namespace Verity.Recursive.RefusalsOf

open Lean (Json)
open Flock (V)
open BaseSetup (bind_ok noIdentifier_of_check forIn_mem body_ok body_yield)
open FlockSoundness.Refine (throw_ne_ok)

theorem mapError_ok {ε A : Type} {x : Except ε A} {f : ε → ε} {a : A} (h : x.mapError f = .ok a) : x = .ok a := by
  cases x with
  | error e => cases h
  | ok a' => cases h; rfl

theorem ok_bind {ε α β : Type} {x : Except ε α} {f : α → Except ε β} {a : α} {b : β} (hx : x = .ok a)
    (hf : f a = .ok b) : (x >>= f) = .ok b := by
  rw [hx]
  exact hf

/-- A blob's bytes don't depend on the name a refusal would give it. -/
theorem get_what {bs : Core.Blobs} {d w w' : String} {b : ByteArray} (h : bs.get d w = .ok b) :
    bs.get d w' = .ok b := by
  unfold Core.Blobs.get at h ⊢
  obtain ⟨a, hd, h⟩ := bind_ok h
  refine ok_bind hd ?_
  try dsimp only at h ⊢
  by_cases hc : (a.size != 64 || Flock.hexOf a != d) = true
  · rw [if_pos hc] at h
    first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
  · rw [if_neg hc] at h ⊢
    cases hf : bs.entries.find? (·.1 == a) with
    | none =>
      rw [hf] at h
      first | cases h | simp [throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    | some e =>
      rw [hf] at h
      obtain ⟨e1, e2⟩ := e
      first | exact h | (simp only at h ⊢; exact h)

/-- A loop each of whose steps passes and goes on passes. -/
theorem forIn_ok_of_all {α : Type} {xs : Array α} {b : α → PUnit → V (ForInStep PUnit)}
    (hb : ∀ a ∈ xs, ∀ u, b a u = .ok (.yield PUnit.unit)) : forIn xs PUnit.unit b = .ok PUnit.unit := by
  rcases xs with ⟨l⟩
  simp only [List.forIn_toArray]
  induction l with
  | nil => rfl
  | cons c l ih =>
    simp only [List.forIn_cons, bind, Except.bind]
    rw [hb c (by simp) PUnit.unit]
    exact ih fun a ha u => hb a (by simp_all) u

/-- `fits` passes only on a file under `2 ^ 56` bytes. -/
theorem fits_lt {b : ByteArray} {w : String} (h : VStar.Recursive.fits b w = .ok ()) : b.size < 2 ^ 56 := by
  unfold VStar.Recursive.fits VStar.Recursive.need at h
  by_cases hc : b.size < 2 ^ 56
  · exact hc
  · have hd : decide (b.size < VStar.Recursive.FIELD_MAX) = false :=
      decide_eq_false (by simpa [VStar.Recursive.FIELD_MAX] using hc)
    simp only [hd, Bool.false_eq_true, ↓reduceIte] at h
    exact absurd h (throw_ne_ok _ _)

/-! ## What `innerHeld` reads -/

/-- **What `innerHeld` read and refused**: the inner circuit, public file and draw blobs, on which the compiled
`innerOk` passes, and the inner record, the coin server's transcript and the staged rows, each under `2 ^ 56` bytes. -/
theorem innerHeld_reads {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} (hh : VStar.Recursive.innerHeld o pr bs = .ok h) :
    ∃ cB pB dB recB trB rowsB,
      bs.get o.circuit "the inner circuit" = .ok cB ∧ bs.get o.pub "the inner public file" = .ok pB ∧
      bs.get o.draw "the inner statement's draw" = .ok dB ∧ VStar.Recursive.innerOk o cB pB dB = .ok () ∧
      bs.get pr.record "the inner session's record" = .ok recB ∧ recB.size < 2 ^ 56 ∧
      bs.get pr.transcript "rec inner transcript" = .ok trB ∧ trB.size < 2 ^ 56 ∧
      bs.get pr.rows "the staged rows" = .ok rowsB ∧ rowsB.size < 2 ^ 56 := by
  unfold VStar.Recursive.innerHeld at hh
  obtain ⟨cB, hc, hh⟩ := bind_ok hh
  obtain ⟨pB, hp, hh⟩ := bind_ok hh
  obtain ⟨_, -, hh⟩ := bind_ok hh
  obtain ⟨recB, hr, hh⟩ := bind_ok hh
  obtain ⟨⟨⟩, hfr, hh⟩ := bind_ok hh
  cases hru : String.fromUTF8? recB with
  | none => (simp only [hru] at hh) <;> exact absurd hh (throw_ne_ok _ _)
  | some record =>
    simp only [hru] at hh
    obtain ⟨dB, hd, hh⟩ := bind_ok hh
    obtain ⟨⟨⟩, hok, hh⟩ := bind_ok hh
    obtain ⟨trB, htr, hh⟩ := bind_ok hh
    obtain ⟨⟨⟩, hftr, hh⟩ := bind_ok hh
    cases htu : String.fromUTF8? trB with
    | none => (simp only [htu] at hh) <;> exact absurd hh (throw_ne_ok _ _)
    | some t =>
      simp only [htu] at hh
      obtain ⟨_, -, hh⟩ := bind_ok hh
      obtain ⟨_, -, hh⟩ := bind_ok hh
      cases hjp : Json.parse t with
      | error err =>
        simp only [hjp] at hh
        obtain ⟨_, hl, -⟩ := bind_ok hh
        exact absurd hl (throw_ne_ok _ _)
      | ok jt =>
        simp only [hjp] at hh
        obtain ⟨_, -, hh⟩ := bind_ok hh
        obtain ⟨rowsB, hrw, hh⟩ := bind_ok hh
        obtain ⟨⟨⟩, hfrw, -⟩ := bind_ok hh
        exact ⟨cB, pB, dB, recB, trB, rowsB, hc, hp, hd, mapError_ok hok, hr, fits_lt hfr, htr, fits_lt hftr, hrw,
          fits_lt hfrw⟩

/-- **The compiled `innerOk`, step by step**: the draw is UTF-8 JSON `j`, the verifier's partition and program give the
population `n`, and the inner statement's tags (`verity/flock-circuit+plain-leaves` on the os coins) pass each call
`FlockDeploy.Ins.setup` makes at the setup `(spec, sts)` of the inner circuit and public file at the draw, partition,
program and `--stratified n`, and the scope check. -/
theorem innerOk_peel {o : VStar.Recursive.Own} {cB pB dB : ByteArray}
    (h : VStar.Recursive.innerOk o cB pB dB = .ok ()) :
    ∃ t j L n spec sts, String.fromUTF8? dB = some t ∧ Json.parse t = .ok j ∧
      Flock.HmRow.stratifiedLaw o.partition o.program 1 = .ok (L, n) ∧
      (Flock.Tags.circuitPlainLeaves.withCoins .os).hm96Rows = true ∧
      Flock.RecSetup.checkInner none none (pure (none, none)) = .ok () ∧
      Flock.Stmt.setupTables (Flock.Tags.circuitPlainLeaves.withCoins .os) cB pB .os 1 [] (some (Flock.norm j))
        (some o.partition) (some o.program) none (some n) none none = .ok (spec, sts) ∧
      (∀ st ∈ sts, Flock.checkRegistered st none = .ok () ∧ SampledProofs.OneStageZk.checkIdentifier st = .ok ()) ∧
      Flock.Registered.checkFull none (some (Flock.norm j)) = .ok () ∧
      Flock.RecSetup.checks none none Flock.Tags.circuitPlainLeaves [] sts cB (some o.partition) (some o.program)
        none none none none = .ok () ∧
      sts.forM (Flock.RecCoins.check · none (Flock.RecCoins.Source.atC3 none none none) (Flock.RecSetup.c3 none none))
        = .ok () ∧
      sts.forM (Flock.RecVStar.checkPubRows · none none none) = .ok () ∧
      Flock.ProvedScope.drawLaw false (Flock.Tags.circuitPlainLeaves.withCoins .os) false (some (Flock.norm j)) =
        .ok () ∧
      Flock.ProvedScope.check (Flock.Tags.circuitPlainLeaves.withCoins .os) false cB pB [] 1 false false = .ok () := by
  have htg : Flock.Tags.find? VStar.Recursive.INNER_STATEMENT = some Flock.Tags.circuitPlainLeaves := rfl
  unfold VStar.Recursive.innerOk at h
  simp only [htg] at h
  cases ht : String.fromUTF8? dB with
  | none => (simp only [ht] at h) <;> exact absurd h (throw_ne_ok _ _)
  | some t =>
    simp only [ht] at h
    obtain ⟨j, hj, h⟩ := bind_ok h
    obtain ⟨⟨L, n⟩, hL, h⟩ := bind_ok h
    have hrows : (Flock.Tags.circuitPlainLeaves.withCoins .os).hm96Rows = true := by
      refine Classical.byContradiction fun hne => ?_
      simp [hne, throw, throwThe, MonadExceptOf.throw, bind, Except.bind] at h
    simp only [hrows, Bool.not_true, Bool.false_eq_true, ↓reduceIte, pure_bind] at h
    obtain ⟨⟨⟩, hci, h⟩ := bind_ok h
    obtain ⟨⟨spec, sts⟩, hsu, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hloop, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hfull, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hchecks, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hcoins, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hpub, h⟩ := bind_ok h
    obtain ⟨⟨⟩, hlaw, h⟩ := bind_ok h
    exact ⟨t, j, L, n, spec, sts, (by first | exact ht | rfl), mapError_ok hj, hL, hrows, hci, hsu,
      fun st hst => (body_ok (A := fun st => Flock.checkRegistered st none)
        (B := SampledProofs.OneStageZk.checkIdentifier)).1 (forIn_mem body_yield hloop st hst),
      hfull, hchecks, hcoins, hpub, hlaw, h⟩

/-! ## The model's reading of the inner statement -/

/-- **The model's reading of the inner statement** the compiled `innerOk` sets up: `verity/flock-circuit+plain-leaves`
on the os coins, the circuit and public file, one table and no lookup tables, the verifier's partition and program, the
draw `j` rebuilt, `--stratified n`, and none of `verify`'s other inputs. -/
def innerIns (cB pB : ByteArray) (o : VStar.Recursive.Own) (j : Json) (n : ℕ) : FlockDeploy.Ins where
  stmtTags := Flock.Tags.circuitPlainLeaves
  registered := none
  read :=
    { I := { tags := Flock.Tags.circuitPlainLeaves.withCoins .os, circuitFile := cB, publicFile := pB, coins := .os,
             count := 1, tables := [], partition := some o.partition, program := some o.program, work := none,
             stratified := some n, closure := none }
      zk := false, draw := some (Flock.norm j), publicInputs := none, registered := false }

/-- **The model reads the inner statement as `innerIns`** where its fields are the ones `InnerNames` pins, its draw the
draw blob and its `--stratified` the population. -/
theorem ins_inner {ri : FlockDeploy.Raw} {o : VStar.Recursive.Own} {dB : ByteArray} {t : String} {j : Json} {n : ℕ}
    (hs : ri.statement = VStar.Recursive.INNER_STATEMENT) (hz : ri.zk = false) (ho : ri.osCoins = true)
    (hc : ri.count = 1) (htb : ri.tables = [])
    (hpart : ri.partition.map (FlockDeploy.jsonOf "partition") = some (.ok o.partition))
    (hprog : ri.program = some o.program) (hd : ri.draw = some dB) (hwk : ri.work = none) (hwt : ri.workTable = none)
    (hst : ri.stratified = some n) (hcl : ri.closure = none) (hreg : ri.registered = none)
    (hpi : ri.publicInputs = none) (ht : String.fromUTF8? dB = some t) (hj : Json.parse t = .ok j) :
    ri.ins = .ok (innerIns ri.circuit ri.publicFile o j n) := by
  have hf : Flock.Tags.byName? ri.statement = some Flock.Tags.circuitPlainLeaves :=
    Flock.Tags.byName?_of_find? (by rw [hs]; rfl)
  have hdm : ri.draw.mapM (FlockDeploy.jsonOf "draw") = .ok (some j) := by
    rw [hd]
    simp [Option.mapM_some, FlockDeploy.jsonOf, ht, hj, Functor.map, Except.map, Except.mapError]
  have hpm : ri.partition.mapM (FlockDeploy.jsonOf "partition") = .ok (some o.partition) := mapM_of_map hpart
  unfold FlockDeploy.Raw.ins
  simp only [hf, hz, ho, hc, htb, hprog, hst, hpi, hwk, hdm, hpm, hwt, hcl, hreg, Option.mapM_none, Bool.false_and,
    Bool.false_eq_true, eq_self_iff_true, ↓reduceIte, bind, Except.bind, pure, Except.pure]
  rfl

/-- **The model's setup of `innerIns` is the compiled `innerOk`'s**, from `innerOk_peel`'s steps: the hm96-sha512 rows
check, the inner checks, the session, each table's registered reads and identifier (`noIdentifier_of_check`), the full
registration, V\*'s checks, check 5, the public rows and the draw's law, with no `--zk` shape check. -/
theorem setup_inner {cB pB : ByteArray} {o : VStar.Recursive.Own} {j : Json} {n : ℕ} {spec : Flock.SessionSpec}
    {sts : Array Flock.Stmt}
    (hhm : (Flock.Tags.circuitPlainLeaves.withCoins .os).hm96Rows = true)
    (hci : Flock.RecSetup.checkInner none none (pure (none, none)) = .ok ())
    (hsu : Flock.Stmt.setupTables (Flock.Tags.circuitPlainLeaves.withCoins .os) cB pB .os 1 [] (some (Flock.norm j))
      (some o.partition) (some o.program) none (some n) none none = .ok (spec, sts))
    (hregs : ∀ st ∈ sts, Flock.checkRegistered st none = .ok () ∧ SampledProofs.OneStageZk.checkIdentifier st = .ok ())
    (hfull : Flock.Registered.checkFull none (some (Flock.norm j)) = .ok ())
    (hchecks : Flock.RecSetup.checks none none Flock.Tags.circuitPlainLeaves [] sts cB (some o.partition)
      (some o.program) none none none none = .ok ())
    (hcoins : sts.forM (Flock.RecCoins.check · none (Flock.RecCoins.Source.atC3 none none none)
      (Flock.RecSetup.c3 none none)) = .ok ())
    (hpub : sts.forM (Flock.RecVStar.checkPubRows · none none none) = .ok ())
    (hlaw : Flock.ProvedScope.drawLaw false (Flock.Tags.circuitPlainLeaves.withCoins .os) false
      (some (Flock.norm j)) = .ok ()) :
    (innerIns cB pB o j n).setup = .ok (spec, sts) := by
  have hsh : (innerIns cB pB o j n).read.tags.hm96Rows = true := hhm
  have hset : (innerIns cB pB o j n).read.setup = .ok (spec, sts) := hsu
  have hreg : (innerIns cB pB o j n).registered = none := rfl
  have hzk : (innerIns cB pB o j n).read.zk = false := rfl
  have htg : (innerIns cB pB o j n).read.tags = Flock.Tags.circuitPlainLeaves.withCoins .os := rfl
  have hdr : (innerIns cB pB o j n).read.draw = some (Flock.norm j) := rfl
  have hds : (innerIns cB pB o j n).read.draw.isSome = true := rfl
  have hst : (innerIns cB pB o j n).stmtTags = Flock.Tags.circuitPlainLeaves := rfl
  have htb : (innerIns cB pB o j n).read.I.tables = [] := rfl
  have hcf : (innerIns cB pB o j n).read.I.circuitFile = cB := rfl
  have hpa : (innerIns cB pB o j n).read.I.partition = some o.partition := rfl
  have hpr : (innerIns cB pB o j n).read.I.program = some o.program := rfl
  have hci' : Flock.RecSetup.checkInner none none (Except.ok (none, none)) = .ok () := hci
  unfold FlockDeploy.Ins.setup
  simp only [hsh, Bool.not_true, Bool.false_eq_true, ↓reduceIte, hci, hset, bind, Except.bind, pure, Except.pure]
  rw [forIn_ok_of_all fun a ha u => by simp only [hreg, (hregs a ha).1, noIdentifier_of_check (hregs a ha).2]]
  simp only [hds, hreg, hdr, hfull, hst, htb, hcf, hpa, hpr, hchecks, hzk, Bool.false_eq_true, eq_self_iff_true,
    ↓reduceIte, hcoins, hpub, Option.isSome_some, htg, hlaw]
  all_goals simp only [hci']

/-! ## `InnerOkAt` -/

/-- **Flock's own checks pass on the inner statement where the compiled component holds the inner session**: where
`ri` is the inner statement `w` carries and `InnerNames` ties it to the component's inputs, the model's `innerOk`
passes on it. -/
theorem innerOk_of_held {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs} {bs : Core.Blobs}
    {h : VStar.Recursive.InnerHeld} {w : RecDeploy.RecX} {ri : FlockDeploy.Raw}
    (hh : VStar.Recursive.innerHeld o pr bs = .ok h) (hi : FlockDeploy.decode w.inner = some ri)
    (hI : InnerNames o pr bs w ri) : RecDeploy.innerOk w.inner = .ok () := by
  obtain ⟨cB, pB, dB, _, _, _, hc, hp, hd, hok, -⟩ := innerHeld_reads hh
  rw [hI.circuit] at hc
  rw [hI.pub] at hp
  obtain rfl : ri.circuit = cB := Except.ok.inj hc
  obtain rfl : ri.publicFile = pB := Except.ok.inj hp
  obtain ⟨t, j, L, n, spec, sts, ht, hj, hL, hhm, hci, hsu, hregs, hfull, hchecks, hcoins, hpub, hlaw, hcheck⟩ :=
    innerOk_peel hok
  have hdr : ri.draw = some dB := by simp [hI.draw, hd, Except.toOption]
  have hst : ri.stratified = some n := by simp [hI.stratified, hL, Except.toOption]
  have hins := ins_inner hI.statement hI.plain hI.os hI.count hI.tables hI.partition hI.program hdr hI.work
    hI.workTable hst hI.closure hI.registered hI.publicInputs ht hj
  have hset := setup_inner (cB := ri.circuit) (pB := ri.publicFile) hhm hci hsu hregs hfull hchecks hcoins hpub hlaw
  have hck : (innerIns ri.circuit ri.publicFile o j n).read.check = .ok () := hcheck
  have hso : FlockDeploy.setupOf false w.inner = some (innerIns ri.circuit ri.publicFile o j n, spec, sts) := by
    simp [FlockDeploy.setupOf, FlockDeploy.insOf, hi, hI.plain, hins, hset, Except.toOption]
  have hds : (innerIns ri.circuit ri.publicFile o j n).read.draw.isSome = true := rfl
  exact RecDeploy.innerOk_iff.2 ⟨_, spec, sts, hso, hck, hds,
    RecDeploy.drawTiedB_iff.2 fun _ => RecDeploy.drawTiedS _ _ _ _ hso hck hds⟩

/-- **`Refusals.innerOk`: where the compiled component accepts and `x` names its inputs, Flock's own checks pass on
the inner statement** (`InnerOkAt`). -/
theorem innerOkAt_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    {x : Claims.Msg} (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) (hn : Names p e bs x) :
    InnerOkAt x := by
  intro w' hw'
  obtain ⟨h, _, hh, -⟩ := parts_of_accepts hacc
  obtain ⟨w, ri, hxw, hi, hI, -, -⟩ := hn
  rw [hxw] at hw'
  cases hw'
  exact innerOk_of_held hh hi hI

/-! ## Plain leaves -/

/-- **The parts' statement has no plain SHA-512 leaves** where the compiled component accepts a
session with a part: that part's verdict passed `need (!tags0.plainLeaves)`. `plainLeaves` compares `Json` (a
`partial` `BEq`), so no computation settles it. -/
theorem plain_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) (h0 : 0 < e.proofs.parts.size) :
    ∀ t, Flock.Tags.find? VStar.Recursive.STATEMENT = some t → t.plainLeaves = false := by
  intro t ht
  have htg : Flock.Tags.find? VStar.Recursive.STATEMENT = some Flock.Tags.circuit := rfl
  rw [htg] at ht
  cases ht
  obtain ⟨h, ss, -, -, -, hsz1, hsz2, hparts, -⟩ := parts_of_accepts hacc
  have hv := hparts 0 (by omega) h0
  unfold VStar.Recursive.partVerdict at hv
  simp only [htg] at hv
  obtain ⟨⟨⟩, hneed, -⟩ := bind_ok hv
  cases hp : Flock.Tags.circuit.plainLeaves with
  | false => rfl
  | true =>
    unfold VStar.Recursive.need at hneed
    simp only [hp, Bool.not_true, Bool.false_eq_true, ↓reduceIte] at hneed
    exact absurd hneed (throw_ne_ok _ _)

/-! ## The sizes -/

/-- **A part's public file is under `2 ^ 56` bytes** where its verdict passes the compiled `fits`. -/
theorem partVerdict_pub {h : VStar.Recursive.InnerHeld} {j : ℕ} {po : VStar.Recursive.PartOwn}
    {pp : VStar.Recursive.PartProof} {bs : Core.Blobs} {v : V Unit}
    (hv : VStar.Recursive.partVerdict h j po pp bs = .ok v) :
    ∃ pubB, bs.get pp.pub s!"part {j}'s public file" = .ok pubB ∧ pubB.size < 2 ^ 56 := by
  have htg : Flock.Tags.find? VStar.Recursive.STATEMENT = some Flock.Tags.circuit := rfl
  unfold VStar.Recursive.partVerdict at hv
  simp only [htg] at hv
  obtain ⟨⟨⟩, -, hv⟩ := bind_ok hv
  obtain ⟨_, -, hv⟩ := bind_ok hv
  obtain ⟨pubB, hp, hv⟩ := bind_ok hv
  obtain ⟨⟨⟩, hf, -⟩ := bind_ok hv
  exact ⟨pubB, hp, fits_lt hf⟩

/-- `nS x` counts no more parts than `x` carries. -/
theorem nS_le {x : Claims.Msg} {w : RecDeploy.RecX} (hx : RecDeploy.decodeX x = some w) :
    RecDeploy.nS x ≤ w.parts.length := by
  unfold RecDeploy.nS
  split
  · next r hr =>
    have hp := RecDeploy.readXPre_of_readX hr
    unfold RecDeploy.readXPre at hp
    simp only [Option.bind_eq_bind, Option.bind_eq_some_iff] at hp
    obtain ⟨w', hw', ri, -, sh, -, pa, -, dims, -, vs, -, hp⟩ := hp
    split_ifs at hp with hl
    · simp at hp
    have h' : (⟨w', ri, pa, sh, dims⟩ : RecDeploy.XR) = r := by simpa using hp
    subst h'
    rw [hx] at hw'
    cases hw'
    exact le_refl _
  · exact Nat.zero_le _

/-! ## The statement's length -/

/-- An optional file's size, `0` when absent. -/
def osz (o : Option ByteArray) : ℕ := (o.map ByteArray.size).getD 0

theorem len_encOptB (o : Option ByteArray) : (FlockDeploy.encOpt FlockDeploy.encBytes o).length ≤ 9 + osz o := by
  cases o with
  | none => simp only [FlockDeploy.encOpt, List.length_cons, List.length_nil] <;> omega
  | some b =>
    have : osz (some b) = b.size := rfl
    simp only [FlockDeploy.encOpt, List.length_cons, Part.len_encBytes] <;> omega

theorem len_encOptN (o : Option ℕ) : (FlockDeploy.encOpt FlockDeploy.encNat o).length ≤ 9 := by
  cases o with
  | none => simp only [FlockDeploy.encOpt, List.length_cons, List.length_nil] <;> omega
  | some n => simp only [FlockDeploy.encOpt, List.length_cons, FlockComplete.encNat_length] <;> omega

/-- A Flock statement with no lookup tables encodes in its files, the bytes of its optional files and at most 200 bytes
of lengths, flags, numbers and format. -/
theorem raw_le (r : FlockDeploy.Raw) (ht : r.tables = []) :
    r.encode.length ≤ 200 + r.statement.toUTF8.size + r.circuit.size + r.publicFile.size + osz r.draw +
      osz r.partition + osz r.program + osz r.workTable + osz r.closure + osz r.registered + osz r.publicInputs := by
  have hf : FlockDeploy.FORMAT.toUTF8.size ≤ 32 := by decide
  have := len_encOptB r.draw
  have := len_encOptB r.partition
  have := len_encOptB r.program
  have := len_encOptB r.workTable
  have := len_encOptB r.closure
  have := len_encOptB r.registered
  have := len_encOptB r.publicInputs
  have := len_encOptN r.work
  have := len_encOptN r.stratified
  simp only [FlockDeploy.Raw.encode, ht, FlockDeploy.encList, FlockDeploy.encBool, List.length_append,
    Part.len_encStr, Part.len_encBytes, FlockComplete.encNat_length, List.flatMap_nil, List.length_nil,
    List.length_cons]
  omega

/-- The compiled statement encodes in its inner statement, inner shape, registration and parts, and at most 100
bytes of lengths and format. -/
theorem recx_le (w : RecDeploy.RecX) :
    w.encode.length ≤ 100 + w.inner.length + w.recInner.size + osz w.registration +
      (w.parts.flatMap RecDeploy.PartX.encode).length := by
  have hf : RecDeploy.XFORMAT.toUTF8.size ≤ 32 := by decide
  have := len_encOptB w.registration
  simp only [RecDeploy.RecX.encode, FlockDeploy.encList, List.length_append, Part.len_encStr, Part.len_encBytes,
    FlockComplete.encNat_length]
  omega

theorem partx_eq (q : RecDeploy.PartX) : q.encode.length = 16 + q.base.encode.length + q.bindings.size := by
  have hs : (⟨q.base.encode.toArray⟩ : ByteArray).size = q.base.encode.length := by
    show q.base.encode.toArray.size = _
    simp
  simp only [RecDeploy.PartX.encode, List.length_append, Part.len_encBytes, hs]
  omega

/-- **Part `j`'s statement is no longer than the compiled `partSize`** where it is named by the component's part `j`
(`PartNames`). -/
theorem part_le {j : ℕ} {po : VStar.Recursive.PartOwn} {pp : VStar.Recursive.PartProof} {bs : Core.Blobs}
    {q : RecDeploy.PartX} {a : ℕ} (hN : PartNames po pp bs j q)
    (hs : VStar.Recursive.partSize j po pp bs = .ok a) : q.encode.length ≤ a := by
  have htg : Flock.Tags.find? VStar.Recursive.STATEMENT = some Flock.Tags.circuit := rfl
  unfold VStar.Recursive.partSize at hs
  simp only [htg] at hs
  obtain ⟨circuitB, hc, hs⟩ := bind_ok hs
  obtain ⟨pubB, hp, hs⟩ := bind_ok hs
  obtain ⟨recordB, hr, hs⟩ := bind_ok hs
  obtain ⟨record, hu⟩ : ∃ record, String.fromUTF8? recordB = some record := by
    cases hu : String.fromUTF8? recordB with
    | none => simp only [hu] at hs <;> exact absurd hs (throw_ne_ok _ _)
    | some record => exact ⟨record, rfl⟩
  simp only [hu] at hs
  obtain ⟨drawB, hd, hs⟩ := bind_ok hs
  obtain ⟨regB, hreg, hs⟩ := bind_ok hs
  obtain ⟨partitionB, hpa, hs⟩ := bind_ok hs
  obtain ⟨bindingsB, hb, hs⟩ := bind_ok hs
  have ha : a = VStar.Recursive.STMT_OVER + VStar.Recursive.STATEMENT.toUTF8.size + circuitB.size + pubB.size +
      po.program.size + partitionB.size + osz regB + osz drawB + bindingsB.size := (Except.ok.inj hs).symm
  have hcB : circuitB = q.base.circuit := Except.ok.inj (hc.symm.trans hN.circuit)
  have hpB : pubB = q.base.publicFile := Except.ok.inj (hp.symm.trans hN.publicFile)
  rw [hcB] at hd
  have hdraw : drawB = q.base.draw := Except.ok.inj (hd.symm.trans (hN.drawB _ _ _ htg hr hu))
  have hregB : regB = q.base.registered := Except.ok.inj (hreg.symm.trans hN.registeredB)
  have hbB : bindingsB = q.bindings := Except.ok.inj (hb.symm.trans hN.bindingsB)
  have hpart : q.base.partition = some partitionB := by
    have := hN.partitionB
    rw [hpa] at this
    exact (Except.ok.inj this).symm
  obtain ⟨hst, -, -, -, htab, -, hwt, -, hcl, hpi⟩ := hN.form
  have hraw := raw_le q.base htab
  have e1 : q.base.statement.toUTF8.size = VStar.Recursive.STATEMENT.toUTF8.size := by rw [hst]
  have e2 : q.base.circuit.size = circuitB.size := by rw [hcB]
  have e3 : q.base.publicFile.size = pubB.size := by rw [hpB]
  have e4 : osz q.base.draw = osz drawB := by rw [hdraw]
  have e5 : osz q.base.partition = partitionB.size := by rw [hpart] <;> rfl
  have e6 : osz q.base.program = po.program.size := by rw [hN.program] <;> rfl
  have e7 : osz q.base.workTable = 0 := by rw [hwt] <;> rfl
  have e8 : osz q.base.closure = 0 := by rw [hcl] <;> rfl
  have e9 : osz q.base.publicInputs = 0 := by rw [hpi] <;> rfl
  have e10 : osz q.base.registered = osz regB := by rw [hregB]
  have e11 : q.bindings.size = bindingsB.size := by rw [hbB]
  have hso : VStar.Recursive.STMT_OVER = 1024 := rfl
  have := partx_eq q
  omega

/-- **The parts' statements are no longer than the compiled `partsSize`**, part `k + i` named by the component's
(`PartNames`). -/
theorem parts_le {bs : Core.Blobs} : ∀ (qs : List RecDeploy.PartX) (ps : List VStar.Recursive.PartOwn)
    (pps : List VStar.Recursive.PartProof) (k T : ℕ), VStar.Recursive.partsSize bs k ps pps = .ok T →
    qs.length = ps.length → ps.length = pps.length →
    (∀ i q po pp, qs[i]? = some q → ps[i]? = some po → pps[i]? = some pp → PartNames po pp bs (k + i) q) →
    (qs.flatMap RecDeploy.PartX.encode).length ≤ T
  | [], _, _, _, _, _, _, _, _ => by simp
  | q :: qs, po :: ps, pp :: pps, k, T, hT, hl1, hl2, hN => by
    simp only [VStar.Recursive.partsSize] at hT
    obtain ⟨a, ha, hT⟩ := bind_ok hT
    obtain ⟨b, hb, hT⟩ := bind_ok hT
    have hab : a + b = T := Except.ok.inj hT
    have h0 := part_le (j := k) (by simpa using hN 0 q po pp (by simp) (by simp) (by simp)) ha
    have h1 := parts_le qs ps pps (k + 1) b hb (by simpa using hl1) (by simpa using hl2)
      fun i q' po' pp' e1 e2 e3 => by
        have := hN (i + 1) q' po' pp' (by simpa using e1) (by simpa using e2) (by simpa using e3)
        rwa [show k + (i + 1) = k + 1 + i by omega] at this
    rw [List.flatMap_cons, List.length_append]
    omega
  | _ :: _, [], _, _, _, _, hl1, _, _ => by simp at hl1
  | _ :: _, _ :: _, [], _, _, _, _, hl2, _ => by simp at hl2

/-- **`x` is no longer than the compiled `statementSize`** where it names the component's inputs (`Names`): every
field `x` encodes is a file the component read or the canonical bytes of a value it read. -/
theorem x_len_le {p : VStar.Recursive.Params} {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {x : Claims.Msg}
    {h : VStar.Recursive.InnerHeld} {T : ℕ} (hn : Names p e bs x)
    (hh : VStar.Recursive.innerHeld p.own e.proofs bs = .ok h) (hl2 : p.own.parts.size = e.proofs.parts.size)
    (hT : VStar.Recursive.statementSize h p.own e.proofs bs = .ok T) : x.length ≤ T := by
  obtain ⟨w, ri, hxw, hi, hI, hl, hN⟩ := hn
  have hx : w.encode = x := RecDeploy.decodeX_encode hxw
  have hin : ri.encode = w.inner := FlockDeploy.decode_encode hi
  unfold VStar.Recursive.statementSize at hT
  obtain ⟨circuitB, hc, hT⟩ := bind_ok hT
  obtain ⟨pubB, hp, hT⟩ := bind_ok hT
  obtain ⟨drawB, hd, hT⟩ := bind_ok hT
  obtain ⟨regB, hreg, hT⟩ := bind_ok hT
  obtain ⟨partitionB, hpa, hT⟩ := bind_ok hT
  obtain ⟨shapeB, hsh, hT⟩ := bind_ok hT
  obtain ⟨partsB, hps, hT⟩ := bind_ok hT
  have hTe : T = VStar.Recursive.STMT_OVER + (VStar.Recursive.STMT_OVER +
      VStar.Recursive.INNER_STATEMENT.toUTF8.size + circuitB.size + pubB.size + drawB.size + p.own.program.size +
      partitionB.size) + shapeB.size + osz regB + partsB := (Except.ok.inj hT).symm
  have hraw := raw_le ri hI.tables
  have hpart : ri.partition = some partitionB := by
    have := hI.partitionB
    rw [hpa] at this
    exact (Except.ok.inj this).symm
  have e1 : ri.statement.toUTF8.size = VStar.Recursive.INNER_STATEMENT.toUTF8.size := by rw [hI.statement]
  have e2 : ri.circuit.size = circuitB.size := by rw [Except.ok.inj (hc.symm.trans hI.circuit)]
  have e3 : ri.publicFile.size = pubB.size := by rw [Except.ok.inj (hp.symm.trans hI.pub)]
  have e4 : osz ri.draw = drawB.size := by rw [hI.draw, hd] <;> rfl
  have e5 : osz ri.partition = partitionB.size := by rw [hpart] <;> rfl
  have e6 : osz ri.program = p.own.program.size := by rw [hI.program] <;> rfl
  have e7 : osz ri.workTable = 0 := by rw [hI.workTable] <;> rfl
  have e8 : osz ri.closure = 0 := by rw [hI.closure] <;> rfl
  have e9 : osz ri.registered = 0 := by rw [hI.registered] <;> rfl
  have e10 : osz ri.publicInputs = 0 := by rw [hI.publicInputs] <;> rfl
  have e11 : w.recInner.size = shapeB.size := by rw [Except.ok.inj ((hI.recInnerB h hh).symm.trans hsh)]
  have e12 : osz w.registration = osz regB := by
    have hr := hI.registration
    cases hor : p.own.registration with
    | none =>
      rw [hor] at hr hreg
      have a1 : regB = none := (Except.ok.inj hreg).symm
      have a2 : w.registration = none := (Option.some.inj hr).symm
      rw [a1, a2]
    | some d =>
      rw [hor, Option.mapM_some] at hr hreg
      cases hg : bs.get d "rec inner registration" with
      | error err =>
        rw [hg] at hreg
        first | cases hreg | simp [Functor.map, Except.map] at hreg
      | ok b =>
        rw [hg] at hreg hr
        have a1 : regB = some b := (Except.ok.inj hreg).symm
        have a2 : w.registration = some b := (Option.some.inj hr).symm
        rw [a1, a2]
  have hps' := parts_le w.parts p.own.parts.toList e.proofs.parts.toList 0 partsB hps (by simp [hl]) (by simp [hl2])
    fun i q po pp e1 e2 e3 => by
      simpa using hN i q po pp e1 (by simpa using e2) (by simpa using e3)
  have hrx := recx_le w
  rw [hx, ← hin] at hrx
  have hso : VStar.Recursive.STMT_OVER = 1024 := rfl
  omega

/-- The compiled `statementFits` at `FIELD_MAX` passes only on a statement size under `2 ^ 56`. -/
theorem statementFits_lt {h : VStar.Recursive.InnerHeld} {o : VStar.Recursive.Own} {pr : VStar.Recursive.Proofs}
    {bs : Core.Blobs} (hf : VStar.Recursive.statementFits VStar.Recursive.FIELD_MAX h o pr bs = .ok ()) :
    ∃ T, VStar.Recursive.statementSize h o pr bs = .ok T ∧ T < 2 ^ 56 := by
  unfold VStar.Recursive.statementFits at hf
  obtain ⟨T, hT, hf⟩ := bind_ok hf
  refine ⟨T, hT, ?_⟩
  unfold VStar.Recursive.need at hf
  by_cases hc : T < 2 ^ 56
  · exact hc
  · have hd : decide (T < VStar.Recursive.FIELD_MAX) = false :=
      decide_eq_false (by simpa [VStar.Recursive.FIELD_MAX] using hc)
    simp only [hd, Bool.false_eq_true, ↓reduceIte] at hf
    exact absurd hf (throw_ne_ok _ _)

/-- **`Refusals.pub`**: where the compiled component accepts, `x` names its inputs, and `P s` is part `s`'s public file
on the blobs, `x` with each part's public file is under `2 ^ 64` bytes at every part `nS x` counts: the component refuses
a statement of `2 ^ 56` bytes or more (`statementFits`, which bounds `x`: `x_len_le`) and a public file of as many
(`fits`). -/
theorem pub_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    {x : Claims.Msg} (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) (hn : Names p e bs x)
    {P : ℕ → ByteArray} {what : String}
    (hP : ∀ s q, e.proofs.parts[s]? = some q → bs.get q.pub what = .ok (P s)) :
    ∀ s < RecDeploy.nS x, x.length + (P s).size < 2 ^ 64 := by
  intro s hs
  obtain ⟨h, ss, hh, -, -, hsz1, hsz2, hparts, hfit, -⟩ := parts_of_accepts hacc
  obtain ⟨T, hT, hTlt⟩ := statementFits_lt hfit
  have hx := x_len_le hn hh (by omega) hT
  obtain ⟨w, ri, hxw, -, -, hl, -⟩ := hn
  have hle := nS_le hxw
  have hs1 : s < p.own.parts.size := by omega
  have hs2 : s < e.proofs.parts.size := by omega
  obtain ⟨pubB, hpb, hlt⟩ := partVerdict_pub (hparts s hs1 hs2)
  have e1 := hP s _ (Array.getElem?_eq_getElem hs2)
  have e2 : P s = pubB := Except.ok.inj ((get_what e1).symm.trans hpb)
  rw [e2]
  have h64 : (2 : ℕ) ^ 56 + 2 ^ 56 ≤ 2 ^ 64 := by norm_num
  omega

/-- **`Refusals.record`**: the inner session's record on the blobs is under `2 ^ 64` bytes where the compiled
component accepts. -/
theorem record_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) {R : ByteArray} {what : String}
    (hR : bs.get e.proofs.record what = .ok R) : R.size < 2 ^ 64 := by
  obtain ⟨h, _, hh, -⟩ := parts_of_accepts hacc
  obtain ⟨_, _, _, recB, _, _, -, -, -, -, hr, hlt, -⟩ := innerHeld_reads hh
  have e1 : R = recB := Except.ok.inj ((get_what hR).symm.trans hr)
  rw [e1]
  exact lt_of_lt_of_le hlt (by norm_num)

/-- **`Refusals.transcript`**: the coin server's transcript on the blobs is under `2 ^ 64` bytes where the compiled
component accepts. -/
theorem transcript_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) {T : ByteArray} {what : String}
    (hT : bs.get e.proofs.transcript what = .ok T) : T.size < 2 ^ 64 := by
  obtain ⟨h, _, hh, -⟩ := parts_of_accepts hacc
  obtain ⟨_, _, _, _, trB, _, -, -, -, -, -, -, htr, hlt, -⟩ := innerHeld_reads hh
  have e1 : T = trB := Except.ok.inj ((get_what hT).symm.trans htr)
  rw [e1]
  exact lt_of_lt_of_le hlt (by norm_num)

/-- **`Refusals.rows`**: V\*'s staged rows on the blobs are under `2 ^ 64` bytes where the compiled component
accepts. -/
theorem rows_of_compiled {p : VStar.Recursive.Params} {c : SampledProofs.OneStage.Commitment}
    {e : VStar.Recursive.Evidence} {bs : Core.Blobs} {pub : SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (hacc : VStar.Recursive.checkBlobs p c e bs = .accept pub prof) {W : ByteArray} {what : String}
    (hW : bs.get e.proofs.rows what = .ok W) : W.size < 2 ^ 64 := by
  obtain ⟨h, _, hh, -⟩ := parts_of_accepts hacc
  obtain ⟨_, _, _, _, _, rowsB, -, -, -, -, -, -, -, -, hrw, hlt⟩ := innerHeld_reads hh
  have e1 : W = rowsB := Except.ok.inj ((get_what hW).symm.trans hrw)
  rw [e1]
  exact lt_of_lt_of_le hlt (by norm_num)

end Verity.Recursive.RefusalsOf
