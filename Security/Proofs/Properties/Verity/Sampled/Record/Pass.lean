module

public import Proofs.Properties.Verity.Sampled.Record.Accepts
public import Proofs.Properties.Verity.Sampled.Record.Nums
public import Proofs.Properties.Spot.Credited
public import Proofs.Flock.Claims.Sampling.Select.Prefix

/-!
# v2's acceptance keeps the run's draw file

Where one-stage v2 (`Spot.OneStageZk.checkBlobsAt`) accepts, at the coins `ξ`, evidence it parsed, the draw
file round keeps the run's draw file (`pass_of_accepts`, `drawOk`). First, the run's draw file proves the units the draw
rounds' coins draw (`units_of_accepts`, `units_of_pass`, `drawOk`'s first half):

* **v2 credits its one session with the coins' draw** (`Properties.Spot.Credited.unitsAt_of_checkBlobsAt`): at
  the public file's load and the session record's draw, `Units.unitsAt` is the units `ξ` draws under the verifier's law
  as derived over the registered population;
* **the registered population is `A`'s** (`Properties.Spot.Credited.population_of_checkBlobsAt`): v2 holds the
  registration to the public file, which is `A`'s (`Names`), and `A`'s load of it counts `A`'s population;
* **that draw is the record's** (`drawnFrom_eq`, `Properties.Spot.SecurityProofs.drawUnits_append`): under the
  law `lawAt` reads over `A`'s population (`SetsUp`'s), v2's draw from `ξ` is `drawn`'s from any bytes `ξ` begins, the
  draw rounds' coins' among them;
* **the session record's draw is the run's draw file** (`Shows`' first half), so `provedOf` reads at the run's draw file
  what `unitsAt` reads.

Second, its session has the statement's numbers (`numsOk`, through `numsOk_of_refOk`): v2 sets the session up at the
run's draw file, at one table whose unit count is the draw's proved units (`shape_sess`,
`FlockSoundness.Discharge.Layout.drawn_spec`), which are its drawn units alone (`proved_of_drawLaw`), as many as the
units v2 credits (`unitsAt_size`), the `A.kd` units the coins draw (`drawUnits_of_accepts`, `drawn_length`).

What it takes beside v2's acceptance: `ξ` is the first bytes of the draw rounds' coins' (`LiveCoins`' first half), and
the run's draw file and `o`'s partition object and registered file are each under `2^64` bytes, the lengths of the
session's statement v2 reads only as the JSON they parse to (`fits_of_accepts`).
-/

@[expose] public section

namespace Verity.Sampled.Record

open Lean (Json)
open Claims (Seen Prover Transcript run coinSeq)
open FlockDeploy (Coin)
open Verity.Recursive.BaseSetup (bind_ok)

/-- **v2's draw under `lawAt`'s law**: where the verifier's law derives `lawJ` and `lawAt` reads the law `L` over `n`
units, v2's draw over `n` units (`Spot.OneStage.drawnFrom`), where it draws, is `Spot.Draw.drawUnits`'s
under `L`: it refuses a Bernoulli `L` (`Spot.OneStage.coveredDesign`) and draws under any other. -/
theorem drawnFrom_eq {v : _root_.Spot.Registration.Verifier} {lawJ : Json} {n : ℕ} {L : Flock.Draw.Law}
    (src : ByteArray) {r : Array ℕ × ℕ}
    (hder : _root_.Spot.Draw.derive v.program v.query v.law v.workTable = .ok lawJ)
    (hlaw : lawAt v n = .ok L) (h : _root_.Spot.OneStage.drawnFrom v lawJ n src = .ok r) :
    _root_.Spot.Draw.drawUnits L n src = .ok r := by
  unfold lawAt at hlaw
  obtain ⟨lawJ', hJ', hlaw⟩ := bind_ok hlaw
  rw [hder] at hJ'
  obtain rfl := Except.ok.inj hJ'
  obtain ⟨derived, hd, hlaw⟩ := bind_ok hlaw
  obtain ⟨fl, hf, hlaw⟩ := bind_ok hlaw
  have hL : derived.getD fl = L := Except.ok.inj hlaw
  unfold _root_.Spot.OneStage.drawnFrom at h
  rw [hd, hf] at h
  change (_root_.Spot.OneStage.coveredDesign (derived.getD fl) >>= fun l =>
    _root_.Spot.Draw.drawUnits l n src) = _ at h
  rw [hL] at h
  revert h
  cases L
  case bernoulli a b =>
    intro h
    change Except.error _ = _ at h
    cases h
  all_goals intro h; exact h

/-- **v2's acceptance keeps the run's draw file's units**: where v2 accepts evidence it parsed at the coins `ξ`, reading
`A`'s and `o`'s inputs (`Names`), with the verifier's law read over `A`'s population as `A`'s (`lawAt`) and `ξ` the
first bytes of the coins `cs`', at a run that reaches the draw file round and whose draw file the blobs show (`Shows`),
the draw file proves the units `cs` draw, ascending. -/
theorem units_of_accepts (A : Live) (o : Own) {b : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.Spot.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {cs : List Coin} (hξ : ∃ t, drawBytes cs = ξ ++ t) {v : Seen} (hlen : DRAW_ROUNDS < v.msgs.length)
    (hshow : Shows A o ev bs v) :
    provedOf A (drawOf v) = some ((drawn A (drawBytes cs)).toArray.qsort (· < ·)) := by
  obtain ⟨s, pubB, B, record, law, n, units, read, hs, hpB, hB, hr, hder, hn', hd, hc⟩ :=
    _root_.Properties.Spot.Credited.unitsAt_of_checkBlobsAt he hacc
  have hpb : pubB = A.I.publicFile := Except.ok.inj (hpB.symm.trans hn.pub)
  subst hpb
  have hpop := _root_.Properties.Spot.Credited.population_of_checkBlobsAt hacc hpB A.hpub
  rw [hn', A.hn] at hpop
  obtain rfl := Option.some.inj hpop
  replace hd := drawnFrom_eq ξ hder hlaw hd
  obtain ⟨t, ht⟩ := hξ
  have hd' := _root_.Properties.Spot.SecurityProofs.drawUnits_append (t := t) hd
  rw [← ht] at hd'
  have hdr : drawn A (drawBytes cs) = units.toList := by
    unfold drawn
    simp only [hd']
  obtain ⟨rec, hrecD, J, hJ, hrJ⟩ := hshow.1 hlen s (by rw [hs]; exact Array.mem_singleton_self s)
  have hrec : rec = record := by
    have h2 := congrArg String.fromUTF8?
      (Except.ok.inj ((get_what (w' := s!"session {s.name}'s record") hrecD).symm.trans hB))
    rw [FlockComplete.fromUTF8?_toUTF8, hr] at h2
    exact Option.some.inj h2
  rw [hrec] at hrJ
  have hu := hc _ _ _ _ _ A.hpub hrJ
  unfold provedOf
  rw [hJ, hdr, Array.toArray_toList]
  exact hu

/-- **v2's draw ends at the coins `cs`**: where v2 accepts evidence it parsed at the coins `ξ`, reading `A`'s and `o`'s
inputs (`Names`), with the verifier's law read over `A`'s population as `A`'s (`lawAt`), `A`'s draw from any bytes `ξ`
begins ends. -/
theorem drawUnits_of_accepts (A : Live) (o : Own) {b : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.Spot.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {cs : List Coin} (hξ : ∃ t, drawBytes cs = ξ ++ t) :
    ∃ u rd, _root_.Spot.Draw.drawUnits A.law A.n (drawBytes cs) = .ok (u, rd) := by
  obtain ⟨s, pubB, B, record, law, n, units, read, hs, hpB, hB, hr, hder, hn', hd, hc⟩ :=
    _root_.Properties.Spot.Credited.unitsAt_of_checkBlobsAt he hacc
  have hpb : pubB = A.I.publicFile := Except.ok.inj (hpB.symm.trans hn.pub)
  subst hpb
  have hpop := _root_.Properties.Spot.Credited.population_of_checkBlobsAt hacc hpB A.hpub
  rw [hn', A.hn] at hpop
  obtain rfl := Option.some.inj hpop
  replace hd := drawnFrom_eq ξ hder hlaw hd
  obtain ⟨t, ht⟩ := hξ
  have hd' := _root_.Properties.Spot.SecurityProofs.drawUnits_append (t := t) hd
  rw [← ht] at hd'
  exact ⟨_, _, hd'⟩

/-- The draw rounds' coins begin with `ξ` where the run's first `DRAW_ROUNDS` coins do (`LiveCoins`' first half). -/
theorem prefix_of_live (A : Live) (o : Own) {ξ : ByteArray} {Ω : Type} (P : Prover Ω) (ω : Ω)
    (r : Fin ((V A o).k (xOf A o)) → Coin)
    (h : ∃ t, drawBytes (((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map
      (coinSeq r)).take DRAW_ROUNDS) = ξ ++ t) :
    ∃ t, drawBytes (coinsAt r) = ξ ++ t := by
  have h64 := reach_draw FlockDeploy.sha512 P ω r
  obtain ⟨t, ht⟩ := h
  refine ⟨t, ?_⟩
  rw [← ht]
  unfold coinsAt
  rw [← List.map_take, List.take_range, Nat.min_eq_left (Nat.le_of_lt h64)]

/-- **v2's acceptance keeps the run's draw file's units, at the run**: where v2 accepts evidence it parsed at the coins
`ξ`, reading `A`'s and `o`'s inputs (`Names`), with the verifier's law read over `A`'s population as `A`'s (`lawAt`),
at a run whose draw file the blobs show (`Shows`) and whose draw rounds' coins begin with `ξ` (`LiveCoins`' first
half), the run's draw file proves the units the draw rounds' coins draw, ascending: `drawOk`'s first half. -/
theorem units_of_pass (A : Live) (o : Own) {b : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.Spot.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V A o).k (xOf A o)) → Coin)
    (hshow : Shows A o ev bs (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen)
    (hlive : LiveCoins A o ξ (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen
      ((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r))) :
    provedOf A (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen) =
      some ((drawn A (drawBytes (coinsAt r))).toArray.qsort (· < ·)) := by
  have h64 := reach_draw FlockDeploy.sha512 P ω r
  have hsm : (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen.msgs =
      (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs := by
    simp only [Transcript.seen]
  exact units_of_accepts A o he hacc hn hlaw (prefix_of_live A o P ω r hlive.1) (by rw [hsm]; exact h64) hshow

/-- **v2's acceptance makes the run's draw file a reference draw file** (`RefOk`): at `pass_of_accepts`'s hypotheses,
its session sets up one table of `A.kd` units. -/
theorem refOk_of_accepts (A : Live) (o : Own) {b : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.Spot.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V A o).k (xOf A o)) → Coin)
    (hshow : Shows A o ev bs (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen)
    (hlive : LiveCoins A o ξ (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen
      ((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r)))
    (hd : (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen).length < 2 ^ 64)
    (hpart : o.partition.size < 2 ^ 64) (hreg : ∀ b ∈ o.registered, b.size < 2 ^ 64) :
    RefOk A o (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen) := by
  have hfits := fits_of_accepts hacc hn hd hpart hreg
  have hu := units_of_pass A o he hacc hn hlaw P ω r hshow hlive
  obtain ⟨s, rd, B, record, ksFiles, population, hs, hpl, hrd, hB, hutf, hv⟩ := verdict_of_accepts hacc hn
  have h64 := reach_draw FlockDeploy.sha512 P ω r
  have hsm : (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen.msgs =
      (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs := by
    simp only [Transcript.seen]
  have hmem : s ∈ ev.sessions := by rw [hs]; exact Array.mem_singleton_self s
  obtain ⟨rec, hrecD, J, hJ, hrJ⟩ := hshow.1 (by rw [hsm]; exact h64) s hmem
  have hrec : rec = record := by
    have h2 := congrArg String.fromUTF8?
      (Except.ok.inj ((get_what (w' := s!"session {s.name}'s record") hrecD).symm.trans hB))
    rw [FlockComplete.fromUTF8?_toUTF8, hutf] at h2
    exact Option.some.inj h2
  obtain ⟨si, hins, hm, hsd⟩ := ins_sess hn hpl hrd hJ
  have hdraw : si.read.draw = Flock.Witness.recordDraw record := by
    rw [hsd, ← hrec, hrJ]
  obtain ⟨spec, tabs, pj, oj, hst⟩ := setupAt_of_verdictOf hv
  obtain ⟨sts, hsu, -⟩ := setup_of_setupAt hm hdraw hst
  obtain ⟨_sts, -, -, hlawc⟩ := setupAt_ok hst
  have hso := setupOf_sess hfits hins hsu
  obtain ⟨t, pi, pop, _B, hsp, hsts, -, -, -, -, -, hsome, htn⟩ := shape_sess hins hsu
  rw [hsp, hsts] at hso
  refine ⟨si, t, hins, hso, ?_⟩
  have hdr := hsome _ hsd
  unfold Flock.HmIn.drawn at hdr
  obtain ⟨D, hD, -, hnD, -⟩ := FlockSoundness.Discharge.Layout.drawn_spec hdr
  rw [hdraw.symm.trans hsd] at hlawc
  rw [ite_eq_left Option.isSome_some] at hlawc
  have hX : _root_.Properties.Spot.Units.unitsAt A.pub.header (Flock.norm J) =
      some ((drawn A (drawBytes (coinsAt r))).toArray.qsort (· < ·)) := by
    have h := hu
    unfold provedOf at h
    rw [hJ] at h
    exact h
  obtain ⟨u, rd', hdu⟩ := drawUnits_of_accepts A o he hacc hn hlaw (prefix_of_live A o P ω r hlive.1)
  rw [htn, hnD, proved_of_drawLaw hD hlawc, ← unitsAt_size hX hD, Verity.QSort.size_qsort,
    List.size_toArray]
  exact drawn_length A hdu

/-- **v2's acceptance keeps the run's draw file**: where v2 accepts evidence it parsed at the coins `ξ`, reading `A`'s
and `o`'s inputs (`Names`), with the verifier's law read over `A`'s population as `A`'s (`lawAt`), at a run whose draw
file the blobs show (`Shows`) and whose draw rounds' coins begin with `ξ` (`LiveCoins`' first half), where the run's
draw file and `o`'s partition object and registered file are each under `2^64` bytes, the draw file round keeps the
run's draw file (`drawOk`): it proves the units the draw rounds' coins draw, its session sets up one table of `A.kd`
units (`refOk_of_accepts`), so it has the statement's numbers, and its session's statement fits its length words
(`fits_of_accepts`). -/
theorem pass_of_accepts (A : Live) (o : Own) {b : ByteArray} {p : _root_.Spot.OneStageZk.Params}
    {c : _root_.Spot.OneStage.Commitment} {ev : _root_.Spot.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.Spot.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.Spot.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.Spot.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {Ω : Type} (P : Prover Ω) (ω : Ω) (r : Fin ((V A o).k (xOf A o)) → Coin)
    (hshow : Shows A o ev bs (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen)
    (hlive : LiveCoins A o ξ (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen
      ((List.range (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).msgs.length).map (coinSeq r)))
    (hd : (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen).length < 2 ^ 64)
    (hpart : o.partition.size < 2 ^ 64) (hreg : ∀ b ∈ o.registered, b.size < 2 ^ 64) :
    drawOk A o (drawOf (run FlockDeploy.sha512 P (V A o) (xOf A o) ω r).seen) (coinsAt r) :=
  ⟨units_of_pass A o he hacc hn hlaw P ω r hshow hlive,
    numsOk_of_refOk (refOk_of_accepts A o he hacc hn hlaw P ω r hshow hlive hd hpart hreg),
    fits_of_accepts hacc hn hd hpart hreg⟩

end Verity.Sampled.Record
