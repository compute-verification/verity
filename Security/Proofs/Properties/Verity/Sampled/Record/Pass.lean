import Proofs.Properties.Verity.Sampled.Record.Accepts
import Proofs.Properties.SampledProofs.Credited
import Proofs.Flock.Claims.Sampling.Select.Prefix

/-!
# v2's acceptance keeps the run's draw file's units

Where one-stage v2 (`SampledProofs.OneStageZk.checkBlobsAt`) accepts, at the coins `ξ`, evidence it parsed, the run's
draw file proves the units the draw rounds' coins draw (`units_of_accepts`, `drawOk`'s first half):

* **v2 credits its one session with the coins' draw** (`Properties.SampledProofs.Credited.unitsAt_of_checkBlobsAt`): at
  the public file's load and the session record's draw, `Units.unitsAt` is the units `ξ` draws under the verifier's law
  as derived over the registered population;
* **the registered population is `A`'s** (`Properties.SampledProofs.Credited.population_of_checkBlobsAt`): v2 holds the
  registration to the public file, which is `A`'s (`Names`), and `A`'s load of it counts `A`'s population;
* **that draw is the record's** (`drawnFrom_eq`, `Properties.SampledProofs.SecurityProofs.drawUnits_append`): under the
  law `lawAt` reads over `A`'s population (`SetsUp`'s), v2's draw from `ξ` is `drawn`'s from any bytes `ξ` begins, the
  draw rounds' coins' among them;
* **the session record's draw is the run's draw file** (`Shows`' first half), so `provedOf` reads at the run's draw file
  what `unitsAt` reads.

What it takes beside v2's acceptance: `ξ` is the first bytes of the draw rounds' coins' (`LiveCoins`' first half).
-/

namespace Verity.Sampled.Record

open Lean (Json)
open Claims (Seen)
open FlockDeploy (Coin)
open Verity.Recursive.BaseSetup (bind_ok)

/-- **v2's draw under `lawAt`'s law**: where the verifier's law derives `lawJ` and `lawAt` reads the law `L` over `n`
units, v2's draw over `n` units (`SampledProofs.OneStage.drawnFrom`) is `SampledProofs.Draw.drawUnits` under `L`. -/
theorem drawnFrom_eq {v : _root_.SampledProofs.Registration.Verifier} {lawJ : Json} {n : ℕ} {L : Flock.Draw.Law}
    (src : ByteArray)
    (hder : _root_.SampledProofs.Draw.derive v.program v.query v.law v.workTable = .ok lawJ)
    (hlaw : lawAt v n = .ok L) :
    _root_.SampledProofs.OneStage.drawnFrom v lawJ n src = _root_.SampledProofs.Draw.drawUnits L n src := by
  unfold lawAt at hlaw
  obtain ⟨lawJ', hJ', hlaw⟩ := bind_ok hlaw
  rw [hder] at hJ'
  obtain rfl := Except.ok.inj hJ'
  obtain ⟨derived, hd, hlaw⟩ := bind_ok hlaw
  obtain ⟨fl, hf, hlaw⟩ := bind_ok hlaw
  have hL : derived.getD fl = L := Except.ok.inj hlaw
  unfold _root_.SampledProofs.OneStage.drawnFrom
  rw [hd, hf]
  exact congrArg (fun L => _root_.SampledProofs.Draw.drawUnits L n src) hL

/-- **v2's acceptance keeps the run's draw file's units**: where v2 accepts evidence it parsed at the coins `ξ`, reading
`A`'s and `o`'s inputs (`Names`), with the verifier's law read over `A`'s population as `A`'s (`lawAt`) and `ξ` the
first bytes of the coins `cs`', at a run that reaches the draw file round and whose draw file the blobs show (`Shows`),
the draw file proves the units `cs` draw, ascending. -/
theorem units_of_accepts (A : Live) (o : Own) {b : ByteArray} {p : _root_.SampledProofs.OneStageZk.Params}
    {c : _root_.SampledProofs.OneStage.Commitment} {ev : _root_.SampledProofs.OneStageZk.Evidence} {bs : Core.Blobs}
    {ξ : ByteArray} {pub : _root_.SampledProofs.OneStage.Public} {prof : Core.Integrity}
    (he : _root_.SampledProofs.OneStageZk.evidenceOf b = .ok ev)
    (hacc : _root_.SampledProofs.OneStageZk.checkBlobsAt p c ξ ev bs = .accept pub prof) (hn : Names A o p bs)
    (hlaw : lawAt p.base.verifier A.n = .ok A.law)
    {cs : List Coin} (hξ : ∃ t, drawBytes cs = ξ ++ t) {v : Seen} (hlen : DRAW_ROUNDS < v.msgs.length)
    (hshow : Shows A o ev bs v) :
    provedOf A (drawOf v) = some ((drawn A (drawBytes cs)).toArray.qsort (· < ·)) := by
  obtain ⟨s, pubB, B, record, law, n, units, read, hs, hpB, hB, hr, hder, hn', hd, hc⟩ :=
    _root_.Properties.SampledProofs.Credited.unitsAt_of_checkBlobsAt he hacc
  have hpb : pubB = A.I.publicFile := Except.ok.inj (hpB.symm.trans hn.pub)
  subst hpb
  have hpop := _root_.Properties.SampledProofs.Credited.population_of_checkBlobsAt hacc hpB A.hpub
  rw [hn', A.hn] at hpop
  obtain rfl := Option.some.inj hpop
  rw [drawnFrom_eq ξ hder hlaw] at hd
  obtain ⟨t, ht⟩ := hξ
  have hd' := _root_.Properties.SampledProofs.SecurityProofs.drawUnits_append (t := t) hd
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

end Verity.Sampled.Record
