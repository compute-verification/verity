import Proofs.Flock.Claims.Recursion.VStarNets
import Proofs.Flock.Claims.Recursion.VStarFitHeld

/-!
# `TraceOk` at the fitted port inputs (`traceOk_fitPin`)

`traceOk_deployed` reads the port inputs only through their refs (`partIn_refs`) and their rows' sizes (`RowsOk`). At
the fitted port inputs `fitPin c pin` the refs are the stager's (`refs_fitPin`) and every row is of its port's words
and every salt 192 bytes (`rows_fitPin`), where each port is the first of its name (`portWords_nodup`). A staged part's
ports are V\*'s (`checkParsed`), so that holds where V\*'s row ports at the statement have distinct names
(`names_nodup`). So `TraceOk` holds at the fitted port inputs with no premise on the stager's rows.

The tops two reps commit at a level above 0 are two runs of the level's cap (`topsAt`) unless they agree in every node
and every 192-byte salt (`sameTop`); salts that differ anywhere keep them apart (`sameTop_false_of_salt`). The salts
are the inner prover's tape's independent uniform draws, so two reps' tops coincide with probability at most
`2^-1536` a pair: the honest theorem's tape exception.
-/

namespace RecDeploy.VStarTraceOk

open _root_.Flock
open Claims (Msg)
open RecDeploy.VStarBuild (partAt progOf portRow fitPin portWords refs_fitPin rows_fitPin)
open RecDeploy.VStarTrace (TraceOk)

/-- **Port `q`'s name finds port `q`**, where the ports' names are distinct, so its word count is port `q`'s. -/
theorem portWords_nodup {c : Circuit} (hnd : (c.ports.toList.map (·.name)).Nodup) :
    ∀ q (hq : q < c.ports.size), portWords c c.ports[q].name = c.ports[q].words := by
  intro q hq
  have hf : c.ports.findIdx? (·.name == c.ports[q].name) = some q := by
    rw [Array.findIdx?_eq_some_iff_getElem]
    refine ⟨hq, by simp, fun j hj => ?_⟩
    have hjs : j < c.ports.size := by omega
    have hne : c.ports[j].name ≠ c.ports[q].name := by
      intro he
      have h := hnd.getElem_inj_iff.1
        (show (c.ports.toList.map (·.name))[j]'(by simpa using hjs) =
          (c.ports.toList.map (·.name))[q]'(by simpa using hq) by simpa using he)
      omega
    simpa using hne
  unfold portWords
  rw [hf]
  simp [Array.getElem?_eq_getElem hq]

/-- **A parsed circuit's port names are V\*'s**: distinct where V\*'s row ports at the statement have distinct names. -/
theorem names_nodup {sv : _root_.VStar.Stmt} {c : Circuit} (hcp : RecVStar.checkParsed sv c = .ok ())
    (hV : ∀ u, ((RecVStar.vstarRows sv.sub u).map (·.1)).Nodup) : (c.ports.toList.map (·.name)).Nodup := by
  obtain ⟨u, -, -, -, -, hr⟩ := FlockSoundness.Discharge.Recursion.checkParsed_ok hcp
  have h2 : c.ports.toList.map (·.name) = (RecVStar.vstarRows sv.sub u).map (·.1) := by
    rw [← hr, List.map_map]
    rfl
  rw [h2]
  exact hV u

/-- **`TraceOk` at a staged part's fitted port inputs**: `traceOk_deployed`'s checks, with the refs kept by fitting
(`refs_fitPin`), the rows fitted to their ports (`rows_fitPin`) and the three nets (`nets_three`), where V\*'s row ports
at the statement have distinct names. -/
theorem traceOk_fitPin {st : Stage} {s : ℕ} {x xO : Msg} (hS : StagedPart st s x xO) {sv : _root_.VStar.Stmt}
    {c : Circuit} (hp : partAt x s = some (sv, c)) {src : _root_.VStar.In.Src} {pin : _root_.VStar.In.PartIn}
    (hi : _root_.VStar.In.partIn src (progOf c) sv = .ok pin)
    (hV : ∀ u, ((RecVStar.vstarRows sv.sub u).map (·.1)).Nodup) :
    TraceOk c (fitPin c pin) := by
  have h3 := VStarNets.nets_three hS hp
  obtain ⟨r, hr, hsv, hc⟩ := partAt_some hp
  obtain ⟨p, r', sv', hx, ht, hpi, hr', ⟨ss, hss, hs⟩, h0, htab⟩ := stagedTables hS
  rw [hr] at hr'
  cases hr'
  have hsv' : ss[s]? = some sv := by
    rw [hss] at hsv
    exact hsv
  rw [hs] at hsv'
  cases hsv'
  have hpc := partCircuit_staged hS hx ht hpi hr hc h0
  obtain ⟨tags, P⟩ := Flock.Readable.parseFacts_of_parse hx ht hpi h0
  have hsc := (staged_scope hS hx h0).1
  have hcp := (htab 0 h0).2.2
  have hk : p.sts[0].c.kLog ≤ 27 := by
    have hx' := hx
    unfold Flock.Readable.parseBytes at hx'
    obtain ⟨rd, -, hrd⟩ := Option.bind_eq_some_iff.1 hx'
    obtain ⟨st0, h0', -, -, -, -, hk, -⟩ :=
      Flock.Readable.table0 (Flock.Readable.tablesOf_of_parses (Flock.Readable.parses_of_parse hrd)) ht hpi
    rw [Array.getElem?_eq_getElem h0] at h0'
    cases h0'
    exact hk
  rw [hpc] at P hsc hcp hk
  exact traceOk_of P hsc hcp hk h3 (refs_fitPin (partIn_refs hi))
    (rows_fitPin c pin (portWords_nodup (names_nodup hcp hV)))

/-- **Two reps' tops at a level stay apart where a salt differs**: `topsAt` merges them only where every node and
every 192-byte salt agree, which on the inner prover's uniform tape happens with probability at most `2^-1536`. -/
theorem sameTop_false_of_salt {a b : Array (ByteArray × ByteArray)} {i : ℕ} (ha : i < a.size) (hb : i < b.size)
    (h : a[i].2.data ≠ b[i].2.data) : _root_.VStar.In.sameTop a b = false := by
  unfold _root_.VStar.In.sameTop
  apply Bool.eq_false_iff.2
  intro ht
  rw [Bool.and_eq_true, Array.all_eq_true] at ht
  have hz : i < (a.zip b).size := by
    rw [Array.size_zip]
    exact Nat.lt_min.2 ⟨ha, hb⟩
  have hi := ht.2 i hz
  simp only [Array.getElem_zip, Bool.and_eq_true, beq_iff_eq] at hi
  exact h hi.2

end RecDeploy.VStarTraceOk
