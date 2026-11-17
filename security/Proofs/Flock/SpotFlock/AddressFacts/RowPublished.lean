module

public import Proofs.Flock.SpotFlock.AddressFacts.RowPlace
public import Proofs.Flock.SpotFlock.Depths
import Proofs.Flock.Bridge.LeafMap
import Proofs.Flock.Claims.JsonNorm
import Proofs.Flock.Soundness.Discharge.Hidden.Public
import Proofs.Flock.SpotFlock.EntriesAt
import Proofs.Flock.SpotFlock.HeaderRoots
import Proofs.Flock.SpotFlock.Opens
import Std.Data.TreeMap.Raw.Lemmas
import Std.Data.TreeMap.Raw.WF
import all Init.Data.Ord.String
import all Lean.Data.Json.Basic

/-! The published rows at a template's own written port with no registered read (`publishedRows_of_runs`). -/

@[expose] public section

namespace SpotFlock

open Claims Claims.Sampling SamplingDeploy
open Lean (Json)

namespace AddressRow

variable {h : Held}

theorem populationOk_count {q : ℕ} {p : Spot.SecondStage.Part} {hd : Json}
    (hp : Spot.SecondStage.populationOk q p hd = true) :
    ∃ m, Spot.nat? (Spot.getD hd "instances") = some m ∧ m = q * p.n := by
  unfold Spot.SecondStage.populationOk at hp
  split at hp
  · simp only [Bool.and_eq_true, decide_eq_true_eq] at hp
    exact ⟨_, ‹_›, eq_of_beq hp.1.1⟩
  · cases hp

theorem unshared_of_stmtOk {x m₁ m₂ : Msg} {s} {c₀ : FlockDeploy.Coin} {c₁ : Stage h} {u : ℕ}
    {ins : FlockDeploy.Ins} {spec : _root_.Flock.SessionSpec} {sts : Array _root_.Flock.Stmt}
    (hx : parse h x = some s) (hok : stmtOk h x m₁ m₂ c₀ c₁ = true)
    (hu : u ∈ h.reg.checked (units (base h) x) (src c₀) (stageSrc c₁))
    (hset : FlockDeploy.setupOf true (unitStmt h m₂ u) = some (ins, spec, sts))
    {j : ℕ} (hj : j < sts.size) {q : ℕ} {pop : _root_.Flock.Public} (hq : q < sts[j].c.ports.size)
    (hnr : ∀ own, ins.registered = some own → ∀ reg ∈ own, reg.name ≠ sts[j].c.ports[q]!.name)
    (hpop : _root_.Flock.HmOut.loadPublic ins.read.tags sts[j].c ins.read.I.publicFile = .ok pop) :
    pop.tables = none ∧ pop.n = (h.reg.sample (src c₀)).size * h.reg.ns.getD (templateOf h.ms u) 0 := by
  have hat := stmtOkAt_unit h hx hok hu
  unfold stmtOkAt at hat
  rw [hx] at hat
  simp only [Bool.and_eq_true] at hat
  obtain ⟨_, _, _, w, S, -, -, -, hw, hS, hc⟩ := readsOk_true h hat.2
  have hwi := ins_of_setupOf hw hset
  obtain ⟨-, hSH, hSr, hSn⟩ := stmtJsonOf_fields h hS
  have hH : headerOf ins.read.I.publicFile = some S.header := by
    rw [(ins_reads_public hwi).2]
    exact hSH
  obtain ⟨-, -, -, hpo, -, -, -, -, -, hrefs, -⟩ := Spot.SecondStage.stmtCodes_nil (List.isEmpty_iff.1 hc)
  obtain ⟨hdr, hhdr, hn, htab⟩ :=
    Depths.loadPublic_counts (FlockSoundness.Discharge.Hidden.loadPublic_nil hpop)
  rw [hH, Option.some.injEq] at hhdr
  subst hhdr
  have hwf := HeaderRoots.wf_of_headerOf hH
  have hnone : pop.tables = none := by
    cases ht : pop.tables with
    | none => rfl
    | some t =>
      exfalso
      obtain ⟨cs, hcs⟩ := htab t ht
      obtain ⟨sr, hsr, hall⟩ := Depths.sharedRows_names hcs
      have hmq : sts[j].c.ports[q]! ∈ sts[j].c.ports := by
        rw [getElem!_pos sts[j].c.ports q hq]
        exact Array.getElem_mem hq
      obtain ⟨v, hv⟩ := hall _ hmq
      obtain ⟨v', hmem⟩ := Depths.raw_mem hwf hsr hv
      simp only [Spot.SecondStage.refsOk] at hrefs
      obtain ⟨⟨nm, e⟩, hme, hnm⟩ := Array.any_eq_true'.1 (Array.all_eq_true_iff_forall_mem.1 hrefs _ hmem)
      have hnm' : nm = sts[j].c.ports[q]!.name := by simpa using hnm
      subst hnm'
      cases hreg : w.registered with
      | none =>
        rw [hSn hreg, show Spot.entries (Spot.getD Json.null "ports") = #[] from rfl] at hme
        exact absurd hme (Array.not_mem_empty _)
      | some b =>
        obtain ⟨jR, own, hjR, hoj, hown⟩ := Depths.ins_of_registered hwi hreg
        have hSj : S.registered = jR := by
          have h4 := hSr b hreg
          rw [hjR] at h4
          exact (Option.some.inj h4).symm
        obtain ⟨kvs, hports, -, hback⟩ := portsOf_entries (portsOf_of_ofJson hoj)
        rw [hSj, getD_of_ok hports] at hme
        obtain ⟨reg, hreg', hrn⟩ := hback _ hme
        exact hnr own hown reg hreg' hrn
  obtain ⟨m, hm, hmq⟩ := populationOk_count (q := (h.reg.sample (src c₀)).size)
    (p := partAt h (h.reg.sample (src c₀)).size (templateOf h.ms u)) hpo
  rw [Depths.raw_nat hwf hn, Option.some.injEq] at hm
  exact ⟨hnone, hm.trans hmq⟩

theorem publishedRows_of_runs (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h) : PublishedRows h := by
  intro r hr y rp hwm hl hno
  have W := hW r hr
  have hc1 := CheckedRun.one_table hr
  have hpos := CheckedRun.sts_pos hr
  have h0 : r.sts[0]! = r.sts[0] := getElem!_pos r.sts 0 hpos
  have hq : ∃ q, q < (memberCircuit h (templateOf h.ms r.u)).ports.size ∧
      (memberCircuit h (templateOf h.ms r.u)).ports[q]!.name = rp.row :=
    W.rows.2 (y, templateOf h.ms r.u, rp) hwm
  have hlt : portNamed h (templateOf h.ms r.u) rp.row < r.sts[0]!.c.ports.size := by
    rw [W.ports r hr]
    exact (portNamed_spec hq).2.1
  refine ⟨hc1.2, fun pop hpop => ?_⟩
  have hlt' : portNamed h (templateOf h.ms r.u) rp.row < r.sts[0].c.ports.size := by
    rw [← h0]
    exact hlt
  have hpop' : _root_.Flock.HmOut.loadPublic r.ins.read.tags r.sts[0].c r.ins.read.I.publicFile = .ok pop := by
    rw [← h0]
    exact hpop
  have hnr : ∀ own, r.ins.registered = some own → ∀ reg ∈ own,
      reg.name ≠ r.sts[0].c.ports[portNamed h (templateOf h.ms r.u) rp.row]!.name :=
    fun own hown reg hmem hn => regNamed_ne_none (by rw [h0]) hown hmem hn hno
  obtain ⟨Pt, d, spec, hx, hP, h2, hin, hok, hu, hck, hb, hρ, hrb, hset⟩ := hr
  obtain ⟨hnone, hn⟩ := unshared_of_stmtOk hx hok hu hset hpos hlt' hnr hpop'
  have hcnt : _root_.Flock.PortRoots.countOf pop (portNamed h (templateOf h.ms r.u) rp.row) = pop.n := by
    simp only [_root_.Flock.PortRoots.countOf, hnone]
  have hleaves : (pubPort r.sts[0]!.c pop (portNamed h (templateOf h.ms r.u) rp.row)).leaves =
      _root_.Flock.PortRoots.countOf pop (portNamed h (templateOf h.ms r.u) rp.row) := rfl
  refine ⟨hleaves.trans (hcnt.trans hn), ?_⟩
  exact congrArg
    (fun L : _root_.Flock.Bridge.LeafMap =>
      L.ref (portNamed h (templateOf h.ms r.u) rp.row) (unitOf r.ins.read.draw (r.n h)))
    (_root_.Flock.Bridge.LeafMap.ofPublic_none (c := r.sts[0]!.c) pop.n hnone)

theorem publishedRows_of_wiring (W : Wiring h) : PublishedRows h :=
  publishedRows_of_runs fun _ _ => W

end AddressRow

theorem rowFact_of_checks {h : Held} (hW : ∀ r : CheckedRun h.stage, r.Ok h → Wiring h)
    (hna : ∀ v, h.reg.always v = false) (hSS : AddressRow.SampleSlots h)
    (hwc : AddressRow.wiresChecked h = true) : RowFact h (wireMap h) (returned h) (addressOf h) :=
  rowFact_of_slots hW hna hSS hwc (AddressRow.publishedRows_of_runs hW)

end SpotFlock
