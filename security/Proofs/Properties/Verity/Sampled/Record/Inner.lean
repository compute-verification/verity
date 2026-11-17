module

public import Proofs.Properties.Verity.Sampled.Record.Opens
public import Proofs.Properties.Verity.Sampled.Record.Scope
public import Proofs.Flock.Claims.Recursion.AcceptedScope
public import Proofs.Flock.Claims.Bridge.Custody
public import Proofs.Flock.Claims.Scope.Setup
public import Proofs.Flock.Claims.KS.Range

/-! The sessions are in Flock's masked scope (`SessionsIn`, `InnerNumbers`). -/

@[expose] public section

namespace Verity.Sampled.Record

open Claims (Msg)
open FlockDeploy (Ins)
open Verity.Recursive.BaseSetup (bind_ok)

variable {A : Live} {o : Own}

theorem inScope_sess {d : Msg} {s : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (hins : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s)
    (hsu : s.setup = .ok (spec, sts)) : SessionInScope spec sts := by
  obtain ⟨t, -, -, -, -, hst, -⟩ := shape_sess hins hsu
  have hz : s.read.zk = true := FlockDeploy.Raw.ins_zk hins
  have hpi : s.read.publicInputs = none := FlockDeploy.Raw.ins_publicInputs hins
  obtain ⟨-, -, hp2, hr⟩ := RecDeploy.read_setup_in (FlockDeploy.Ins.setup_read hsu) hpi
  have hsh := Claims.Bridge.Custody.setup_shapeOk hsu hz
  refine ⟨by simp [hst], fun st hst' => ?_, fun st hst' => hr st hst', hp2.le⟩
  obtain ⟨j, hj, rfl⟩ := Array.mem_iff_getElem.1 (Array.mem_toList_iff.1 hst')
  have h1 : Flock.Zk.shapeOk sts[j] = .ok () := Flock.Readable.array_forM_ok hsh j hj
  unfold Flock.Zk.shapeOk at h1
  obtain ⟨_, hq, -⟩ := bind_ok h1
  exact FlockDeploy.KS.queries0_range hq

theorem ins_of_setupOf {d : Msg} (hfits : Fits A o d) {s : Ins} {spec : Flock.SessionSpec} {sts : Array Flock.Stmt}
    (h : FlockDeploy.setupOf true (sessStmt A o d) = some (s, spec, sts)) :
    ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins = .ok s ∧ s.setup = .ok (spec, sts) := by
  obtain ⟨hi, hsu⟩ := FlockDeploy.setupOf_some h
  refine ⟨?_, hsu⟩
  unfold FlockDeploy.insOf sessStmt at hi
  rw [Verity.Recursive.Part.decode_encode_fits hfits] at hi
  obtain ⟨w, hw, hs⟩ := Option.bind_eq_some_iff.1 hi
  obtain rfl := Option.some.inj hw
  split_ifs at hs with hz
  · cases hs
  change ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins.toOption = some s at hs
  cases hr : ({ baseRaw A o with draw := some ⟨d.toArray⟩ } : FlockDeploy.Raw).ins with
  | error e => rw [hr] at hs; cases hs
  | ok s' => rw [hr] at hs; cases hs; rfl

theorem N₀_none {x : Msg} (h : FlockDeploy.setupOf true x = none) : FlockDeploy.N₀ true x = 0 := by
  simp only [FlockDeploy.N₀, FlockDeploy.tables, h, List.map_nil, List.sum_nil]

theorem N₀_of_nums {x : Msg} {k : ℕ} {N : ℝ} (h : Nums x = (k, N)) : FlockDeploy.N₀ true x = N := by
  have h2 := congrArg Prod.snd h
  simpa only [Nums] using h2

theorem N₀_eq_of_nums {x y : Msg} (h : Nums x = Nums y) : FlockDeploy.N₀ true x = FlockDeploy.N₀ true y := by
  have h2 := congrArg Prod.snd h
  simpa only [Nums] using h2

theorem N₀_pos (h : ∃ d, RefOk A o d) : 0 < FlockDeploy.N₀ true (xS₀ A o) := by
  obtain ⟨_, t, -, hset, -⟩ : RefOk A o (d₀ A o) := Classical.epsilon_spec h
  rw [show xS₀ A o = sessStmt A o (d₀ A o) from rfl, N₀_of_nums (nums_one hset)]
  unfold FlockDeploy.tableLen
  positivity

theorem ref_in (h : ∃ d, RefOk A o d) :
    ∃ s spec sts, FlockDeploy.ZkIn (xS₀ A o) s spec sts ∧ sts.size ≤ 1 := by
  obtain ⟨s, t, hins, hset, -⟩ : RefOk A o (d₀ A o) := Classical.epsilon_spec h
  have hin := inScope_sess hins (FlockDeploy.setupOf_some hset).2
  exact ⟨s, _, _, zkIn_of_inScope hset hin, hin.1.le⟩

theorem kept_in (h : ∃ d, RefOk A o d) {d : Msg} (hn : numsOk A o d) (hf : Fits A o d) :
    ∃ s spec sts, FlockDeploy.ZkIn (sessStmt A o d) s spec sts ∧ sts.size ≤ 1 := by
  cases hsd : FlockDeploy.setupOf true (sessStmt A o d) with
  | none =>
    have hpos := N₀_pos h
    rw [← N₀_eq_of_nums hn, N₀_none hsd] at hpos
    exact absurd hpos (lt_irrefl _)
  | some p =>
    obtain ⟨s, spec, sts⟩ := p
    obtain ⟨hins, hsu⟩ := ins_of_setupOf hf hsd
    have hin := inScope_sess hins hsu
    exact ⟨s, spec, sts, zkIn_of_inScope hsd hin, hin.1.le⟩

theorem sessionOf_fst (v : Claims.Seen) :
    (sessionOf A o v).1 = xS₀ A o ∨ ∃ d, numsOk A o d ∧ Fits A o d ∧ (sessionOf A o v).1 = sessStmt A o d := by
  unfold sessionOf
  split
  · split_ifs with h
    · exact .inr ⟨_, h.1, h.2, by simp only⟩
    · exact .inl (by simp only [refused])
  · exact .inl (by simp only [refused])

theorem code_tables (e : Ext) (x : Msg) : (code A o e).tables x = 1 := rfl

theorem code_innerN₀ (e : Ext) (x : Msg) : (code A o e).innerN₀ x = FlockDeploy.N₀ true (xS₀ A o) := by
  simp only [code]

theorem code_unitN₀ (e : Ext) (x : Msg) (u : ℕ) : (code A o e).unitN₀ x u = FlockDeploy.N₀ true (xS₀ A o) := by
  simp only [code]

theorem sessionsIn (e : Ext) (h : ∃ d, RefOk A o d) : SamplingDeploy.SessionsIn (code A o e) := by
  intro x u v xS vS hs
  change (if x = xOf A o then some (sessionOf A o v) else none) = some (xS, vS) at hs
  rw [code_tables, code_unitN₀]
  by_cases hx : x = xOf A o
  swap
  · rw [ite_eq_right hx] at hs
    cases hs
  rw [ite_eq_left hx] at hs
  have hxS : xS = (sessionOf A o v).1 := (congrArg Prod.fst (Option.some.inj hs)).symm
  subst hxS
  rcases sessionOf_fst (A := A) (o := o) v with h1 | ⟨d, hn, hf, h1⟩
  · rw [h1]
    exact ⟨ref_in h, rfl⟩
  · rw [h1]
    exact ⟨kept_in h hn hf, N₀_eq_of_nums hn⟩

theorem sessionsIn_atInnerN₀ {c : SamplingDeploy.Code} (h : SamplingDeploy.SessionsIn c)
    (hu : ∀ x u, c.unitN₀ x u = c.innerN₀ x) : SamplingDeploy.SessionsIn c.atInnerN₀ :=
  fun x u v xS vS hs => ⟨(h x u v xS vS hs).1, (h x u v xS vS hs).2.trans (hu x u)⟩

theorem innerNumbers (e : Ext) (h : ∃ d, RefOk A o d) : Claims.Sampling.InnerNumbers (record A o e) :=
  SamplingDeploy.innerNumbers_deploy _root_.Flock.Readable.relation (code A o e)
    (sessionsIn_atInnerN₀ (sessionsIn e h) fun x u => (code_unitN₀ e x u).trans (code_innerN₀ e x).symm)

end Verity.Sampled.Record
